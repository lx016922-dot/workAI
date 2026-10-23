"""Reload frozen learned weights and reproduce saved independent-test metrics."""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
import numpy as np

def main(args):
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();run=args.run;sys.path.insert(0,str(run/'source_snapshot'))
    import torch
    from modal3d.energy import EnergyCoupling
    from modal3d.basis import build_basis
    from modal3d.response import response
    from solid.mesh import prepare_mesh,locate
    from operator3d.data import tensors
    from operator3d.train import relative,coefficients
    config=json.loads((run/'config.json').read_text());receipt=json.loads((run/'energy-layer.json').read_text());metrics=json.loads((run/'metrics.json').read_text());campaign=json.loads(args.campaign.read_text());dataset=json.loads(args.dataset.read_text());assert digest(run/'best.pt')==receipt['checkpoint_sha256']==metrics['checkpoint_sha256'];assert digest(args.dataset)==receipt['dataset_sha256'];assert digest(args.campaign)==metrics['evaluation_campaign_sha256']
    for name,sha in config['source_sha256'].items():assert digest(run/'source_snapshot/modal3d'/name)==sha
    for row in dataset['cases']:assert digest(args.dataset.parent/row['file'])==row['sha256']
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=True);model=EnergyCoupling();model.load_state_dict(checkpoint['model_state']);alpha=model().detach().numpy();np.testing.assert_array_equal(alpha,receipt['multipliers']);saved=np.load(run/'probe-predictions.npz',allow_pickle=False);checked=[]
    for row in [r for r in metrics['cases'] if r['split']=='test']:
        physical=next(c for c in campaign['cases'] if c['case_id']==row['case_id']);scene=physical['scene'];path=run/row['field_file'];assert digest(path)==row['field_sha256']
        with np.load(path,allow_pickle=False) as d:pred=d['parts_m'];times=d['times_us'];mesh=prepare_mesh(d['points_mm']*.001,d['tetra'])
        fields=[]
        for c in physical['controls']:
            directory=args.root/'solid/jobs'/c['job_id'];assert digest(directory/'result.json')==c['result_sha256'];r=json.loads((directory/'result.json').read_text());assert json.loads((directory/'status.json').read_text())['result_sha256']==c['result_sha256'];raw=gzip.decompress((directory/'field.bin.gz').read_bytes());assert digest(directory/'field.bin.gz')==r['field']['sha256'] and hashlib.sha256(raw).hexdigest()==r['field']['uncompressed_sha256'];fields.append(np.frombuffer(raw,dtype='<f8').reshape(r['field']['shape']))
        truth=np.stack([fields[0],fields[1]-fields[0],fields[2]-fields[1]],2);mass=tensors(scene,mesh)['mass'].numpy()[None,:,None];np.testing.assert_allclose(relative(pred.sum(2),truth.sum(2),mass),row['full_field_relative_l2'],rtol=1e-11,atol=1e-15);np.testing.assert_allclose(relative(pred[:,:,2],truth[:,:,2],mass),row['parts_field_relative_l2']['material_increment'],rtol=1e-11,atol=1e-15)
        for i,rx in enumerate(scene['receivers']):
            cell,w=locate(mesh,rx['point_mm']);p=np.einsum('tnkc,n->tkc',pred[:,mesh['tetra'][cell]],w);prefix=row['case_id']+'_P'+str(i+1);np.testing.assert_allclose(p,saved[prefix+'_prediction'],rtol=1e-12,atol=1e-23);t=np.einsum('tnkc,n->tkc',truth[:,mesh['tetra'][cell]],w);direction=np.array(rx['direction']);pc=coefficients(times,p[:,2]@direction,2*scene['source']['frequency_mhz']);tc=coefficients(times,t[:,2]@direction,2*scene['source']['frequency_mhz']);np.testing.assert_allclose(abs(pc-tc)/abs(tc),row['probes'][i]['material_2f_complex_relative_error'],rtol=1e-10,atol=1e-14)
        if not checked:
            input_directory=args.root/'solid/jobs'/physical['controls'][0]['job_id'];input_record=json.loads((input_directory/'result.json').read_text());assert digest(input_directory/'mesh.npz')==input_record['mesh_sha256']
            # Reuse the exact SI mesh input. A millimetre round trip can change
            # eigenvectors by a few ulps and is not the original solver input.
            with np.load(input_directory/'mesh.npz',allow_pickle=False) as original:input_mesh=prepare_mesh(original['points'],original['tetra'])
            np.testing.assert_array_equal(input_mesh['points']*1000,input_record['nodes_mm'])
            basis=build_basis(input_mesh,scene,config['modes']);again=response(input_mesh,scene,basis,config['step_us'],True);coeff=again['material_coefficients']*alpha;phi=basis['phi'].reshape(-1,3,config['modes']);replayed=again['parts_m'].copy();replayed[:,:,2]=np.einsum('tkm,k,ncm->tnc',again['modal_basis_responses'][:,2:],coeff,phi)+np.einsum('k,ktd->td',coeff,again['static_basis_responses'][1:]).reshape(len(times),-1,3);np.testing.assert_allclose(replayed,pred,rtol=1e-8,atol=1e-20)
        checked.append(row['case_id'])
    output={'passed':True,'checkpoint_sha256':receipt['checkpoint_sha256'],'trainable_parameters':sum(p.numel() for p in model.parameters()),'verified_development_cache_files':len(dataset['cases']),'test_fields_and_metrics_reproduced':checked,'one_prediction_recomputed_from_frozen_source':checked[0],'new_FEM_solves':0,'new_training_steps':0};args.output.write_text(json.dumps(output,indent=2));print(json.dumps(output))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/workspace/workai-remote'));p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
