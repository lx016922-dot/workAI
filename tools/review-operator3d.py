"""Reload a frozen 3-D checkpoint and reproduce held-out field/probe metrics."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(args):
    run=args.run;out=args.output;out.mkdir(parents=True,exist_ok=False);config=json.loads((run/'config.json').read_text());metrics=json.loads((run/'metrics.json').read_text());snapshot=run/'source_snapshot'
    for name,sha in config['source_sha256'].items():assert hashlib.sha256((snapshot/'operator3d'/name).read_bytes()).hexdigest()==sha
    assert hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()==metrics['checkpoint_sha256'];assert hashlib.sha256((run/'dataset.json').read_bytes()).hexdigest()==config['dataset_sha256']==metrics['dataset_sha256']
    frozen=json.loads((run/'frozen-checkpoint-before-test.json').read_text());assert frozen['sha256']==metrics['checkpoint_sha256'] and not frozen['test_labels_used_for_selection']
    sys.path.insert(0,str(snapshot));from operator3d.data import load_dataset
    from operator3d import model as model_module
    from operator3d.train import predict_case,relative
    from operator3d.physics import U0
    torch.set_num_threads(2);cases,scale,description=load_dataset(args.root,args.campaign);assert description['campaign_sha256']==json.loads((run/'dataset.json').read_text())['campaign_sha256']
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=True);np.testing.assert_allclose(checkpoint['scales'],scale,rtol=1e-13)
    if hasattr(model_module,'make_model'):model=model_module.make_model(scale,checkpoint['width'],checkpoint.get('architecture','baseline'))
    else:model=model_module.GeometryConditionedField(scale,checkpoint['width'])
    model.load_state_dict(checkpoint['model_state']);model.eval()
    cases=[c for c in cases if c['split']=='test'];predictions=np.load(run/'probe-predictions.npz',allow_pickle=False);fig,axes=plt.subplots(len(cases),2,figsize=(12,2.6*len(cases)),squeeze=False);verified=[]
    for i,case in enumerate(cases):
        prediction=predict_case(model,case);truth=case['parts'];row=next(r for r in metrics['cases'] if r['case_id']==case['case_id']);mass=case['mass'].numpy()[None,:,None]
        np.testing.assert_allclose(relative(prediction.sum(2),truth.sum(2),mass),row['full_field_relative_l2'],rtol=1e-11,atol=1e-13)
        cell,weights=case['receivers'][0];nodes=case['tetra'][cell].numpy();pv=np.einsum('takc,a->tkc',prediction[:,nodes],weights)*U0;tv=np.einsum('takc,a->tkc',truth[:,nodes],weights)*U0
        np.testing.assert_allclose(pv,predictions[case['case_id']+'_P1_prediction'],rtol=1e-10,atol=1e-20);np.testing.assert_allclose(tv,predictions[case['case_id']+'_P1_fem'],rtol=1e-12,atol=1e-22)
        for rx_index,((cell_rx,weights_rx),receiver) in enumerate(zip(case['receivers'],case['scene']['receivers'])):
            nodes_rx=case['tetra'][cell_rx].numpy();pr=np.einsum('takc,a->tkc',prediction[:,nodes_rx],weights_rx)*U0;tr=np.einsum('takc,a->tkc',truth[:,nodes_rx],weights_rx)*U0;name=case['case_id']+'_P'+str(rx_index+1)
            np.testing.assert_allclose(pr,predictions[name+'_prediction'],rtol=1e-10,atol=1e-20);np.testing.assert_allclose(tr,predictions[name+'_fem'],rtol=1e-12,atol=1e-22)
            direction=np.asarray(receiver['direction']);pr=pr@direction;tr=tr@direction;metric=row['probes'][rx_index]
            np.testing.assert_allclose(np.linalg.norm(pr.sum(1)-tr.sum(1))/np.linalg.norm(tr.sum(1)),metric['full_waveform_relative_l2'],rtol=1e-11,atol=1e-13)
            times=case['times'];window=np.hanning(len(times));wave=np.exp(-4j*np.pi*case['scene']['source']['frequency_mhz']*times)
            def coefficient(signal):return 2*np.sum(window*wave*(signal-np.sum(window*signal)/window.sum()))/window.sum()
            pc,tc=coefficient(pr[:,2]),coefficient(tr[:,2]);np.testing.assert_allclose([abs(pc)*1e9,abs(tc)*1e9],[metric['material_2f_predicted_nm'],metric['material_2f_fem_nm']],rtol=1e-10,atol=1e-14)
            if metric['material_2f_above_reporting_floor']:np.testing.assert_allclose(abs(pc-tc)/abs(tc),metric['material_2f_complex_relative_error'],rtol=1e-10,atol=1e-12)
        for j,(p,t,label) in enumerate([(pv.sum(1)[:,0],tv.sum(1)[:,0],'Total X displacement / nm'),(pv[:,2,0],tv[:,2,0],'Material increment X / nm')]):
            ax=axes[i,j];ax.plot(case['times'],t*1e9,label='FEM',color='#245c80',lw=1.3);ax.plot(case['times'],p*1e9,label='AI (experimental)',color='#cb6a45',ls='--',lw=1.1);ax.set_title(case['case_id']+' / P1',fontsize=10);ax.set_xlabel('Time / us');ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=8)
        verified.append({'case_id':case['case_id'],'full_field_metric_reproduced':True,'probe_prediction_and_fem_reproduced':True,'both_receiver_directions_and_material_2f_reproduced':True})
    fig.suptitle('Frozen 3-D model: previously unseen geometries\nFull traces include reflections; predictions are not accepted for solver routing',fontsize=12);fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(out/'held-out-waveforms.png',dpi=140);fig.savefig(out/'held-out-waveforms.pdf');plt.close(fig)
    result={'passed':True,'checkpoint_sha256':metrics['checkpoint_sha256'],'selected_step':metrics['selected_step'],'frozen_source_used':True,'cases':verified,'reproduction_not_independent_physics_validation':True,'new_training_steps':0};(out/'reload-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/workspace/workai-remote'));p.add_argument('--run',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
