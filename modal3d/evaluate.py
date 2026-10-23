"""Frozen hybrid prediction against SHA-bound, previously withheld FEM fields.

Predictions for test geometries are written before their transient labels are
read. Classical modal preparation and propagation are timed separately.
"""
import argparse, gzip, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from solid.mesh import prepare_mesh, locate
from operator3d.data import tensors, check_splits
from operator3d.train import atomic, now, relative, coefficients
from operator3d.physics import weak_residual, U0
from .basis import build_basis
from .response import response

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def reference(root, campaign, row):
    fields=[]; first=None; bindings=[]
    assert [c['role'] for c in row['controls']]==['linear','geometric','material']
    for control, law in zip(row['controls'],['linear','stvk','murnaghan']):
        path=root/'solid/jobs'/control['job_id']; result=json.loads((path/'result.json').read_text()); status=json.loads((path/'status.json').read_text())
        assert status['phase']=='completed' and digest(path/'result.json')==control['result_sha256']==status['result_sha256']
        expected=dict(row['scene']);expected['constitutive']=law
        assert expected==result['scene'] and result['job_id']==control['job_id'] and digest(path/'scene.json')==result['scene_sha256']
        for name, sha in result['source_sha256'].items():
            assert digest(campaign.parent/'source_snapshot/solid'/name)==sha
        packed=(path/'field.bin.gz').read_bytes();raw=gzip.decompress(packed);spec=result['field']
        assert spec['dtype']=='little_endian_float64' and hashlib.sha256(packed).hexdigest()==spec['sha256'] and hashlib.sha256(raw).hexdigest()==spec['uncompressed_sha256']
        assert len(packed)==spec['compressed_bytes'] and len(raw)==spec['uncompressed_bytes']
        values=np.frombuffer(raw,dtype='<f8').reshape(spec['shape']);assert np.isfinite(values).all()
        if first:
            for key in ['nodes_mm','tetra','times_us']:assert result[key]==first[key]
            assert result['diagnostics']['internal_step_us']==first['diagnostics']['internal_step_us']
        else:first=result
        fields.append(values);bindings.append(control|{'field_sha256':spec['sha256'],'seconds':status['elapsed_s']})
    return np.stack([fields[0],fields[1]-fields[0],fields[2]-fields[1]],2), first, bindings

def physics(predicted, times, mesh, scene, scales):
    case=tensors(scene,mesh);checks=[];delta=float(times[1]-times[0]);fixed=case['fixed'];mass=case['mass'];boundary=case['boundary_nodes']&~fixed;interior=(~case['boundary_nodes'])&~fixed
    with torch.no_grad():
        for frame in np.unique(np.linspace(1,len(times)-2,9,dtype=int)):
            p=torch.tensor(predicted[frame-1:frame+2]/U0,dtype=torch.float64)
            residual,strain,jac=weak_residual(*p,torch.tensor(times[frame]),delta,case)
            normalizer=torch.tensor(scales)[:,None]*(2*np.pi*scene['source']['frequency_mhz'])**2
            squared=((residual/normalizer)**2).mean((1,2))
            def rms(mask):return float(torch.sqrt((squared[mask]*mass[mask]).sum()/mass[mask].sum())) if mask.any() else 0.
            checks.append({'time_us':float(times[frame]),'interior_rms':rms(interior),'boundary_rms':rms(boundary),'max_strain':float(torch.linalg.matrix_norm(strain).max()),'min_J':float(jac.min())})
    guard={'max_gradient':0.,'max_green_strain':0.,'min_J':float('inf')}
    for start in range(0,len(times),24):
        total=predicted[start:start+24].sum(2);g=np.einsum('teai,eaj->teij',total[:,mesh['tetra']],mesh['grad']);f=np.eye(3)+g;e=(np.swapaxes(f,-1,-2)@f-np.eye(3))*.5
        guard['max_gradient']=max(guard['max_gradient'],float(np.linalg.norm(g,axis=(-2,-1)).max()));guard['max_green_strain']=max(guard['max_green_strain'],float(np.linalg.norm(e,axis=(-2,-1)).max()));guard['min_J']=min(guard['min_J'],float(np.linalg.det(f).min()))
    guard['fixed_nodes_zero']=bool(np.all(predicted[:,fixed.numpy()]==0));guard['initial_displacement_zero']=bool(np.all(predicted[0]==0));return checks,guard

def score(row,predicted,truth,times,mesh,scales,saved):
    scene=row['scene'];case=tensors(scene,mesh);mass=case['mass'].numpy()[None,:,None];probes=[]
    for i,rx in enumerate(scene['receivers']):
        cell,weights=locate(mesh,rx['point_mm']);nodes=mesh['tetra'][cell];pv=np.einsum('takc,a->tkc',predicted[:,nodes],weights);tv=np.einsum('takc,a->tkc',truth[:,nodes],weights);direction=np.asarray(rx['direction']);direction/=np.linalg.norm(direction);p=pv@direction;t=tv@direction;pc=coefficients(times,p[:,2],2*scene['source']['frequency_mhz']);tc=coefficients(times,t[:,2],2*scene['source']['frequency_mhz']);above=abs(tc)>1e-13
        probes.append({'name':rx['name'],'point_mm':rx['point_mm'],'direction':direction.tolist(),'full_waveform_relative_l2':relative(p.sum(1),t.sum(1)),'material_waveform_relative_l2':relative(p[:,2],t[:,2]),'material_2f_predicted_nm':float(abs(pc)*1e9),'material_2f_fem_nm':float(abs(tc)*1e9),'material_2f_complex_relative_error':float(abs(pc-tc)/abs(tc)) if above else None,'material_2f_above_reporting_floor':bool(above),'window_us':[float(times[0]),float(times[-1])],'includes_reflections':True})
        key=row['case_id']+'_P'+str(i+1);saved[key+'_prediction']=pv;saved[key+'_fem']=tv
    saved[row['case_id']+'_times_us']=times
    checks,guard=physics(predicted,times,mesh,scene,scales)
    return {'case_id':row['case_id'],'split':row['split'],'geometry':scene['geometry'],'source':scene['source'],'boundary':scene['boundary'],'full_field_relative_l2':relative(predicted.sum(2),truth.sum(2),mass),'parts_field_relative_l2':{name:relative(predicted[:,:,k],truth[:,:,k],mass) for k,name in enumerate(['linear','geometric_increment','material_increment'])},'probes':probes,'sampled_physics':checks,'all_saved_frame_guard':guard}

def main(args):
    torch.set_num_threads(2);start=time.monotonic();run=args.run;weights=json.loads((run/'energy-layer.json').read_text());assert digest(run/'best.pt')==weights['checkpoint_sha256'];alpha=np.array(weights['multipliers']);development=json.loads(args.dataset.read_text());assert digest(args.dataset)==weights['dataset_sha256'];campaign=json.loads(args.campaign.read_text());assert check_splits(campaign)=={'train':48,'validation':8,'test':8};scales=development['labels']['branch_scales_in_100nm']
    atomic(run/'frozen-checkpoint-before-test.json',{'sha256':weights['checkpoint_sha256'],'step':weights['selected_step'],'frozen_at_utc':weights['frozen_at_utc'],'test_labels_used_for_selection':False,'evaluation_started_at_utc':now()});rows=[];saved={};field_dir=run/'test-fields';field_dir.mkdir(exist_ok=False)
    for row in development['cases']:
        path=args.dataset.parent/row['file'];assert digest(path)==row['sha256']
        with np.load(path,allow_pickle=False) as data:
            predicted=np.concatenate([data['base_parts_m'],np.einsum('tnkc,k->tnc',data['material_features_m'],alpha)[:,:,None,:]],2);truth=data['reference_parts_m'];times=data['times_us'];mesh=prepare_mesh(data['points_mm']*.001,data['tetra'])
        result=score(row,predicted,truth,times,mesh,scales,saved);result['timing']={'basis_seconds':row['preparation_seconds'],'basis_response_seconds':row['response_seconds'],'timing_scope':'cached physical preparation; learned contraction negligible but not separately timed'};rows.append(result)
    for row in campaign['cases']:
        if row['split']!='test':continue
        # Mesh is geometry input only. No transient field is opened here.
        directory=args.root/'solid/jobs'/row['controls'][0]['job_id']
        with np.load(directory/'mesh.npz',allow_pickle=False) as data:mesh=prepare_mesh(data['points'],data['tetra'])
        basis=build_basis(mesh,row['scene'],development['modes']);prediction=response(mesh,row['scene'],basis,development['step_us'],True);predicted=prediction['parts_m'].copy();coeff=prediction['material_coefficients']*alpha;phi=basis['phi'].reshape(-1,3,development['modes']);predicted[:,:,2]=np.einsum('tkm,k,ncm->tnc',prediction['modal_basis_responses'][:,2:],coeff,phi)+np.einsum('k,ktd->td',coeff,prediction['static_basis_responses'][1:]).reshape(len(predicted),-1,3);times=prediction['times_us']
        file=field_dir/(row['case_id']+'.npz');np.savez_compressed(file,parts_m=predicted,times_us=times,points_mm=mesh['points']*1000,tetra=mesh['tetra'],surface=mesh['surface']);prediction_sha=digest(file);before_labels=now()
        truth,record,bindings=reference(args.root,args.campaign,row);assert digest(directory/'mesh.npz')==record['mesh_sha256'];np.testing.assert_array_equal(mesh['points']*1000,record['nodes_mm']);np.testing.assert_array_equal(times,record['times_us'])
        result=score(row,predicted,truth,times,mesh,scales,saved);result.update(reference_controls=bindings,field_file=str(file.relative_to(run)),field_sha256=prediction_sha,prediction_saved_before_label_load_utc=before_labels,timing={'basis_seconds':basis['preparation_seconds'],'response_seconds':prediction['response_seconds'],'reference_material_job_seconds':bindings[2]['seconds'],'includes_classical_FEM_assembly_and_modes':True,'reference_threads':1,'prediction_BLAS_threads':2});rows.append(result);atomic(run/'evaluation-progress.json',{'completed_test_cases':sum(r['split']=='test' for r in rows),'last_case':result});print(json.dumps({'case_id':row['case_id'],'full_error':result['full_field_relative_l2'],'material_error':result['parts_field_relative_l2']['material_increment'],'max_2f_error':max((p['material_2f_complex_relative_error'] or 0) for p in result['probes']),'max_residual':max(max(p['interior_rms'],p['boundary_rms']) for p in result['sampled_physics'])}),flush=True)
    np.savez_compressed(run/'probe-predictions.npz',**saved);test=[r for r in rows if r['split']=='test'];threshold=.05
    gates={'held_out_full_fields':all(r['full_field_relative_l2'] is not None and r['full_field_relative_l2']<=threshold for r in test),'held_out_material_increment':all(r['parts_field_relative_l2']['material_increment'] is not None and r['parts_field_relative_l2']['material_increment']<=threshold for r in test),'held_out_material_2f':all(p['material_2f_complex_relative_error'] is not None and p['material_2f_complex_relative_error']<=threshold for r in test for p in r['probes']),'sampled_weak_physics':all(max(p['interior_rms'],p['boundary_rms'])<=threshold for r in test for p in r['sampled_physics']),'strain_and_positive_J':all(r['all_saved_frame_guard']['max_green_strain']<=.001 and r['all_saved_frame_guard']['max_gradient']<=.001 and r['all_saved_frame_guard']['min_J']>0 and r['all_saved_frame_guard']['fixed_nodes_zero'] and r['all_saved_frame_guard']['initial_displacement_zero'] for r in test),'label_discretization_verified':False}
    metrics={'schema':'workai.modal_energy_evaluation.v1','kind':'modal_energy','cases':rows,'acceptance':{'threshold':threshold,**gates,'all_gates_passed':all(gates.values())},'accepted_for_solver_routing':False,'test_used_for_checkpoint_selection':False,'final_test_evaluated':True,'selected_step':weights['selected_step'],'checkpoint_sha256':weights['checkpoint_sha256'],'dataset_sha256':weights['dataset_sha256'],'evaluation_campaign_sha256':digest(args.campaign),'completed_at_utc':now(),'elapsed_s':time.monotonic()-start,'material_harmonic_scope':'Full-record Hann complex difference, including reflections; no separated nonlinear guided mode.','physics_scope':'9 predeclared saved times; .05 us central difference, original full Murnaghan Piola; P1 nodal weak residual, not continuous-domain proof.','model_scope':'Classical weakly nonlinear modal propagation plus 3 learned objective-energy couplings. Not a pure neural operator or arbitrary-shape FEM replacement.'}
    assert digest(run/'best.pt')==weights['checkpoint_sha256'];atomic(run/'metrics.json',metrics);status=json.loads((run/'status.json').read_text());status.update(status='completed_experimental',updated_at_utc=now(),evaluation_completed=True,acceptance=metrics['acceptance']);atomic(run/'status.json',status);print(json.dumps({'complete':True,'acceptance':metrics['acceptance']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/workspace/workai-remote'));p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);main(p.parse_args())
