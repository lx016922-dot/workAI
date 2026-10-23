"""Immutable spatial/time reference checks, with resumable job bookkeeping.

This runs local reference FEM only. A failed budget check is recorded, never
silently relaxed or retried. No new weights are fitted to these checks.
"""
import argparse,copy,hashlib,json,shutil,subprocess,sys,threading,uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from operator3d.train import atomic,now,coefficients,relative
from solid.mesh import prepare_mesh,locate
from .evaluate import reference,digest

def compare(root,campaign,left,right):
    a,ra,_=reference(root,campaign,left);b,rb,_=reference(root,campaign,right)
    ma=prepare_mesh(np.asarray(ra['nodes_mm'])*.001,ra['tetra']);mb=prepare_mesh(np.asarray(rb['nodes_mm'])*.001,rb['tetra']);times=np.array(ra['times_us']);assert ra['times_us']==rb['times_us'];rows=[]
    for rx in left['scene']['receivers']:
        ca,wa=locate(ma,rx['point_mm']);cb,wb=locate(mb,rx['point_mm']);va=np.einsum('takc,a->tkc',a[:,ma['tetra'][ca]],wa);vb=np.einsum('takc,a->tkc',b[:,mb['tetra'][cb]],wb);direction=np.asarray(rx['direction']);pa=va@direction;pb=vb@direction;ac=coefficients(times,pa[:,2],2*left['scene']['source']['frequency_mhz']);bc=coefficients(times,pb[:,2],2*left['scene']['source']['frequency_mhz'])
        rows.append({'name':rx['name'],'full_vector_relative_l2':relative(va.sum(1),vb.sum(1)),'material_vector_relative_l2':relative(va[:,2],vb[:,2]),'full_direction_relative_l2':relative(pa.sum(1),pb.sum(1)),'material_direction_relative_l2':relative(pa[:,2],pb[:,2]),'material_2f_complex_relative_error':float(abs(ac-bc)/abs(bc)) if abs(bc)>1e-13 else None,'finer_material_2f_nm':float(abs(bc)*1e9)})
    keys=['full_vector_relative_l2','material_vector_relative_l2','full_direction_relative_l2','material_direction_relative_l2','material_2f_complex_relative_error']
    same_mesh=np.array_equal(ma['points'],mb['points']) and np.array_equal(ma['tetra'],mb['tetra']);fields=None
    if same_mesh:
        from operator3d.data import tensors
        mass=tensors(left['scene'],ma)['mass'].numpy()[None,:,None];fields={'full_relative_l2':relative(a.sum(2),b.sum(2),mass),'material_relative_l2':relative(a[:,:,2],b[:,:,2],mass)}
    return {'left_controls':left['controls'],'right_controls':right['controls'],'left_elements':len(ma['tetra']),'right_elements':len(mb['tetra']),'left_step_us':ra['diagnostics']['internal_step_us'],'right_step_us':rb['diagnostics']['internal_step_us'],'same_mesh':same_mesh,'rows':rows,'threshold':.05,'all_probe_checks_pass':all(r[k] is not None and r[k]<=.05 for r in rows for k in keys),'same_mesh_field_checks':fields,'scope':'Two retained receivers, all vector components and requested projections, full record including reflections. A spatial pair is not a continuous-domain or arbitrary-point error certificate.'}

def main(args):
    out=args.output;lock=threading.RLock();source=json.loads(args.campaign.read_text());assert source['complete'];source_sha=digest(args.campaign)
    if args.resume:
        manifest=json.loads((out/'manifest.json').read_text());assert manifest['source_campaign_sha256']==source_sha and manifest['refined_h_mm']==args.h
    else:
        out.mkdir(parents=True,exist_ok=False);shutil.copytree(args.campaign.parent/'source_snapshot',out/'source_snapshot');shutil.copyfile(Path(__file__),out/'source_snapshot/modal-refine.py');cases=[]
        for original in source['cases']:
            if original['split']!='test':continue
            plans=[]
            for label,cfl in [('space',original['scene']['numerics']['cfl']),('time',original['scene']['numerics']['cfl']/2)]:
                scene=copy.deepcopy(original['scene']);scene['numerics'].update(h_mm=args.h,cfl=cfl)
                plans.append({'level':label,'scene':scene,'controls':[{'role':role,'job_id':'solid-'+uuid.uuid4().hex} for role in ['linear','geometric','material']]})
            cases.append({'case_id':original['case_id'],'original':original,'levels':plans,'phase':'pending'})
        manifest={'schema':'workai.modal_reference_refinement.v1','phase':'planned','source_campaign_sha256':source_sha,'refined_h_mm':args.h,'threshold':.05,'started_at_utc':now(),'workers':args.workers,'cases':cases,'accepted_for_solver_routing':False};atomic(out/'manifest.json',manifest)
    def publish():
        with lock:manifest['updated_at_utc']=now();atomic(out/'manifest.json',manifest)
    def work(case):
        if case['phase'] in ['completed','failed']:return
        case['phase']='running';publish()
        try:
            for li,level in enumerate(case['levels']):
                for index,(control,law) in enumerate(zip(level['controls'],['linear','stvk','murnaghan'])):
                    job=args.root/'solid/jobs'/control['job_id']
                    if 'result_sha256' in control:
                        assert digest(job/'result.json')==control['result_sha256'];continue
                    if job.exists():
                        status=json.loads((job/'status.json').read_text()) if (job/'status.json').exists() else {}
                        if status.get('phase')=='completed':control['result_sha256']=digest(job/'result.json');assert control['result_sha256']==status['result_sha256'];publish();continue
                        raise RuntimeError('Prior job requires inspection; not automatically retried: '+job.name)
                    job.mkdir();scene=copy.deepcopy(level['scene']);scene['constitutive']=law;atomic(job/'scene.json',scene);publish()
                    if index:mesh_ref=args.root/'solid/jobs'/level['controls'][0]['job_id'];scope='controls'
                    elif li:mesh_ref=args.root/'solid/jobs'/case['levels'][0]['controls'][0]['job_id'];scope='time'
                    else:mesh_ref=None;scope='controls'
                    command=[str(args.fem_python),'-c','from solid.worker import run;import sys;run(sys.argv[1],sys.argv[2] or None,sys.argv[3],"float64")',str(job),str(mesh_ref) if mesh_ref else '',scope]
                    with (job/'worker.log').open('x') as stream:subprocess.run(command,cwd=out/'source_snapshot',stdout=stream,stderr=subprocess.STDOUT,check=True)
                    status=json.loads((job/'status.json').read_text());assert status['phase']=='completed';control['result_sha256']=digest(job/'result.json');assert control['result_sha256']==status['result_sha256'];publish()
            case['spatial_comparison']=compare(args.root,args.campaign,case['original'],case['levels'][0]);case['time_comparison']=compare(args.root,args.campaign,case['levels'][0],case['levels'][1]);case['phase']='completed';publish();print(json.dumps({'case_id':case['case_id'],'space_pass':case['spatial_comparison']['all_probe_checks_pass'],'time_pass':case['time_comparison']['all_probe_checks_pass']}),flush=True)
        except Exception as error:
            case['phase']='failed';case['error']=str(error);publish();print(json.dumps({'case_id':case['case_id'],'error':str(error)}),flush=True)
    manifest['phase']='running';publish()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(work,manifest['cases']))
    manifest['phase']='completed' if all(c['phase']=='completed' for c in manifest['cases']) else 'completed_with_failures';manifest['all_probe_checks_pass']=all(c['phase']=='completed' and c['spatial_comparison']['all_probe_checks_pass'] and c['time_comparison']['all_probe_checks_pass'] for c in manifest['cases']);publish();print(json.dumps({'phase':manifest['phase'],'all_probe_checks_pass':manifest['all_probe_checks_pass']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/workspace/workai-remote'));p.add_argument('--fem-python',type=Path,default=Path('/workspace/workai-remote/solid/venv/bin/python'));p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--h',type=float,default=.4);p.add_argument('--workers',type=int,choices=[1,2],default=2);p.add_argument('--resume',action='store_true');main(p.parse_args())
