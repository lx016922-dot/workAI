"""Run bounded physical geometry cases and record mesh/time comparisons.

This explicitly starts local finite-element solves, never COMSOL or training.
"""
import argparse,gzip,hashlib,json,sys,uuid
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from solid.schema import default_scene,validate_scene
from solid.worker import run,write_json
from solid.mesh import prepare_mesh,locate

def load(job):
    result_bytes=(job/'result.json').read_bytes();status=json.loads((job/'status.json').read_text())
    assert status['phase']=='completed' and hashlib.sha256(result_bytes).hexdigest()==status['result_sha256']
    r=json.loads(result_bytes);packed=(job/'field.bin.gz').read_bytes();assert hashlib.sha256(packed).hexdigest()==r['field']['sha256']
    raw=gzip.decompress(packed);assert hashlib.sha256(raw).hexdigest()==r['field']['uncompressed_sha256']
    return r,np.frombuffer(raw,dtype='<f4').reshape(r['field']['shape']),prepare_mesh(np.asarray(r['nodes_mm'])*.001,r['tetra'])

def compare(first,second):
    a,fa,ma=load(first);b,fb,mb=load(second)
    sa=json.loads(json.dumps(a['scene']));sb=json.loads(json.dumps(b['scene']));sa.pop('name');sb.pop('name');sa.pop('numerics');sb.pop('numerics');assert sa==sb
    for name in ['fem.py','mesh.py','schema.py']:assert a['source_sha256'][name]==b['source_sha256'][name]
    rows=[]
    for rx in a['scene']['receivers']:
        i,wa=locate(ma,rx['point_mm']);j,wb=locate(mb,rx['point_mm'])
        va=np.einsum('tak,a->tk',fa[:,ma['tetra'][i]],wa);vb=np.einsum('tak,a->tk',fb[:,mb['tetra'][j]],wb)
        direction=np.array(rx['direction']);direction/=np.linalg.norm(direction)
        vector_error=float(np.linalg.norm(va-vb)/np.linalg.norm(vb));signal_a=va@direction;signal_b=vb@direction
        projected_error=float(np.linalg.norm(signal_a-signal_b)/np.linalg.norm(signal_b)) if np.linalg.norm(signal_b)>1e-15 else None
        rows.append({'name':rx['name'],'point_mm':rx['point_mm'],'direction':direction.tolist(),'vector_relative_l2':vector_error,'direction_relative_l2':projected_error,'max_absolute_component_error_nm':(np.max(abs(va-vb),axis=0)*1e9).tolist()})
    return {'first_job':a['job_id'],'second_job':b['job_id'],'first_scene_sha256':a['scene_sha256'],'second_scene_sha256':b['scene_sha256'],'first_elements':len(a['tetra']),'second_elements':len(b['tetra']),
            'first_internal_step_us':a['diagnostics']['internal_step_us'],'second_internal_step_us':b['diagnostics']['internal_step_us'],'rows':rows,'threshold':.05,'selected_receivers_within_5_percent':all(r['vector_relative_l2']<=.05 and r['direction_relative_l2'] is not None and r['direction_relative_l2']<=.05 for r in rows),
            'scope':'Selected receiver time histories, not a global error bound or independent COMSOL validation.'}

def execute(root,kind,h,cfl=.3,mesh_reference=None):
    scene=default_scene(kind);scene['numerics']['h_mm']=h;scene['numerics']['cfl']=cfl
    scene['name']=f'{kind} · 三维超弹性 · h={h} mm · CFL={cfl}'
    directory=root/'solid/jobs'/('solid-'+uuid.uuid4().hex);directory.mkdir(parents=True)
    write_json(directory/'scene.json',validate_scene(scene));run(directory,mesh_reference);return directory

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/workspace/workai-remote'));args=parser.parse_args()
    evidence=args.root/'validation/20261022-solid';evidence.mkdir(parents=True,exist_ok=True)
    manifest={'schema':'workai.solid.validation.v1','cases':[],'comsol_jobs_started':0,'ai_training_started':False}
    retained=evidence/'retained-plate.json'
    for kind in ['plate','pipe','hole','notch']:
        if kind=='plate' and retained.exists():
            prior=json.loads(retained.read_text());coarse=args.root/'solid/jobs'/prior['coarse'];fine=args.root/'solid/jobs'/prior['fine'];load(coarse);load(fine)
        else:coarse=execute(args.root,kind,.5);fine=execute(args.root,kind,.35)
        timecheck=execute(args.root,kind,.35,.15,fine)
        record={'kind':kind,'coarse':coarse.name,'fine':fine.name,'timecheck':timecheck.name,'spatial_with_changed_stable_step':compare(coarse,fine),'temporal_same_mesh':compare(fine,timecheck)}
        a,_,_=load(fine);b,_,_=load(timecheck);assert a['nodes_mm']==b['nodes_mm'] and a['tetra']==b['tetra']
        manifest['cases'].append(record);write_json(evidence/'convergence.json',manifest)
        print(json.dumps({'case':kind,'space_pass':record['spatial_with_changed_stable_step']['selected_receivers_within_5_percent'],'time_pass':record['temporal_same_mesh']['selected_receivers_within_5_percent']}),flush=True)

if __name__=='__main__':main()
