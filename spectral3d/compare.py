"""Same-input probe checks, including components hidden by vector norms.

Small components are reported below the absolute reporting floor, never given
a misleading relative accuracy certificate. This is not an arbitrary-point or
continuous-field convergence proof.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from solid.mesh import prepare_mesh,locate

FLOOR_M=1e-13


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def spectral(path):
    manifest=json.loads(path.read_text());assert manifest['phase']=='completed';cases={}
    for name,sha in manifest['source_sha256'].items():assert digest(path.parent/'source_snapshot'/name)==sha
    for case in manifest['cases']:
        values=[];diagnostics=[]
        assert [c['law'] for c in case['controls']]==['linear','stvk','murnaghan']
        assert digest(path.parent/case['mesh_file'])==case['mesh_sha256']
        for control in case['controls']:
            directory=path.parent/control['directory']
            for file,key in [('probes.npz','probes_sha256'),('diagnostics.json','diagnostics_sha256'),('scene.json','scene_sha256')]:assert digest(directory/file)==control[key]
            with np.load(directory/'probes.npz',allow_pickle=False) as data:values.append(data['probes_m']);times=data['times_us']
            diagnostics.append(json.loads((directory/'diagnostics.json').read_text()))
        cases[case['case_id']]={'scene':case['scene'],'times':times,'parts':np.stack([values[0],values[1]-values[0],values[2]-values[1]],axis=2),'diagnostics':diagnostics}
    return cases


def p1_refinement(path,root):
    manifest=json.loads(path.read_text());cases={}
    for case in manifest['cases']:
        level=case['levels'][0];values=[];diagnostics=[]
        for control in level['controls']:
            directory=root/'solid/jobs'/control['job_id'];assert digest(directory/'result.json')==control['result_sha256'];record=json.loads((directory/'result.json').read_text())
            with np.load(directory/'mesh.npz',allow_pickle=False) as saved:mesh=prepare_mesh(saved['points'],saved['tetra'])
            packed=(directory/'field.bin.gz').read_bytes();raw=gzip.decompress(packed);assert hashlib.sha256(packed).hexdigest()==record['field']['sha256'];assert hashlib.sha256(raw).hexdigest()==record['field']['uncompressed_sha256']
            field=np.frombuffer(raw,dtype='<f8').reshape(record['field']['shape']);probes=[]
            for receiver in level['scene']['receivers']:
                cell,w=locate(mesh,receiver['point_mm']);probes.append(np.einsum('tac,a->tc',field[:,mesh['tetra'][cell]],w))
            values.append(np.stack(probes,axis=1));diagnostics.append(record['diagnostics'])
        cases[case['case_id']]={'scene':level['scene'],'times':np.asarray(record['times_us']),'parts':np.stack([values[0],values[1]-values[0],values[2]-values[1]],axis=2),'diagnostics':diagnostics}
    return cases


def complex_coefficient(times,values,frequency):
    weights=.5*(1-np.cos(2*np.pi*(times-times[0])/(times[-1]-times[0])));mean=np.tensordot(weights,values,axes=(0,0))/weights.sum()
    return 2*np.tensordot(weights*np.exp(-2j*np.pi*frequency*times),values-mean,axes=(0,0))/weights.sum()


def metric(a,b,threshold=.05):
    absolute=float(np.sqrt(np.mean(abs(a-b)**2)));scale=float(np.sqrt(np.mean(abs(b)**2)));above=scale>FLOOR_M
    relative=absolute/scale if above else None
    return {'relative_l2':relative,'difference_rms_nm':absolute*1e9,'finer_rms_nm':scale*1e9,
            'above_reporting_floor':above,'pass':bool(relative<=threshold if above else absolute<=FLOOR_M)}


def compare(left,right):
    rows=[]
    for case_id in sorted(set(left)&set(right)):
        a,b=left[case_id],right[case_id]
        for key in ['geometry','material','source','boundary','acquisition','receivers']:assert a['scene'][key]==b['scene'][key]
        np.testing.assert_array_equal(a['times'],b['times']);probes=[]
        for i,rx in enumerate(a['scene']['receivers']):
            fulla=a['parts'][:,i].sum(1);fullb=b['parts'][:,i].sum(1);mata=a['parts'][:,i,2];matb=b['parts'][:,i,2]
            ca=complex_coefficient(a['times'],mata,2*a['scene']['source']['frequency_mhz']);cb=complex_coefficient(b['times'],matb,2*a['scene']['source']['frequency_mhz'])
            direction=np.asarray(rx['direction']);direction/=np.linalg.norm(direction)
            checks={'full_vector':metric(fulla,fullb),'material_vector':metric(mata,matb),'material_2f_vector':metric(ca,cb)}
            for name,projection in zip(['x','y','z','requested'],[*np.eye(3),direction]):
                checks['full_'+name]=metric(fulla@projection,fullb@projection)
                checks['material_'+name]=metric(mata@projection,matb@projection)
                checks['material_2f_'+name]=metric(ca@projection,cb@projection)
            probes.append({'name':rx['name'],'point_mm':rx['point_mm'],'checks':checks,'all_checks_pass':all(c['pass'] for c in checks.values())})
        rows.append({'case_id':case_id,'probes':probes,'all_probe_checks_pass':all(p['all_checks_pass'] for p in probes),
                     'left_diagnostics':[{k:v for k,v in d.items() if k!='energy_samples'} for d in a['diagnostics']],
                     'right_diagnostics':[{k:v for k,v in d.items() if k!='energy_samples'} for d in b['diagnostics']]})
    if not rows:raise ValueError('No matching cases.')
    return {'schema':'workai.spectral_reference_comparison.v1','threshold':.05,'reporting_floor_nm':FLOOR_M*1e9,'cases':rows,
            'all_probe_checks_pass':all(r['all_probe_checks_pass'] for r in rows),'accepted_for_solver_routing':False,
            'scope':'Retained receivers only, full record including reflections. Cartesian components and requested direction checked; arbitrary-point and arbitrary-direction relative error not certified.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True);p.add_argument('--left-p1',action='store_true');p.add_argument('--root',type=Path,default=Path('/workspace/workai-remote'));p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    left=p1_refinement(args.left,args.root) if args.left_p1 else spectral(args.left);right=spectral(args.right);result=compare(left,right);result.update(left_manifest_sha256=digest(args.left),right_manifest_sha256=digest(args.right))
    with args.output.open('x') as file:json.dump(result,file,ensure_ascii=False,indent=2,allow_nan=False)
    for case in result['cases']:
        print(json.dumps({'case_id':case['case_id'],'pass':case['all_probe_checks_pass'],'max_full_vector':max(p['checks']['full_vector']['relative_l2'] for p in case['probes']),'max_material_vector':max(p['checks']['material_vector']['relative_l2'] for p in case['probes']),'failed_components':[[k for k,v in p['checks'].items() if not v['pass']] for p in case['probes']]}),flush=True)
