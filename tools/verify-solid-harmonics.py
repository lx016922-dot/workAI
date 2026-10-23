"""Independent NumPy barycentric interpolation and complex projection check."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('evidence',type=Path);args=parser.parse_args()
    report=json.loads((args.evidence/'harmonic-validation.json').read_text());verified=[]
    for ref in report['references']:
        directory=args.root/'solid/studies'/ref['study_id'];data=(directory/'result.json').read_bytes();assert hashlib.sha256(data).hexdigest()==ref['result_sha256']
        study=json.loads(data);signals={};probes={};analyses={}
        for index,rx in enumerate(study['scene']['receivers']):
            base=args.evidence/(ref['level']+'-P'+str(index+1));probes[index]=json.loads(Path(str(base)+'-probe.json').read_text());analyses[index]=json.loads(Path(str(base)+'-analysis.json').read_text());signals[index]={}
        for control in study['controls']:
            job=args.root/'solid/jobs'/control['job_id'];data=(job/'result.json').read_bytes();assert hashlib.sha256(data).hexdigest()==control['result_sha256'];r=json.loads(data)
            packed=(job/'field.bin.gz').read_bytes();assert hashlib.sha256(packed).hexdigest()==r['field']['sha256'];raw=gzip.decompress(packed);assert hashlib.sha256(raw).hexdigest()==r['field']['uncompressed_sha256']
            dtype='<f8' if r['field']['dtype']=='little_endian_float64' else '<f4';field=np.frombuffer(raw,dtype=dtype).reshape(r['field']['shape']);nodes=np.array(r['nodes_mm']);tetra=np.array(r['tetra']);vertices=nodes[tetra]
            for index,rx in enumerate(study['scene']['receivers']):
                point=np.array(rx['point_mm']);candidate=np.flatnonzero(np.all(point>=vertices.min(axis=1)-1e-7,axis=1)&np.all(point<=vertices.max(axis=1)+1e-7,axis=1));found=False
                for cell in candidate:
                    v=vertices[cell];w=np.linalg.solve((v[1:]-v[0]).T,point-v[0]);w=np.r_[1-w.sum(),w]
                    if w.min()>=-1e-7 and w.max()<=1+1e-7:found=True;break
                assert found;w=np.clip(w,0,1);w/=w.sum();xyz=np.einsum('tak,a->tk',field[:,tetra[cell]],w);stored=probes[index]['controls'][control['role']]
                np.testing.assert_allclose(xyz,stored['xyz_m'],rtol=1e-12,atol=1e-22)
                direction=np.array(rx['direction']);direction/=np.linalg.norm(direction);signals[index][control['role']]=xyz@direction
        for index in probes:
            p=probes[index];a=analyses[index];time=np.array(p['times_us']);start,end=a['window_us'];window=np.where((time>=start)&(time<=end),.5*(1-np.cos(2*np.pi*(time-start)/(end-start))),0);f=p['scene']['source']['frequency_mhz'];values=signals[index]
            def coefficient(signal,harmonic):
                mean=np.sum(window*signal)/window.sum();return 2*np.sum(window*(signal-mean)*np.exp(-2j*np.pi*harmonic*f*time))/window.sum()
            checks={'fundamental':coefficient(values['material_A'],1),'linear_second':coefficient(values['linear_A'],2),'material_second':coefficient(values['material_A']-values['geometric_A'],2),'geometric_second':coefficient(values['geometric_A']-values['linear_A'],2),'material_double_second':coefficient(values['material_2A']-values['geometric_2A'],2)}
            for key,value in checks.items():np.testing.assert_allclose([value.real,value.imag],[a[key]['re_m'],a[key]['im_m']],rtol=1e-7,atol=1e-23)
            verified.append({'level':ref['level'],'receiver':index,'all_five_fields_verified':True,'interpolation_and_complex_projection_agree':True})
    result={'passed':True,'independent_numpy_checks':verified,'not_an_independent_physics_solver_comparison':True};(args.evidence/'independent-numpy.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
