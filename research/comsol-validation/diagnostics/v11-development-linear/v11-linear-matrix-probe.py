import sys,json,copy,time,importlib.util
from pathlib import Path
import numpy as np
sys.path.insert(0,'/workspace/workAI')
from graphwave.graded_mesh import make_graded_mesh
from graphwave.modal_calibration import read_modes,modal_sampling
from spectral3d.mesh import boundary_data
from tools.comsol_modal_response import oscillator_response,response
spec=importlib.util.spec_from_file_location('compare','/workspace/workAI/tools/validate-comsol-linear-matrix.py');compare=importlib.util.module_from_spec(spec);spec.loader.exec_module(compare)
root=Path('/workspace/workAI');base=Path('/workspace/workai-remote/graphwave/modal-v11/baseline');cfg=json.loads((base/'config.json').read_text());scene=cfg['scene'];mesh=make_graded_mesh(scene,6,1.,(20.,12.,12.),(8,3,3))
with np.load(base/'basis.npz') as f:fields=f['fields'];frequency=f['frequency_hz']
receivers=modal_sampling(mesh,np.array([r['point_mm'] for r in scene['receivers']])*1e-3)(fields)
plan=json.loads((root/'research/comsol-validation/plans/v10-p6-h16.json').read_text())
output=base/'linear-matrix-graded035';output.mkdir(exist_ok=False)
reference=read_modes(root/'research/comsol-validation/data/eigen-graded20-035-v11');rows=[]
for case in plan['cases']:
 scene=case['scene'];times=np.arange(25001)*.2
 load,_,_=boundary_data(mesh,scene);coupling=np.einsum('mnc,nc->m',fields,load)
 coordinates,_=oscillator_response(scene,frequency,times)
 actual=np.einsum('tm,m,mrc->trc',coordinates,coupling,receivers,optimize=True)
 _,expected,_=response(reference,scene)
 _,truncated,_=response(reference,scene,count=160)
 result={'id':case['id'],'linear_comparison':compare.check(actual,expected,scene),'reference_truncation':compare.check(truncated,expected,scene),'reference_identity':reference['identity'],'native_nonlinear_reference':False,'full_neural_nonlinear_prediction':False,'scientific_accuracy_certified':False}
 np.savez_compressed(output/(case['id']+'.npz'),actual=actual,expected=expected,times_us=times)
 (output/(case['id']+'.json')).write_text(json.dumps(result,indent=2)+'\n');rows.append(result)
 print(json.dumps({'case':case['id'],'vector_relative_l2':result['linear_comparison']['maximum_vector_waveform_relative_l2'],'passed':result['linear_comparison']['passed'],'truncation_passed':result['reference_truncation']['passed']}),flush=True)
(output/'summary.json').write_text(json.dumps({'scope':'Development linear phase/truncation diagnosis using the candidate graph eigenpairs; NOT a nonlinear prediction or accuracy certificate','results':rows},indent=2)+'\n')
