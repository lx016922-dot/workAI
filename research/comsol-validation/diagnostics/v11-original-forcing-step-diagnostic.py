import json,time,sys,importlib.util
from pathlib import Path
import numpy as np
sys.path.insert(0,'/workspace/workAI')
from graphwave.quarter_symmetry import make_quarter_mesh,sampling
from graphwave.modal_reduction import oscillator_flow
from graphwave.signals import pulse_values
from spectral3d.mesh import boundary_data
from tools.comsol_modal_response import oscillator_response
spec=importlib.util.spec_from_file_location('comparison','/workspace/workAI/tools/validate-comsol-linear-matrix.py');comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)
r=Path('/workspace/workAI');b=Path('/workspace/workai-remote/graphwave/modal-v11/quarter-baseline-frozen')
cfg=json.loads((b/'model.json').read_text());plan=json.loads((r/'research/comsol-validation/plans/v11-quarter-validation.json').read_text())
mesh=make_quarter_mesh(cfg['scene'],6,1.,(50.,12.,12.),(12,3,3))
with np.load(b/'basis.npz') as f:frequency=f['frequency_hz'];fields=f['fields']
rx=sampling(mesh,np.array([r['point_mm'] for r in cfg['scene']['receivers']])*1e-3)(fields)
weights=np.array(json.loads((r/'offline/model/time-coefficients.json').read_text())['weights']);omega=2*np.pi*frequency
rows=[]
for case in plan['cases']:
 if case['id'] not in ['hann-01','rectangular-05','chirp-20']:continue
 scene=case['scene'];times=np.arange(25001)*.2;coupling=np.einsum('mnc,nc->m',fields,boundary_data(mesh,scene)[0])
 exact,_=oscillator_response(scene,frequency,times);expected=np.einsum('tm,m,mrc->trc',exact,coupling,rx,optimize=True)
 for step in [.2,.1,.05]:
  n=round(5000/step);stride=round(.2/step);q=np.zeros(200);v=q.copy();trace=np.zeros_like(expected)
  pulse=pulse_values(scene,(np.arange(n)[:,None]+np.r_[0,np.cumsum(weights)][None,:])*step)
  start=time.monotonic()
  for i in range(n):
   for stage,w in enumerate(weights):
    h=w*step*1e-6;v+=.5*h*coupling*pulse[i,stage];q,v=oscillator_flow(q,v,omega,h);v+=.5*h*coupling*pulse[i,stage+1]
   if (i+1)%stride==0:trace[(i+1)//stride]=np.einsum('m,mrc->rc',q,rx)
  check=comparison.check(trace,expected,scene);row={'case':case['id'],'step_us':step,'maximum_vector_error':check['maximum_vector_waveform_relative_l2'],'maximum_error':check['maximum_relative_l2'],'passed':check['passed'],'elapsed_s':time.monotonic()-start};rows.append(row);print(json.dumps(row),flush=True)
Path('/workspace/recovery/v11-linear-time-diagnostic.json').write_text(json.dumps(rows,indent=2)+'\n')
