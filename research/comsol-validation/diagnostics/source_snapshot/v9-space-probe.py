import sys, importlib.util, json, time
from pathlib import Path
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
sys.path.insert(0,'/workspace/workAI')
from graphwave.modal_calibration import read_modes,stiffness,modal_sampling,match
from spectral3d.mesh import make_mesh,boundary_data
spec=importlib.util.spec_from_file_location('quad','/workspace/workAI/tools/diagnose-comsol-quadrature.py')
quad=importlib.util.module_from_spec(spec);spec.loader.exec_module(quad)
root=Path('/workspace/workAI');ref=read_modes(root/'research/comsol-validation/data/eigen-030-v7')
scene=ref['scene'];label=make_mesh(scene,4,2.5)
with np.load(root/'offline/model/continued-weights.npz',allow_pickle=False) as saved: weights=dict(saved)
rows=[]
for h in [2.5,1.6,1.4]:
 mesh=make_mesh(scene,5,h);load,fixed,_=boundary_data(mesh,scene)
 free=np.ones(mesh['points'].size,bool);free[(fixed[:,None]*3+np.arange(3)).ravel()]=False
 mass=diags(np.repeat(mesh['mass'],3)).tocsr();sample=modal_sampling(mesh,ref['coordinates'])
 exact_k,_=quad.exact_operators(mesh,weights['quadratic_tensor'],scene['material']['rho_kg_m3'])
 for method,k in [('gll',stiffness(mesh,mesh['derivative'],weights['quadratic_tensor'])),('gauss_stiffness_gll_mass',exact_k)]:
  start=time.monotonic();val,vec=eigsh(k[free][:,free],M=mass[free][:,free],k=36,sigma=0,which='LM',tol=1e-11,v0=np.ones(free.sum()))
  order=np.argsort(val);f=np.sqrt(val[order])/(2*np.pi);fields=np.zeros((36,mesh['points'].size));fields[:,free]=vec[:,order].T
  mapping,mac=match(ref['fields'][:32],sample(fields.reshape(36,-1,3)),label['mass'])
  selected=[5,11,20,22,25,28,29];err=f[mapping]-ref['frequency_hz'][:32]
  row={'degree':5,'h_mm':h,'elements':len(mesh['elements']),'nodes':len(mesh['points']),'method':method,'dominant_difference_hz':float(err[5]),'selected_difference_hz':err[selected].tolist(),'minimum_selected_mac':float(mac[selected].min()),'seconds':time.monotonic()-start}
  rows.append(row);print(json.dumps(row),flush=True)
  Path('/workspace/recovery/v9-space-probe.json').write_text(json.dumps({'reference_identity':ref['identity'],'scope':'Analytic spatial diagnostic, not learned inference or accuracy certification','rows':rows},indent=2)+'\n')
