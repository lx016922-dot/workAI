import sys,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
sys.path.insert(0,'/workspace/workAI')
from graphwave.modal_calibration import read_modes,stiffness,modal_sampling,match,filter_basis
from spectral3d.mesh import make_mesh,boundary_data
root=Path('/workspace/workAI');ref=read_modes(root/'research/comsol-validation/data/eigen-030-v7')
scene=ref['scene'];label=make_mesh(scene,4,2.5);selected=[5,11,20,22,25,28,29]
with np.load(root/'offline/model/continued-weights.npz',allow_pickle=False) as f:weights=dict(f)
rows=[]
for degree,h in [(5,1.6),(6,2.5),(6,1.6)]:
 started=time.monotonic();mesh=make_mesh(scene,degree,h);_,fixed,_=boundary_data(mesh,scene)
 free=np.ones(mesh['points'].size,bool);free[(fixed[:,None]*3+np.arange(3)).ravel()]=False
 k=stiffness(mesh,mesh['derivative'],weights['quadratic_tensor']);mass=diags(np.repeat(mesh['mass'],3)).tocsr()
 val,vec=eigsh(k[free][:,free],M=mass[free][:,free],k=36,sigma=0,which='LM',tol=1e-11,v0=np.ones(free.sum()))
 order=np.argsort(val);frequency=np.sqrt(val[order])/(2*np.pi);fields=np.zeros((36,mesh['points'].size));fields[:,free]=vec[:,order].T
 mapping,mac=match(ref['fields'][:32],modal_sampling(mesh,ref['coordinates'])(fields.reshape(36,-1,3)),label['mass'])
 error=frequency[mapping]-ref['frequency_hz'][:32]
 subdivisions=int(np.ceil(.2e-6/(.3*mesh['min_node_spacing']/10000)))
 row={'degree':degree,'h_mm':h,'elements':len(mesh['elements']),'nodes':len(mesh['points']),'min_node_spacing_m':mesh['min_node_spacing'],'subdivisions':subdivisions,'internal_step_us':.2/subdivisions,'dominant_difference_hz':float(error[5]),'selected_difference_hz':error[selected].tolist(),'selected_error_norm_hz':float(np.linalg.norm(error[selected])),'minimum_selected_mac':float(mac[selected].min()),'sbp_c2_free_parameters':len(filter_basis(mesh['gll'],2,mesh['weights_1d'])),'seconds':time.monotonic()-started}
 rows.append(row);print(json.dumps(row),flush=True)
 Path('/workspace/recovery/v10-space-probe.json').write_text(json.dumps({'schema':'workai.spatial_discretization_probe.v1','reference_identity':ref['identity'],'scope':'Analytic GLL discretization diagnosis using only archived development eigenpairs; no transient test records or accuracy certification','rows':rows},indent=2)+'\n')
