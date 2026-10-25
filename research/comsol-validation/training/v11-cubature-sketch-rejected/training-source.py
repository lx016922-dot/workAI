"""Development preflight: positive conservative nonlinear modal quadrature.
No measured COMSOL or held-out waveform is used to fit the sampling weights.
"""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from numba import njit
from scipy.optimize import nnls
repo=Path('/workspace/workAI');sys.path.insert(0,str(repo))
from graphwave.modal_reduction import attach_basis,nonlinear_force
from graphwave.modal_calibration import digest,save
from graphwave.quarter_symmetry import make_quarter_mesh
from graphwave.buffered_inference import _activation
base=Path('/workspace/workai-remote/graphwave/modal-v11');folder=base/'quarter-phase-300';output=Path('/workspace/workai-remote/graphwave/runs/v11-positive-cubature-preflight');output.mkdir(exist_ok=False)
started=time.monotonic();config=json.loads((folder/'model.json').read_text());scene=config['scene']
with np.load(folder/'weights.npz',allow_pickle=False) as f:weights=dict(f)
with np.load(folder/'basis.npz',allow_pickle=False) as f:frequency,fields=f['frequency_hz'],f['fields']
mesh=make_quarter_mesh(scene,config['degree'],config['h_mm'],config['grading_ratios'],config['axis_elements']);model=attach_basis(mesh,scene,weights,frequency,fields)
points=len(model['volume']);modes=len(frequency);full_volume=model['volume'].copy();volume_total=full_volume.sum();normalized_volume=full_volume/volume_total
rng=np.random.default_rng(2026102557);ntrain=32;ntest=32;directions=8;rows=ntrain*(2*directions*3+2)+64
feature=np.empty((rows,points),dtype=float)
@njit(cache=True)
def density(g,quadratic,cubic):
 strain,second,linear_stress,geometric,full=[np.empty((3,3)) for _ in range(5)]
 z,linear,nonlinear=[np.empty(6) for _ in range(3)]
 stress=np.empty((2,len(g),3,3));energy=np.empty((2,len(g)))
 for p in range(len(g)):
  e0,_,_=_activation(g[p],quadratic,cubic,0,strain,z,linear,nonlinear,second,linear_stress)
  e1,_,_=_activation(g[p],quadratic,cubic,1,strain,z,linear,nonlinear,second,geometric)
  e2,_,_=_activation(g[p],quadratic,cubic,2,strain,z,linear,nonlinear,second,full)
  stress[0,p]=geometric-linear_stress;stress[1,p]=full-geometric
  energy[0,p]=e1-e0;energy[1,p]=e2-e1
 return stress,energy
qscale=(abs(model['load'])+.01*np.linalg.norm(model['load'])/np.sqrt(modes))/model['omega']**2
samples=[];at=0
for index in range(ntrain):
 q=rng.normal(size=modes)*qscale
 gradient=(q@model['gradient']).reshape(points,3,3)
 factor=rng.uniform(2e-5,1e-4)/np.sqrt(np.sum(gradient*gradient,axis=(1,2))).max();q*=factor;gradient*=factor;samples.append(q)
 stress,energy=density(gradient,model['quadratic'],model['cubic'])
 virtual=rng.normal(size=(directions,modes))/np.sqrt(modes)
 virtual[directions//2:]*=frequency[0]/frequency[None,:]
 test_gradient=(virtual@model['gradient']).reshape(directions,points,3,3)
 block=np.einsum('lpab,dpab->ldap',stress,test_gradient,optimize=True).reshape(-1,points)
 feature[at:at+len(block)]=block;at+=len(block)
 feature[at:at+2]=energy;at+=2
 print(json.dumps({'phase':'physics_features','completed':index+1,'total':ntrain,'elapsed_s':time.monotonic()-started}),flush=True)
coordinates=mesh['points'][mesh['elements']].reshape(points,3);lengths=coordinates.max(axis=0)
scaled=2*coordinates/lengths-1
polynomial=[np.polynomial.legendre.legvander(scaled[:,i],3) for i in range(3)]
for i in range(4):
 for j in range(4):
  for k in range(4):feature[at]=polynomial[0][:,i]*polynomial[1][:,j]*polynomial[2][:,k];at+=1
assert at==rows
normalization=np.sqrt((feature*feature)@normalized_volume);active=normalization>1e-24
feature=feature[active]/normalization[active,None];target=feature@normalized_volume
np.savez_compressed(output/'training-states.npz',coordinates=np.array(samples),normalization=normalization,active=active)
# Random projection controls memory and fit cost; admission always rechecks the
# complete unprojected feature system and fresh full modal force/energy vectors.
rank=384;projection=rng.normal(size=(rank,len(target)))/np.sqrt(rank)
compressed=projection@feature;expected=projection@target
norm=np.linalg.norm(compressed,axis=0);norm=np.maximum(norm,1e-30)
selected=[];residual=expected.copy();history=[];positive=None
for iteration in range(256):
 score=(compressed.T@residual)/norm
 if selected:score[selected]=-np.inf
 point=int(np.argmax(score))
 if score[point]<=0:raise RuntimeError('Positive fitting direction exhausted')
 selected.append(point)
 positive,_=nnls(compressed[:,selected],expected,maxiter=5000)
 residual=expected-compressed[:,selected]@positive
 if iteration%16==15 or iteration==255:
  normalized=positive/positive.sum()
  full_error=float(np.linalg.norm(feature[:,selected]@normalized-target)/np.linalg.norm(target))
  record={'points':len(selected),'projected_relative_l2':float(np.linalg.norm(residual)/np.linalg.norm(expected)),'complete_feature_relative_l2':full_error,'elapsed_s':time.monotonic()-started}
  history.append(record);print(json.dumps(record),flush=True);save(output/'training-progress.json',history)
  if full_error<.001:break
positive/=positive.sum();selected=np.array(selected,dtype=np.int64)
keep=positive>0;selected=selected[keep];positive=positive[keep]
reduced=dict(model);reduced['gradient']=np.ascontiguousarray(model['gradient'].reshape(modes,points,9)[:,selected,:].reshape(modes,-1));reduced['volume']=positive*volume_total
validation=[];independent=np.random.default_rng(2026102559)
# The full-field gradient upper bound can guard every original integration point
# without treating the selected points as a substitute for the strain/J guard.
gradient_bound=np.linalg.norm(model['gradient'].reshape(modes,points,9),axis=-1).max(axis=1)
for index in range(ntest):
 q=independent.normal(size=modes)*qscale;g=(q@model['gradient']).reshape(points,3,3)
 q*=independent.uniform(2e-5,1e-4)/np.sqrt(np.sum(g*g,axis=(1,2))).max()
 qrows=np.stack([q,q]);full=nonlinear_force(model,qrows,[1,2]);sparse=nonlinear_force(reduced,qrows,[1,2])
 errors=[]
 for law in range(2):
  for scaling in [np.ones(modes),1/model['omega'],1/model['omega']**2]:
   errors.append(float(np.linalg.norm((sparse[0][law]-full[0][law])*scaling)/max(np.linalg.norm(full[0][law]*scaling),1e-30)))
  errors.append(float(abs(sparse[1][law]-full[1][law])/max(abs(full[1][law]),1e-30)))
 validation.append({'field':index,'errors':errors,'passed':max(errors)<=.025,'full_gradient_norm_upper_bound':float(abs(q)@gradient_bound)})
volume_error=float(abs(reduced['volume'].sum()-volume_total)/volume_total)
np.savez_compressed(output/'candidate-cubature.npz',point_indices=selected,volume=reduced['volume'],gradient_norm_bound=gradient_bound)
report={'schema':'workai.positive_modal_cubature_preflight.v1','model_sha256':digest(folder/'model.json'),'basis_sha256':config['basis_sha256'],'weights_sha256':config['weights_sha256'],'script_sha256':digest(__file__),'train_seed':2026102557,'validation_seed':2026102559,'training_fields':ntrain,'independent_validation_fields':ntest,'original_quadrature_points':points,'retained_quadrature_points':len(selected),'all_weights_positive':bool(np.all(reduced['volume']>0)),'relative_volume_error':volume_error,'history':history,'validation':validation,'validation_passed':all(row['passed'] for row in validation),'maximum_validation_error':max(max(row['errors']) for row in validation),'uses_comsol_transient_or_validation_waveforms_for_training':False,'material_tensors_changed':False,'waveform_accuracy_verified':False,'scientific_accuracy_certified':False,'elapsed_s':time.monotonic()-started,'scope':'Restricted modal energy/force preflight; not a replacement for existing full physical-constraint admission or independent COMSOL waveform validation'}
save(output/'report.json',report);print(json.dumps({k:report[k] for k in ['retained_quadrature_points','validation_passed','maximum_validation_error','elapsed_s']}),flush=True)
