"""Fresh nonlinear weak-momentum and energy admission, analytic material target."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss
repo=Path('/workspace/workAI');sys.path.insert(0,str(repo))
from graphwave.modal_reduction import attach_basis,_nonlinear_stresses
from graphwave.modal_sensitivity import modal_gradients
from graphwave.modal_calibration import digest,save
from graphwave.quarter_symmetry import make_quarter_mesh
from tools.comsol_nodal_reference import stresses as analytic_stresses
from spectral3d.mesh import gll
p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True);group.add_argument('--gauss',type=int,choices=[3,4,5,7]);group.add_argument('--cubature',type=Path);p.add_argument('--fields',type=int,default=64);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
started=time.monotonic();folder=Path('/workspace/workai-remote/graphwave/modal-v11/quarter-phase-300');conf=json.loads((folder/'model.json').read_text());scene=conf['scene']
for name,key in [('basis.npz','basis_sha256'),('weights.npz','weights_sha256')]:assert digest(folder/name)==conf[key]
with np.load(folder/'weights.npz',allow_pickle=False) as f:weights=dict(f)
with np.load(folder/'basis.npz',allow_pickle=False) as f:frequency,fields=f['frequency_hz'],f['fields']
mesh=make_quarter_mesh(scene,conf['degree'],conf['h_mm'],conf['grading_ratios'],conf['axis_elements']);full=attach_basis(mesh,scene,weights,frequency,fields);candidate=dict(full)
if a.gauss:
 x,w=leggauss(a.gauss);nodes=gll(6)[0];L=[]
 for i,n in enumerate(nodes):
  others=np.delete(nodes,i);L.append((Polynomial.fromroots(others)/np.prod(n-others))(x))
 L=np.array(L).T;np.testing.assert_allclose(L@np.ones(7),np.ones(len(x)),rtol=0,atol=1e-13);np.testing.assert_allclose(L@nodes,x,rtol=0,atol=1e-13)
 gradient=full['gradient'].reshape(len(frequency),len(mesh['elements']),7,7,7,3,3)
 candidate['gradient']=np.ascontiguousarray(np.einsum('ai,bj,ck,meijkuv->meabcuv',L,L,L,gradient,optimize=True).reshape(len(frequency),-1))
 candidate['volume']=(np.prod(mesh['size'],axis=1)[:,None,None,None]/8*np.einsum('i,j,k->ijk',w,w,w)[None]).ravel()
 label='gauss'+str(a.gauss);selection_identity={'method':'Gauss-Legendre polynomial lift of learned gradient','points_per_axis':a.gauss}
else:
 with np.load(a.cubature,allow_pickle=False) as f:
  selected=f['point_indices'];volume=f['volume']
 assert selected.ndim==1 and len(set(selected.tolist()))==len(selected) and np.all((selected>=0)&(selected<len(full['volume']))) and np.all(volume>0)
 candidate['gradient']=np.ascontiguousarray(full['gradient'].reshape(len(frequency),-1,9)[:,selected,:].reshape(len(frequency),-1));candidate['volume']=volume
 label='positive-cubature';selection_identity={'method':'trained positive quadrature','artifact_sha256':digest(a.cubature)}
np.testing.assert_allclose(candidate['volume'].sum(),full['volume'].sum(),rtol=1e-12,atol=0)
# The target uses the exact polynomial derivative and analytic constitutive law.
reference_gradient=np.ascontiguousarray(modal_gradients(mesh,mesh['derivative'],fields).reshape(len(frequency),-1));del fields
constants=np.array([scene['material'][k] for k in ['lambda_pa','mu_pa','l_pa','m_pa','n_pa']])
def parts(stress,energy,gradient):
 # Keep x/y/z weak momentum separately so cancellation cannot hide a component.
 shaped=gradient.reshape(len(frequency),-1,3,3)
 moments=np.einsum('lpab,mpab->lam',stress,shaped,optimize=True)
 moments=np.concatenate([moments,moments.sum(axis=1,keepdims=True)],axis=1)
 return np.stack([moments[0],moments[1]-moments[0],moments[1]]),np.array([energy[0],energy[1]-energy[0],energy[1]])
def compare(actual,expected):
 out=[]
 for part in range(3):
  for axis in range(4):
   for scale in [np.ones(len(frequency)),1/full['omega'],1/full['omega']**2]:
    out.append(float(np.linalg.norm((actual[0][part,axis]-expected[0][part,axis])*scale)/max(np.linalg.norm(expected[0][part,axis]*scale),1e-30)))
  out.append(float(abs(actual[1][part]-expected[1][part])/max(abs(expected[1][part]),1e-30)))
 return out
rng=np.random.default_rng(a.seed);qscale=(abs(full['load'])+.01*np.linalg.norm(full['load'])/np.sqrt(len(frequency)))/full['omega']**2
rows=[];states=[]
for index in range(a.fields):
 q=rng.normal(size=len(frequency))*qscale;g=(q@reference_gradient).reshape(-1,3,3)
 q*=rng.uniform(2e-5,1e-4)/np.linalg.norm(g,axis=(1,2)).max();states.append(q)
 values=np.stack([(q@reference_gradient).reshape(-1,3,3)]*2)
 stress,energy,strain,jac=analytic_stresses(values,full['volume'],constants,np.array([1,2],dtype=np.int64));expected=parts(stress,energy,reference_gradient)
 errors=[]
 for model in [full,candidate]:
  values=np.stack([(q@model['gradient']).reshape(-1,3,3)]*2)
  stress,energy,_,_=_nonlinear_stresses(values,model['volume'],model['quadratic'],model['cubic'],np.array([1,2],dtype=np.int64))
  errors.append(compare(parts(stress,energy,model['gradient']),expected))
 row={'field':index,'full_graph_errors':errors[0],'candidate_errors':errors[1],'full_graph_passed':max(errors[0])<=.025,'candidate_passed':max(errors[1])<=.025,'maximum_reference_strain':float(strain.max()),'minimum_reference_j':float(jac.min())};rows.append(row)
 if (index+1)%8==0:print(json.dumps({'validated_fields':index+1,'candidate_max_so_far':max(max(r['candidate_errors']) for r in rows),'elapsed_s':time.monotonic()-started}),flush=True)
np.savez_compressed(a.output/'independent-states.npz',coordinates=np.array(states))
report={'schema':'workai.independent_modal_physics_admission.v1','scope':'Geometric/material/full nonlinear increments; x,y,z and summed weak momentum; raw and frequency-weighted modal forces; nonlinear energies. Separate from waveform validation.','label':label,'selection':selection_identity,'source_sha256':digest(__file__),'base_weights_sha256':conf['weights_sha256'],'base_basis_sha256':conf['basis_sha256'],'validation_seed':a.seed,'fields':a.fields,'constraints_per_field':39,'hard_relative_limit':.025,'original_points':len(full['volume']),'candidate_points':len(candidate['volume']),'reference_uses_learned_material_tensors':False,'reference_uses_learned_spatial_derivative':False,'full_graph_passed':all(r['full_graph_passed'] for r in rows),'candidate_passed':all(r['candidate_passed'] for r in rows),'maximum_full_graph_error':max(max(r['full_graph_errors']) for r in rows),'maximum_candidate_error':max(max(r['candidate_errors']) for r in rows),'rows':rows,'elapsed_s':time.monotonic()-started,'scientific_accuracy_certified':False,'independent_states_sha256':digest(a.output/'independent-states.npz')}
save(a.output/'report.json',report)
print(json.dumps({k:report[k] for k in ['label','candidate_points','full_graph_passed','candidate_passed','maximum_full_graph_error','maximum_candidate_error','elapsed_s']}),flush=True)
