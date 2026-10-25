import hashlib,importlib.util,json,time,sys
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss
repo=Path('/workspace/workAI');sys.path.insert(0,str(repo))
from graphwave.modal_reduction import attach_basis,propagate
from graphwave.quarter_symmetry import make_quarter_mesh
from graphwave.modal_calibration import digest,save
from spectral3d.mesh import gll
base=Path('/workspace/workai-remote/graphwave/modal-v11');folder=base/'quarter-phase-frozen'
config=json.loads((folder/'model.json').read_text());plan=json.loads((repo/'research/comsol-validation/plans/v11-quarter-native-driven.json').read_text());scene=plan['cases'][0]['scene']
for name,sha in plan['source_sha256'].items():assert digest(repo/name)==sha
with np.load(folder/'weights.npz',allow_pickle=False) as a:weights=dict(a)
with np.load(folder/'basis.npz',allow_pickle=False) as a:frequency,fields=a['frequency_hz'],a['fields']
mesh=make_quarter_mesh(scene,config['degree'],config['h_mm'],config['grading_ratios'],config['axis_elements']);model=attach_basis(mesh,scene,weights,frequency,fields)
output=base/'quarter-phase-gauss9-native0025/native-20us';output.mkdir(parents=True,exist_ok=False)
q=9;points,w=leggauss(q);old=gll(config['degree'])[0];interpolation=[]
for i,x in enumerate(old):
 others=np.delete(old,i);interpolation.append((Polynomial.fromroots(others)/np.prod(x-others))(points))
interpolation=np.array(interpolation).T
original=model['gradient'].reshape(len(frequency),len(mesh['elements']),7,7,7,3,3)
gradient=np.einsum('ai,bj,ck,meijkuv->meabcuv',interpolation,interpolation,interpolation,original,optimize=True)
model['gradient']=np.ascontiguousarray(gradient.reshape(len(frequency),-1));del original,gradient
model['volume']=(np.prod(mesh['size'],axis=1)[:,None,None,None]/8*np.einsum('i,j,k->ijk',w,w,w)[None]).ravel()
identity={'base_model_sha256':digest(folder/'model.json'),'basis_sha256':config['basis_sha256'],'weights_sha256':config['weights_sha256'],'script_sha256':digest(__file__),'case':plan['cases'][0],'source_sha256':plan['source_sha256'],'step_us':.025,'modes':200,'nonlinear_quadrature':'Gauss-Legendre 9 per axis after polynomial lifting of the learned nodal gradient','linear_graph_spectrum_unchanged':True,'reference_arrays_loaded_before_prediction':False,'scope':'Diagnostic of nonlinear quadrature only; mixed original linear tangent and overintegrated nonlinear potential, not a promoted candidate'}
save(output/'identity.json',identity);started=time.monotonic()
def progress(value):
 value.update(elapsed_s=time.monotonic()-started);save(output/'progress.json',value)
result=propagate(model,scene,step_us=.025,composition=np.array(json.loads((repo/'offline/model/time-coefficients.json').read_text())['weights']),keep_coordinates=True,progress=progress)
np.savez_compressed(output/'prediction.npz',**{name:result[name] for name in ['times_us','traces','energy','coordinates']})
spec=importlib.util.spec_from_file_location('checks',repo/'tools/validate-comsol-linear-matrix.py');checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
raw,receipt,native=checks.comparison.load_case(repo/'research/comsol-validation/data/native-quarter-075-20us-v11');assert raw['scene']==scene
expected=checks.comparison.parts(native);actual=checks.comparison.parts({i:result['traces'][i] for i in range(3)})
with np.load(base/'quarter-phase-driven-native-0025/native-20us/prediction.npz') as a:baseline=checks.comparison.parts({i:a['traces'][i] for i in range(3)})
report={'identity':identity,'elapsed_s':time.monotonic()-started,'prediction_sha256':digest(output/'prediction.npz'),'energy_relative_error':result['energy_relative_error'].tolist(),'native_reference_job_id':receipt['job_id'],'native_comparison':{k:checks.check(actual[k],expected[k],scene) for k in expected},'quadrature_change':{k:checks.check(baseline[k],actual[k],scene) for k in actual},'scientific_accuracy_certified':False}
save(output/'metrics.json',report);save(repo/'research/comsol-validation/diagnostics/v11-native-overintegration.json',report)
print(json.dumps({k:{name:row[name] for name in ['maximum_vector_waveform_relative_l2','maximum_relative_l2','passed']} for k,row in report['native_comparison'].items()}),flush=True)
