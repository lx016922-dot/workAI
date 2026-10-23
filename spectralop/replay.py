"""Reload frozen network and independently reproduce retained test metrics."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .data import load_dataset,digest
from operator3d.propagation_model import PropagationSeparatedField
from operator3d.train import predict_case
from operator3d.physics import U0


def run(a):
 torch.set_num_threads(2);root=a.run;config=json.loads((root/'config.json').read_text());metrics=json.loads((root/'metrics.json').read_text());freeze=json.loads((root/'checkpoint-freeze.json').read_text())
 assert digest(root/'best.pt')==metrics['checkpoint_sha256']==freeze['checkpoint_sha256'] and not freeze['test_fields_loaded']
 assert digest(root/'dataset.json')==config['dataset_sha256'] and digest(root/'probe-predictions.npz')==metrics['probe_archive_sha256']
 for name,sha in config['source_sha256'].items():assert digest(root/'source_snapshot'/name)==sha
 cases,scales,description=load_dataset(a.dataset,splits=('train','validation','test'),admission_path=a.admission);record=json.loads((root/'evaluation-dataset.json').read_text());assert description==record
 saved=torch.load(root/'best.pt',map_location='cpu',weights_only=True);np.testing.assert_array_equal(scales,saved['scales'].numpy());model=PropagationSeparatedField(scales,saved['width']);model.load_state_dict(saved['model_state']);results=[];probes=np.load(root/'probe-predictions.npz',allow_pickle=False)
 for c in [c for c in cases if c['split']=='test']:
  row=next(r for r in metrics['cases'] if r['case_id']==c['case_id']);path=root/(c['case_id']+'-prediction.npz');assert digest(path)==row['prediction_archive_sha256']
  with np.load(path,allow_pickle=False) as f:pred=f['parts_m']
  truth=c['parts']*U0;mass=c['mass'].numpy()[None,:,None];checks=0
  for label,p,t in [('full',pred.sum(2),truth.sum(2)),('material',pred[:,:,2],truth[:,:,2]),('geometric',pred[:,:,1],truth[:,:,1])]:
   for axis,name in [(None,'vector'),(0,'x'),(1,'y'),(2,'z')]:
    aa=p if axis is None else p[:,:,axis:axis+1];bb=t if axis is None else t[:,:,axis:axis+1];expected=row['field_checks'][label+'_'+name]
    if expected['relative_l2'] is not None:
     relative=float(np.sqrt(np.sum(mass*(aa-bb)**2)/np.sum(mass*bb**2)));np.testing.assert_allclose(relative,expected['relative_l2'],rtol=1e-12,atol=1e-14)
    checks+=1
  for i,(ids,weights) in enumerate(c['receivers']):
   p=np.einsum('takc,a->tkc',pred[:,ids],weights);t=np.einsum('takc,a->tkc',truth[:,ids],weights);prefix=c['case_id']+f'_P{i+1}';np.testing.assert_array_equal(p,probes[prefix+'_prediction']);np.testing.assert_array_equal(t,probes[prefix+'_fem']);direction=np.array(row['probes'][i]['direction']);direction/=np.linalg.norm(direction);times=c['times'];w=np.hanning(len(times));frequency=2*c['scene']['source']['frequency_mhz']
   def coef(v):return 2*np.sum(w*(v-np.dot(w,v)/w.sum())*np.exp(-2j*np.pi*frequency*times))/w.sum()
   pc=coef(p[:,2]@direction);tc=coef(t[:,2]@direction);pr=row['probes'][i];np.testing.assert_allclose([abs(pc)*1e9,abs(tc)*1e9],[pr['material_2f_predicted_nm'],pr['material_2f_fem_nm']],rtol=1e-12,atol=1e-14)
   if pr['material_2f_complex_relative_error'] is not None:np.testing.assert_allclose(abs(pc-tc)/abs(tc),pr['material_2f_complex_relative_error'],rtol=1e-12,atol=1e-14)
   for key,aa,bb in [('full_waveform_relative_l2',p.sum(1)@direction,t.sum(1)@direction),('material_waveform_relative_l2',p[:,2]@direction,t[:,2]@direction)]:
    if pr[key] is not None:np.testing.assert_allclose(np.linalg.norm(aa-bb)/np.linalg.norm(bb),pr[key],rtol=1e-12,atol=1e-14)
  recomputed=False
  if not results:
   fresh=predict_case(model,c)*U0;np.testing.assert_array_equal(fresh,pred);recomputed=True
  results.append({'case_id':c['case_id'],'field_metrics_reproduced':checks,'all_probe_arrays_exact':True,'complex_harmonics_reproduced':True,'full_network_prediction_recomputed_exactly':recomputed})
 result={'passed':True,'checkpoint_sha256':metrics['checkpoint_sha256'],'cases':results,'all_source_and_dataset_hashes_verified':True,'new_training_steps':0,'new_fem_solves':0}
 with a.output.open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--admission',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
