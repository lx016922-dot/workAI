from pathlib import Path
import importlib.util,json,shutil,sys
import numpy as np
repo=Path('/workspace/workAI');sys.path.insert(0,str(repo));base=Path('/workspace/workai-remote/graphwave/modal-v11')
from graphwave.modal_calibration import digest,save
from graphwave.quarter_symmetry import make_quarter_mesh
spec=importlib.util.spec_from_file_location('checks',repo/'tools/validate-comsol-linear-matrix.py');checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
a=base/'quarter-phase-driven-native-0025/native-20us';b=base/'quarter-phase-300-native0025/native-20us'
ia,ib=[json.loads((p/'identity.json').read_text()) for p in [a,b]]
assert ia['case']==ib['case'] and ia['weights_sha256']==ib['weights_sha256'] and ia['source_sha256']==ib['source_sha256'] and ia['step_us']==ib['step_us']==.025
scene=ia['case']['scene'];arrays=[]
for p in [a,b]:
 pred=json.loads((p/'prediction.json').read_text());assert digest(p/'prediction.npz')==pred['prediction_sha256']
 with np.load(p/'prediction.npz',allow_pickle=False) as f:
  assert np.array_equal(f['times_us'],np.arange(201)*.1)
  arrays.append(checks.comparison.parts({i:f['traces'][i] for i in range(3)}))
conf=json.loads((base/'quarter-phase-300/model.json').read_text());mesh=make_quarter_mesh(scene,6,1,conf['grading_ratios'],conf['axis_elements'])
with np.load(base/'quarter-phase-frozen/basis.npz',allow_pickle=False) as f:f200,phi200=f['frequency_hz'],f['fields']
with np.load(base/'quarter-phase-300/basis.npz',allow_pickle=False) as f:f300,phi300=f['frequency_hz'],f['fields'][:200]
mac=np.einsum('mnc,mnc,n->m',phi200,phi300,mesh['mass'],optimize=True)**2
report={'schema':'workai.graph_modal_truncation_check.v1','duration_us':20,'mode_counts':[200,300],'weights_sha256':ia['weights_sha256'],'step_us':.025,'maximum_common_frequency_difference_hz':float(abs(f200-f300[:200]).max()),'minimum_common_mode_mac':float(mac.min()),'difference':{k:checks.check(arrays[0][k],arrays[1][k],scene) for k in arrays[0]},'native_comparison_300':json.loads((b/'metrics.json').read_text())['native_comparison'],'scientific_accuracy_certified':False,'full_5ms_modal_convergence_verified':False}
output=repo/'research/comsol-validation/diagnostics/v11-native-modal-truncation';output.mkdir(exist_ok=False)
for name,p in [('modes200',a),('modes300',b)]:
 (output/name).mkdir()
 for file in ['identity.json','prediction.json','prediction.npz']:
  shutil.copyfile(p/file,output/name/file)
shutil.copyfile(base/'quarter-phase-300/model.json',output/'compiled-model300.json')
shutil.copyfile(__file__,output/'audit.py')
report['files_sha256']={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
save(output/'summary.json',report)
print(json.dumps({'maximum_common_frequency_difference_hz':report['maximum_common_frequency_difference_hz'],'minimum_common_mode_mac':report['minimum_common_mode_mac'],'truncation_change':{k:{q:v[q] for q in ['maximum_relative_l2','maximum_vector_waveform_relative_l2','passed']} for k,v in report['difference'].items()}}),flush=True)
