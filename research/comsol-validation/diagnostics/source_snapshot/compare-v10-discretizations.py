"""Compare audited analytic P5/P6 discretizations, separate from weight training."""
import argparse,copy,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np

repo=next(p for p in Path(__file__).resolve().parents if (p/'graphwave').is_dir())
sys.path.insert(0,str(repo))
from spectral3d.mesh import gll
root=repo/'research/comsol-validation'
if not root.is_dir():root=repo/'research'  # Layout inside the offline evidence archive.
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
folders=[root/'candidate-matrix20-v9-untrained',root/'candidate-matrix20-v10-untrained']
plan_paths=[root/'plans/v9-p5-h16.json',root/'plans/v10-p6-h16.json']
weight_paths=[root/'training/v9-untrained/candidate-weights.npz',root/'training/v10-untrained/candidate-weights.npz']
reports=[read(p/'summary.json') for p in folders];plans=[read(p) for p in plan_paths]
for report,plan,weights in zip(reports,plan_paths,weight_paths):
 if report['completed_cases']!=20 or not report['metrics_independently_recomputed']:raise ValueError('Complete independent audit required')
 if report['plan_sha256']!=sha(plan) or report['weights_sha256']!=sha(weights):raise ValueError('Frozen inputs changed')
for key in ['reference_identity','refinement_identity']:
 if reports[0][key]!=reports[1][key]:raise ValueError('Different COMSOL reference')
for key in ['source_sha256','temporal_sha256','policy']:
 if plans[0][key]!=plans[1][key]:raise ValueError('Different inference implementation or policy')
normalized=[]
for plan,degree in zip(plans,[5,6]):
 cases=copy.deepcopy(plan['cases'])
 if any(c['candidate_degree']!=degree or c['h_mm']!=1.6 for c in cases):raise ValueError('Unexpected discretization')
 for c in cases:del c['candidate_degree']
 normalized.append(cases)
if normalized[0]!=normalized[1]:raise ValueError('Physical cases differ')
weights=[]
for path,degree in zip(weight_paths,[5,6]):
 with np.load(path,allow_pickle=False) as saved:value=dict(saved)
 np.testing.assert_allclose(value[f'derivative_{degree}'],gll(degree)[2],rtol=0,atol=5e-13)
 weights.append(value)
if set(weights[1])-set(weights[0])!={'derivative_6'} or set(weights[0])-set(weights[1]):raise ValueError('Unexpected added parameters')
for key in weights[0]:np.testing.assert_array_equal(weights[0][key],weights[1][key])
indexes=[{row['case_id']:row for row in report['rows']} for report in reports]
if len(indexes[0])!=20 or indexes[0].keys()!=indexes[1].keys():raise ValueError('Incomplete matrix')
rows=[];steps=[set(),set()]
for case in indexes[0]:
 vectors=[]
 for n,(folder,index,report) in enumerate(zip(folders,indexes,reports)):
  path=folder/'metrics'/(case+'.json')
  if sha(path)!=index[case]['metrics_sha256']:raise ValueError('Audited metrics changed')
  record=read(path);prediction=record['prediction'];steps[n].add(prediction['internal_step_us'])
  if prediction['weights_sha256']!=report['weights_sha256'] or record['reference_identity']!=report['reference_identity']:raise ValueError('Per-case identity differs')
  values=[row['vector']['relative_l2'] for row in record['linear_comparison']['waveform']]
  if len(values)!=8 or any(x is None or not math.isfinite(x) for x in values):raise ValueError('Eight nonweak vector comparisons required')
  if not math.isclose(max(values),index[case]['max_linear_vector_relative_l2'],rel_tol=1e-7,abs_tol=1e-14):raise ValueError('Summary differs')
  vectors.append(values)
 rows.append({'case_id':case,'p5_max_vector_relative_l2':max(vectors[0]),'p6_max_vector_relative_l2':max(vectors[1]),
  'change_percentage_points':100*(max(vectors[1])-max(vectors[0])),
  'improved_receivers':sum(b<a-1e-10 for a,b in zip(*vectors))})
if any(len(x)!=1 for x in steps):raise ValueError('Unexpected changing time step')
dt=[next(iter(x)) for x in steps]
if [round(.2/x) for x in dt]!=[38,53]:raise ValueError('Unexpected CFL subdivision')
result={'schema':'workai.comsol_discretization_comparison.v1','degrees':[5,6],'h_mm':1.6,
 'internal_step_us':dt,'substeps_per_sample':[38,53],
 'both_selected_filters_are_analytic_gll_within_5e_13':True,'all_shared_weights_unchanged':True,
 'same_physical_scenes_material_time_coefficients_inference_and_references_verified':True,
 'summary_sha256':[sha(p/'summary.json') for p in folders],'plan_sha256':[sha(p) for p in plan_paths],
 'weights_sha256':[sha(p) for p in weight_paths],'analysis_source_sha256':sha(Path(__file__)),
 'completed_cases':20,'receiver_comparisons':160,'improved_case_maxima':sum(r['change_percentage_points']<-1e-8 for r in rows),
 'improved_receivers':sum(r['improved_receivers'] for r in rows),'direction_count_roundoff_relative_l2':1e-10,
 'scope':'Analytic discretization change from P5 to P6 including its automatically smaller CFL time step. This does not isolate spatial order alone and is not a weight-training gain.',
 'scientific_accuracy_certified':False,'rows':rows}
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
with args.output.with_suffix('.csv').open('w',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
