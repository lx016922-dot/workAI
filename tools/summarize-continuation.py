"""Preserve the initial eight-case failure and apply its predeclared refinement."""
import argparse
import copy
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--campaign',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=args.campaign;out=args.output
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=read(root/'planned-inputs.json');initial=read(root/'metrics-buffered/metrics.json')
assert initial['planned_inputs_sha256']==sha(root/'planned-inputs.json')
assert 'At most one predeclared h refinement' in plan['reference_fallback']
assert len(initial['cases'])==8 and plan['weights_sha256']==initial['weights_sha256']
refined=root/'reference-refinement/continued-hole-fixed'
outcome=read(refined/'metrics/refinement-outcome.json')
assert outcome['original_failed_checks_retained']
protocol=read(refined/'protocol.json')
assert protocol['predeclared_plan_sha256']==sha(root/'planned-inputs.json')
assert protocol['unchanged_weights_sha256']==plan['weights_sha256']
final=copy.deepcopy(initial)
for row in final['cases']:
    if row['case_id']=='continued-hole-fixed':
        quality=read(refined/'metrics/reference-quality.json')
        row.update(reference_quality_passed=quality['passed'],reference_quality_sha256=sha(refined/'metrics/reference-quality.json'))
        for label in ['parent','continued']:
            row[label]=read(refined/'metrics'/f'{label}-metrics.json')
        row['passed']=quality['passed'] and row['continued']['passed']

def maximum(row):
    checks=list(row['field_checks'].values())+[v for p in row['probes'] for v in p['checks'].values()]
    return max(c['relative_l2'] for c in checks if c['relative_l2'] is not None)

for row in final['cases']:
    for label in ['parent','continued']:
        row[label+'_worst_relative_error']=maximum(row[label])
        assert row[label]['weights_sha256']==plan['weights_sha256' if label=='continued' else 'parent_weights_sha256']
for label in ['parent','continued']:
    final['maximums'][label]={'worst_reported_relative_error':max(maximum(r[label]) for r in final['cases']),
        **{part:max(r[label]['field_checks'][part]['relative_l2'] for r in final['cases'])
           for part in ['full_vector','material_vector','geometric_vector','material_2f_vector']}}
physics=read(root/'physics-buffered/metrics.json')
assert physics['weights_sha256']==plan['weights_sha256'] and len(physics['cases'])==8
final['all_reference_and_prediction_checks_passed']=all(r['passed'] for r in final['cases'])
final['improves_predeclared_metrics']=all(final['maximums']['continued'][k]<=final['maximums']['parent'][k]
    for k in ['worst_reported_relative_error','full_vector'])
final['physics_checks_passed']=physics['all_checks_passed']
final['eligible_for_further_release_checks']=final['all_reference_and_prediction_checks_passed'] and final['improves_predeclared_metrics'] and physics['all_checks_passed']
final['accepted_for_offline_release']=False
final['long_record_accuracy_certified']=False
final['duration_us']=14
final['timing_note']='Observed per-case wall times under concurrent workloads; not a controlled speed benchmark.'
final['initial_metrics_sha256']=sha(root/'metrics-buffered/metrics.json')
final['reference_refinement']={'case_id':'continued-hole-fixed','protocol':protocol,'outcome':outcome,'initial_failure_preserved':True}
final['completed_at_utc']=datetime.now(timezone.utc).isoformat()
final['scope']+=' All eight records are 14 us, not 5 ms. No long-record accuracy promotion.'
out.mkdir(parents=True,exist_ok=True)
for src,name in [(root/'planned-inputs.json','planned-inputs.json'),(root/'promotion-policy.json','promotion-policy.json'),
                 (root/'metrics-buffered/metrics.json','initial-metrics.json'),(root/'physics-buffered/metrics.json','physics.json')]:
    shutil.copyfile(src,out/name)
for row in initial['cases']:
    target=out/'initial-reference-quality'/f"{row['case_id']}.json";target.parent.mkdir(exist_ok=True)
    shutil.copyfile(root/'metrics-buffered'/row['case_id']/'reference-quality.json',target)
for name in ['protocol.json','metrics/reference-quality.json','metrics/parent-metrics.json','metrics/continued-metrics.json','metrics/refinement-outcome.json']:
    target=out/'hole-refinement'/Path(name).name;target.parent.mkdir(exist_ok=True)
    shutil.copyfile(refined/name,target)
(out/'metrics.json').write_text(json.dumps(final,indent=2,ensure_ascii=False)+'\n')
summary={k:v for k,v in final.items() if k!='cases'}
(out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
rows=[]
for row in final['cases']:
    item={'case_id':row['case_id'],'duration_us':14,'reference_passed':row['reference_quality_passed'],
          'continued_passed':row['continued']['passed'],'reference_and_prediction_passed':row['passed']}
    for label in ['parent','continued']:
        item[label+'_maximum_relative_error_percent']=maximum(row[label])*100
        for part in ['full_vector','material_vector','geometric_vector','material_2f_vector']:
            item[label+'_'+part+'_percent']=row[label]['field_checks'][part]['relative_l2']*100
        item[label+'_elapsed_s']=row[label]['elapsed_s']
    rows.append(item)
with (out/'cases.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(summary,ensure_ascii=False))
