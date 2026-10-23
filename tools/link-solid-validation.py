"""Attach separately computed comparison receipts without changing solver output."""
import hashlib,json
from pathlib import Path
ROOT=Path('/workspace/workai-remote');evidence=ROOT/'validation/20261022-solid'
records=json.loads((evidence/'convergence.json').read_text())['cases']
refined=json.loads((evidence/'notch-refinement.json').read_text())
final=[r for r in records if r['kind']!='notch']+[refined]
summary=[]
for r in final:
    refs=[]
    for key in ['coarse','fine','timecheck']:
        job=ROOT/'solid/jobs'/r[key];data=(job/'result.json').read_bytes();status=json.loads((job/'status.json').read_text())
        digest=hashlib.sha256(data).hexdigest();assert status['phase']=='completed' and digest==status['result_sha256']
        refs.append({'job_id':r[key],'result_sha256':digest})
    for key in ['fine','timecheck']:
        p=ROOT/'solid/jobs'/r[key]/'validation.json'
        receipt={'schema':'workai.solid.comparison.v1','job_id':r[key],'result_sha256':next(a['result_sha256'] for a in refs if a['job_id']==r[key]),'references':refs,
                 'spatial_with_changed_stable_step':r['spatial_with_changed_stable_step'],'temporal_same_mesh':r['temporal_same_mesh'],'not_global_error_bound':True,'comsol_agreement_verified':False}
        text=json.dumps(receipt,ensure_ascii=False,indent=2)+'\n'
        if p.exists():assert p.read_text()==text
        else:p.write_text(text)
    summary.append({'kind':r['kind'].replace('_refined',''),'display_job':r['timecheck'],'space_max_relative_l2':max(p['vector_relative_l2'] for p in r['spatial_with_changed_stable_step']['rows']),
                    'time_max_relative_l2':max(p['vector_relative_l2'] for p in r['temporal_same_mesh']['rows']),'selected_receivers_passed':r['spatial_with_changed_stable_step']['selected_receivers_within_5_percent'] and r['temporal_same_mesh']['selected_receivers_within_5_percent']})
(evidence/'final-cases.json').write_text(json.dumps({'cases':summary,'prior_notch_failure_preserved':True,'not_independent_comsol_validation':True},indent=2)+'\n')
print(json.dumps(summary))
