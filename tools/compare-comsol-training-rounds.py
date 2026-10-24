"""Summarize two frozen candidates against exactly the same COMSOL reference."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    old=json.loads((args.baseline/'summary.json').read_text())
    new=json.loads((args.candidate/'summary.json').read_text())
    if old['reference_identity']!=new['reference_identity'] or old['plan_sha256']!=new['plan_sha256']:
        raise ValueError('References or validation plans differ; this would confound training with reference changes')
    if not old['predictions_unchanged'] or not new['metrics_independently_recomputed']:
        raise ValueError('Frozen baseline and independently audited candidate required')
    a={r['case_id']:r for r in old['rows']};b={r['case_id']:r for r in new['rows']}
    if len(a)!=20 or len(b)!=20 or a.keys()!=b.keys():raise ValueError('Incomplete 20-case matrix')
    rows=[];direction_roundoff=1e-10
    for case,x in a.items():
        y=b[case];pa=args.baseline/'metrics'/(case+'.json');pb=args.candidate/'metrics'/(case+'.json')
        if sha(pa)!=x['new_metrics_sha256'] or sha(pb)!=y['metrics_sha256']:
            raise ValueError('Metrics changed after audit')
        ma=json.loads(pa.read_text());mb=json.loads(pb.read_text())
        if ma['reference_identity']!=new['reference_identity'] or mb['reference_identity']!=new['reference_identity']:
            raise ValueError('Per-case references differ')
        av=[r['vector']['relative_l2'] for r in ma['linear_comparison']['waveform']]
        bv=[r['vector']['relative_l2'] for r in mb['linear_comparison']['waveform']]
        if len(av)!=8 or len(bv)!=8 or any(v is None for v in av+bv):
            raise ValueError('This summary requires eight nonweak vector comparisons per case')
        if not math.isclose(max(av),x['max_linear_vector_relative_l2'],rel_tol=1e-7,abs_tol=1e-14) or not math.isclose(max(bv),y['max_linear_vector_relative_l2'],rel_tol=1e-7,abs_tol=1e-14):
            raise ValueError('Summary differs from detailed metrics')
        rows.append({'case_id':case,'baseline_max_vector_relative_l2':max(av),
                     'candidate_max_vector_relative_l2':max(bv),'change_percentage_points':100*(max(bv)-max(av)),
                     'improved_receivers':sum(v<u-direction_roundoff for u,v in zip(av,bv)),
                     'baseline_case_passed':x['linear_comparison_passed'],'candidate_case_passed':y['linear_comparison_passed']})
    result={'schema':'workai.comsol_paired_training_comparison.v1',
            'baseline_weights_sha256':old['weights_sha256'],'candidate_weights_sha256':new['weights_sha256'],
            'reference_identity':new['reference_identity'],'baseline_summary_sha256':sha(args.baseline/'summary.json'),
            'candidate_summary_sha256':sha(args.candidate/'summary.json'),'completed_cases':len(rows),
            'cases_with_lower_maximum_vector_difference':sum(r['change_percentage_points']<-100*direction_roundoff for r in rows),
            'minimum_change_for_direction_count_relative_l2':direction_roundoff,
            'direction_count_roundoff_scope':'Ignore CPU replay roundoff only when counting improvement; preserve raw metrics and every original 5% verdict',
            'receivers_with_lower_vector_difference':sum(r['improved_receivers'] for r in rows),
            'receiver_comparisons':160,'baseline_comparison_passes':old['linear_comparison_passes'],
            'candidate_comparison_passes':new['linear_comparison_passes'],
            'scope':'Actual complete 5 ms predictions against the same COMSOL linear modal reference; neither a native nonlinear comparison nor certification',
            'scientific_accuracy_certified':False,'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with args.output.with_suffix('.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
