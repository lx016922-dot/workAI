"""Compare two audited grids with identical weights and physical problems."""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for name in ['baseline','candidate','baseline-plan','candidate-plan','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    reports=[json.loads((path/'summary.json').read_text()) for path in [args.baseline,args.candidate]]
    plans=[]
    for path,report in zip([args.baseline_plan,args.candidate_plan],reports):
        if sha(path)!=report['plan_sha256'] or report['completed_cases']!=20 or not report['metrics_independently_recomputed']:
            raise ValueError('Complete independent audit and its exact plan required')
        plans.append(json.loads(path.read_text()))
    for key in ['weights_sha256','reference_identity','refinement_identity']:
        if reports[0][key]!=reports[1][key]:raise ValueError('This comparison must change only the grid: '+key)
    for key in ['source_sha256','temporal_sha256']:
        if plans[0][key]!=plans[1][key]:raise ValueError('Inference implementation changed')
    if plans[0]['policy']!=plans[1]['policy']:raise ValueError('Acceptance policy changed')
    normalized=[];sizes=[]
    for plan in plans:
        cases=copy.deepcopy(plan['cases']);spacing={case['h_mm'] for case in cases}
        if len(spacing)!=1:raise ValueError('A single grid size per matrix is required')
        sizes.append(spacing.pop())
        for case in cases:
            if case['scene']['numerics']['h_mm']!=case['h_mm']:
                raise ValueError('Scene and prediction grid metadata disagree')
            del case['h_mm'];del case['scene']['numerics']['h_mm']
        normalized.append(cases)
    if normalized[0]!=normalized[1] or not 0<sizes[1]<sizes[0]:
        raise ValueError('Physical scenes/order differ, or the grid did not change')
    indexes=[{row['case_id']:row for row in report['rows']} for report in reports]
    if len(indexes[0])!=20 or indexes[0].keys()!=indexes[1].keys():raise ValueError('Incomplete matrix')
    rows=[]
    for case in indexes[0]:
        vectors=[]
        for folder,index in zip([args.baseline,args.candidate],indexes):
            path=folder/'metrics'/(case+'.json')
            if sha(path)!=index[case]['metrics_sha256']:raise ValueError('Audited metrics changed')
            record=json.loads(path.read_text())
            if record['prediction']['weights_sha256']!=reports[0]['weights_sha256']:
                raise ValueError('Prediction weights differ')
            if record['reference_identity']!=reports[0]['reference_identity']:
                raise ValueError('Per-case COMSOL references differ')
            values=[row['vector']['relative_l2'] for row in record['linear_comparison']['waveform']]
            if len(values)!=8 or any(value is None or not math.isfinite(value) for value in values):
                raise ValueError('Eight nonweak vector comparisons required')
            if not math.isclose(max(values),index[case]['max_linear_vector_relative_l2'],rel_tol=1e-7,abs_tol=1e-14):
                raise ValueError('Summary differs from detailed metrics')
            vectors.append(values)
        rows.append({'case_id':case,'baseline_max_vector_relative_l2':max(vectors[0]),
                     'finer_grid_max_vector_relative_l2':max(vectors[1]),
                     'change_percentage_points':100*(max(vectors[1])-max(vectors[0])),
                     'improved_receivers':sum(b<a-1e-10 for a,b in zip(*vectors))})
    result={'schema':'workai.comsol_mesh_only_comparison.v1','weights_sha256':reports[0]['weights_sha256'],
            'baseline_h_mm':sizes[0],'finer_h_mm':sizes[1],
            'baseline_summary_sha256':sha(args.baseline/'summary.json'),
            'finer_summary_sha256':sha(args.candidate/'summary.json'),
            'baseline_plan_sha256':reports[0]['plan_sha256'],'finer_plan_sha256':reports[1]['plan_sha256'],
            'reference_identity':reports[0]['reference_identity'],
            'refinement_identity':reports[0]['refinement_identity'],
            'analysis_source_sha256':sha(Path(__file__)),
            'same_physics_order_inference_time_weights_and_references_verified':True,
            'completed_cases':20,'receiver_comparisons':160,
            'improved_case_maxima':sum(row['change_percentage_points']<-1e-8 for row in rows),
            'improved_receivers':sum(row['improved_receivers'] for row in rows),
            'scope':'Mesh-only changes in complete 5 ms comparisons; weights are identical. Separate from training gains and from native nonlinear certification.',
            'direction_count_roundoff_relative_l2':1e-10,
            'scientific_accuracy_certified':False,'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with args.output.with_suffix('.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({key:value for key,value in result.items() if key!='rows'}))


if __name__=='__main__':main()
