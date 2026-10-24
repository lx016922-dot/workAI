"""Compare an audited frozen candidate with new COMSOL labels without inference.

The original predictions and reports remain immutable. This isolates changes
caused by the reference from changes caused by subsequent model training.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from graphwave.modal_calibration import read_modes,digest
from tools.comsol_modal_response import response
spec=importlib.util.spec_from_file_location('matrix',Path(__file__).with_name('validate-comsol-linear-matrix.py'))
matrix=importlib.util.module_from_spec(spec);spec.loader.exec_module(matrix)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--plan',type=Path,default=Path('research/long-validation/plan.json'))
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    audit=json.loads(args.audit.read_text());plan=json.loads(args.plan.read_text())
    if not audit['metrics_independently_recomputed'] or audit['completed_cases']!=20 or audit['plan_sha256']!=digest(args.plan):
        raise ValueError('A complete independent audit for the unchanged plan is required')
    cases={c['id']:c for c in plan['cases']}
    if len(audit['rows'])!=20 or {r['case_id'] for r in audit['rows']}!=set(cases):
        raise ValueError('Incomplete or duplicate case matrix')
    reference=read_modes(args.reference)
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'metrics').mkdir()
    rows=[]
    for receipt in audit['rows']:
        case=receipt['case_id'];choices=[args.input/case/case,args.input/('comsol-linear-'+case)/case,args.input/case]
        present=[f for f in choices if (f/'metrics.json').exists()]
        if len(present)!=1:raise ValueError('Missing or ambiguous frozen record: '+case)
        folder=present[0]
        for name,key in [('identity.json','identity_sha256'),('metrics.json','metrics_sha256')]:
            if digest(folder/name)!=receipt[key]:raise ValueError('Frozen identity or metrics changed')
        identity=json.loads((folder/'identity.json').read_text())
        original=json.loads((folder/'metrics.json').read_text())
        if identity['case']!=cases[case] or identity['weights_sha256']!=audit['weights_sha256']:
            raise ValueError('Wrong frozen model or scene')
        if identity['reference_arrays_loaded_before_prediction'] or identity['numerical_corrector']:
            raise ValueError('Reference-independent prediction required')
        chunks=[];first=0
        for block in original['prediction']['probe_blocks']:
            name=block['file']
            if Path(name).name!=name or block['first_frame']!=first or digest(folder/name)!=block['sha256']:
                raise ValueError('Changed or discontinuous frozen probe block')
            with np.load(folder/name,allow_pickle=False) as saved:
                values=saved['predicted'][0]
                if len(values)!=block['last_frame']-first+1 or not np.isfinite(values).all():
                    raise ValueError('Malformed frozen probe data')
                chunks.append(values)
            first=block['last_frame']+1
        actual=np.concatenate(chunks)
        times,expected,method=response(reference,cases[case]['scene'])
        if first!=25001 or actual.shape!=(25001,8,3) or expected.shape!=actual.shape or times[-1]!=5000:
            raise ValueError('Incomplete actual 5 ms record')
        comparison=matrix.check(actual,expected,cases[case]['scene'])
        report={'schema':'workai.comsol_frozen_recomparison.v1','case_id':case,
                'weights_sha256':audit['weights_sha256'],'original_identity_sha256':receipt['identity_sha256'],
                'original_metrics_sha256':receipt['metrics_sha256'],'reference_identity':reference['identity'],
                'reference_method':method,'linear_comparison':comparison,'verified_probe_chunks':len(chunks),
                'predictions_unchanged':True,'native_comsol_time_dependent':False,
                'nonlinear_comsol_reference':False,'scientific_accuracy_certified':False}
        target=args.output/'metrics'/(case+'.json')
        target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        row={'case_id':case,'max_linear_vector_relative_l2':comparison['maximum_vector_waveform_relative_l2'],
             'linear_comparison_passed':comparison['passed'],'new_metrics_sha256':digest(target),
             'verified_probe_chunks':len(chunks)}
        rows.append(row);print(json.dumps(row),flush=True)
    result={'schema':'workai.comsol_frozen_recomparison_matrix.v1','completed_cases':len(rows),
            'weights_sha256':audit['weights_sha256'],'source_audit_sha256':digest(args.audit),
            'plan_sha256':digest(args.plan),'original_workflow_run_url':audit['workflow_run_url'],
            'reference_identity':reference['identity'],'linear_comparison_passes':sum(r['linear_comparison_passed'] for r in rows),
            'predictions_unchanged':True,'reference_arrays_loaded_before_prediction':False,
            'scientific_accuracy_certified':False,'rows':rows}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (args.output/'cases.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
