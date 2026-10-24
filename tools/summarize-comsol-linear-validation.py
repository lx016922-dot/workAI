"""Audit every saved 5 ms neural probe block and recompute COMSOL comparisons."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from graphwave.modal_calibration import digest,read_modes
from tools.comsol_modal_response import response
spec=importlib.util.spec_from_file_location('matrix',Path(__file__).with_name('validate-comsol-linear-matrix.py'))
matrix=importlib.util.module_from_spec(spec);spec.loader.exec_module(matrix)


def same(actual,expected,path='',absolute_m=1e-23,relative_atol=1e-23):
    if isinstance(expected,dict):
        if set(actual)!=set(expected):raise ValueError('Metric keys changed')
        for key in expected:same(actual[key],expected[key],path+'.'+key,absolute_m,relative_atol)
    elif isinstance(expected,list):
        if len(actual)!=len(expected):raise ValueError('Metric length changed')
        for i,(a,b) in enumerate(zip(actual,expected)):same(a,b,path+f'[{i}]',absolute_m,relative_atol)
    elif isinstance(expected,float):
        tolerance=absolute_m if path.endswith('_m') else relative_atol if path.endswith('relative_l2') else 1e-23
        if not np.isclose(actual,expected,rtol=1e-7,atol=tolerance):raise ValueError(f'Metric recomputation differs at {path}: {actual!r} versus {expected!r}')
    elif actual!=expected:raise ValueError('Metric verdict differs')


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--plan',type=Path,default=Path('research/long-validation/plan.json'))
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--check-reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workflow-run',type=int);args=p.parse_args()
    plan=json.loads(args.plan.read_text());reference=read_modes(args.reference);refined=read_modes(args.check_reference)
    rows=[];args.output.mkdir(parents=True,exist_ok=True);(args.output/'metrics').mkdir(exist_ok=True)
    for case in plan['cases']:
        choices=[args.input/case['id']/case['id'],args.input/('comsol-linear-'+case['id'])/case['id']]
        present=[path for path in choices if (path/'metrics.json').exists()]
        if len(present)!=1:raise ValueError('Missing or ambiguous completed case: '+case['id'])
        folder=present[0];report=json.loads((folder/'metrics.json').read_text())
        identity=json.loads((folder/'identity.json').read_text());prediction=report['prediction']
        if identity['case']!=case or identity['weights_sha256']!=digest(args.weights) or identity['plan_sha256']!=digest(args.plan):
            raise ValueError('Wrong frozen prediction identity')
        if identity['inference_source_sha256']!=plan['source_sha256'] or identity['temporal_sha256']!=plan['temporal_sha256']:
            raise ValueError('Frozen inference source or time weights differ')
        if report['reference_identity']!=reference['identity'] or report['refinement_identity']!=refined['identity']:
            raise ValueError('Wrong COMSOL identity')
        if not prediction['completed'] or prediction['frames']!=25001 or prediction['duration_us']!=5000:
            raise ValueError('Incomplete 5 ms prediction')
        arrays=[];energies=[];first=0
        for block in prediction['probe_blocks']:
            name=block['file'];path=folder/name
            if Path(name).name!=name or block['first_frame']!=first or digest(path)!=block['sha256']:
                raise ValueError('Changed or discontinuous probe block')
            last=block['last_frame'];count=last-first+1
            with np.load(path,allow_pickle=False) as saved:
                value=saved['predicted'];energy=saved['energy']
                if value.shape!=(3,count,8,3) or energy.shape[:2]!=(3,count) or not np.isfinite(value).all() or not np.isfinite(energy).all():
                    raise ValueError('Invalid saved arrays')
                arrays.append(value);energies.append(energy)
            first=last+1
        if first!=25001:raise ValueError('Missing final samples')
        actual=np.concatenate(arrays,axis=1);energy=np.concatenate(energies,axis=1)
        if np.any(actual[:,0]):raise ValueError('Nonzero initial displacement')
        total=energy[:,:,1]+energy[:,:,2];work=energy[:,:,3]
        balance=np.max(abs(total-work),axis=1)/np.maximum(np.maximum(np.max(abs(total),axis=1),np.max(abs(work),axis=1)),1e-30)
        np.testing.assert_allclose(balance,prediction['energy_balance_relative'],rtol=1e-8,atol=1e-20)
        times,expected,_=response(reference,case['scene'])
        with np.load(folder/'comsol-modal-reference.npz',allow_pickle=False) as saved:
            np.testing.assert_array_equal(times,saved['times_us'])
            np.testing.assert_allclose(expected,saved['linear'],rtol=1e-7,atol=1e-20)
        checks={'linear_comparison':matrix.check(actual[0],expected,case['scene'])}
        _,truncated,_=response(reference,case['scene'],count=report['modal_truncation_counts'][0])
        _,tighter,_=response(reference,case['scene'],rtol=1e-12)
        _,fine,_=response(refined,case['scene'])
        checks.update(modal_truncation=matrix.check(truncated,expected,case['scene']),
                      time_integration_refinement=matrix.check(expected,tighter,case['scene']),
                      mesh_refinement=matrix.check(expected,fine,case['scene']))
        for key,value in checks.items():
            # Refinement differences can be at machine precision (e.g. 1e-22 m
            # for a 7e-8 m waveform). Replaying that subtraction on another CPU
            # need not reproduce its tiny relative error digit-for-digit.
            # This is only an audit roundoff allowance: every 5% / 1e-13 m
            # scientific verdict must still agree exactly below.
            waveform_scale=max(m['reference_rms_m'] for row in value['waveform'] for m in row.values())
            roundoff=max(1e-23,256*np.finfo(float).eps*waveform_scale)
            if key=='time_integration_refinement':
                # The adaptive forced oscillator integration is requested at
                # rtol=1e-11 and replayed at 1e-12. Its near-zero difference is
                # not a bitwise invariant across CPU/BLAS implementations.
                roundoff=max(roundoff,8e-11*waveform_scale)
            scales=[m['reference_rms_m'] for group in ['waveform','spectral'] for row in value[group] for m in row.values() if m['relative_l2'] is not None]
            relative_roundoff=max(256*np.finfo(float).eps,roundoff/max(1e-13,min(scales,default=1e-13)))
            same(value,report[key],case['id']+'.'+key,roundoff,relative_roundoff)
        shutil.copyfile(folder/'metrics.json',args.output/'metrics'/(case['id']+'.json'))
        row={'case_id':case['id'],'duration_us':5000,'samples':25001,'receivers':8,
             'max_linear_vector_relative_l2':checks['linear_comparison']['maximum_vector_waveform_relative_l2'],
             'linear_comparison_passed':checks['linear_comparison']['passed'],
             'modal_truncation_passed':checks['modal_truncation']['passed'],
             'time_refinement_passed':checks['time_integration_refinement']['passed'],
             'mesh_refinement_passed':checks['mesh_refinement']['passed'],
             'energy_passed':bool(np.all(balance<=.01)),
             'identity_sha256':digest(folder/'identity.json'),'metrics_sha256':digest(folder/'metrics.json'),
             'comsol_modal_reference_sha256':digest(folder/'comsol-modal-reference.npz'),
             'verified_probe_chunks':len(arrays),'scientific_accuracy_certified':False}
        rows.append(row);print(json.dumps(row),flush=True)
    summary={'schema':'workai.comsol_candidate_matrix_audit.v1','completed_cases':len(rows),
             'weights_sha256':digest(args.weights),'plan_sha256':digest(args.plan),
             'reference_identity':reference['identity'],'refinement_identity':refined['identity'],
             'workflow_run_url':f'https://github.com/lx016922-dot/workAI/actions/runs/{args.workflow_run}' if args.workflow_run else None,
             'linear_comparison_passes':sum(r['linear_comparison_passed'] for r in rows),
             'energy_passes':sum(r['energy_passed'] for r in rows),
             'modal_truncation_passes':sum(r['modal_truncation_passed'] for r in rows),
             'mesh_refinement_passes':sum(r['mesh_refinement_passed'] for r in rows),
             'time_refinement_passes':sum(r['time_refinement_passed'] for r in rows),
             'metrics_independently_recomputed':True,'native_comsol_time_dependent':False,
             'audit_roundoff_allowance':'max(1e-23 m, 256*float64 epsilon*reference waveform RMS); propagated to relative ratios; all scientific verdicts must match exactly',
             'adaptive_time_refinement_replay_allowance':'8 * requested 1e-11 integration tolerance * reference waveform RMS; comparison thresholds remain 5% and 1e-13 m',
             'nonlinear_comsol_reference':False,'long_accuracy_certified':False,'rows':rows}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (args.output/'cases.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
