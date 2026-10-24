"""Real 5 ms NN rollout followed by a separate COMSOL linear modal comparison.

The native COMSOL nonlinear verification remains a different, explicitly
reported requirement. Reference arrays are first loaded after NN completion.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from numba import set_num_threads
from graphwave.long_campaign import Driver
from graphwave.long_rollout import atomic,digest,save_arrays


def run(args):
    repo=Path(__file__).resolve().parents[1]
    plan=json.loads(args.plan.read_text())
    for name,sha in plan['source_sha256'].items():
        if digest(repo/name)!=sha:raise ValueError('Frozen inference implementation changed: '+name)
    selected=[c for c in plan['cases'] if c['id']==args.case]
    if len(selected)!=1:raise ValueError('Unknown predeclared signal/cycle case')
    case=selected[0];scene=case['scene'];acq=scene['acquisition']
    if acq!={'duration_us':5000.,'sample_step_us':.2}:raise ValueError('Expected full 5 ms record')
    directory=args.output/args.case;directory.mkdir(parents=True,exist_ok=False)
    identity={'case':case,'weights_sha256':digest(args.weights),'temporal_sha256':digest(args.temporal),
              'plan_sha256':digest(args.plan),'runner_sha256':digest(__file__),
              'inference_source_sha256':plan['source_sha256'],'numerical_corrector':False,
              'reference_arrays_loaded_before_prediction':False}
    atomic(directory/'identity.json',identity)
    set_num_threads(3)
    with np.load(args.weights,allow_pickle=False) as file:weights=dict(file)
    driver=Driver(scene,case['candidate_degree'],case['h_mm'],weights,json.loads(args.temporal.read_text()))
    if driver.reference:raise AssertionError('Neural branch required')
    frames=25001;blocks=[];started=time.monotonic();balance=np.zeros(3);scale=np.zeros(3);fixed_error=0.
    for first in range(0,frames,256):
        last=min(frames-1,first+255);fields,probes,energy=driver.advance(first,last)
        if not np.isfinite(fields).all() or not np.isfinite(energy).all():raise ValueError('Nonfinite prediction')
        if first==0 and np.any(fields[:,0]):raise ValueError('Initial displacement differs from zero')
        if len(driver.fixed):fixed_error=max(fixed_error,float(np.max(abs(fields[:,:,driver.fixed]))))
        total=energy[:,:,1]+energy[:,:,2];work=energy[:,:,3]
        balance=np.maximum(balance,np.max(abs(total-work),axis=1))
        scale=np.maximum(scale,np.maximum(np.max(abs(total),axis=1),np.max(abs(work),axis=1)))
        path=directory/f'probes-{first:06d}.npz';save_arrays(path,predicted=probes,energy=energy)
        blocks.append({'file':path.name,'sha256':digest(path),'first_frame':first,'last_frame':last})
        progress={'phase':'running','completed_time_us':last*.2,'duration_us':5000,
                  'completed_frames':last+1,'elapsed_s':time.monotonic()-started}
        atomic(directory/'progress.json',progress);print(json.dumps(progress),flush=True)
    rows=[]
    for block in blocks:
        with np.load(directory/block['file'],allow_pickle=False) as file:rows.append(file['predicted'])
    actual=np.concatenate(rows,axis=1)
    energy_error=balance/np.maximum(scale,1e-30)
    prediction={'completed':True,'frames':frames,'duration_us':5000.,'controls':['linear','stvk','full'],
                'probe_blocks':blocks,'weights_sha256':identity['weights_sha256'],
                'internal_step_us':driver.dt*1e6,'energy_balance_relative':energy_error.tolist(),
                'energy_within_1_percent':bool(np.all(energy_error<=.01)),
                'maximum_fixed_displacement_m':fixed_error,'max_strain':driver.state['strains'].tolist(),
                'min_J':driver.state['jacobians'].tolist(),'elapsed_s':time.monotonic()-started}
    atomic(directory/'prediction.json',prediction)

    # COMSOL labels cannot affect the saved neural trajectory above.
    from graphwave.modal_calibration import read_modes
    from tools.comsol_modal_response import response
    spec=importlib.util.spec_from_file_location('matrix',Path(__file__).with_name('validate-comsol-linear-matrix.py'))
    compare=importlib.util.module_from_spec(spec);spec.loader.exec_module(compare)
    reference=read_modes(args.reference)
    if not 0<args.truncation<len(reference['frequency_hz']):raise ValueError('Truncation check needs strictly fewer modes')
    times,expected,method=response(reference,scene)
    _,truncated,_=response(reference,scene,count=args.truncation)
    _,tighter,_=response(reference,scene,rtol=1e-12)
    save_arrays(directory/'comsol-modal-reference.npz',times_us=times,linear=expected)
    report={'schema':'workai.comsol_candidate_linear_validation.v1','case_id':args.case,
            'prediction':prediction,'reference_identity':reference['identity'],'reference_method':method,
            'linear_comparison':compare.check(actual[0],expected,scene),
            'modal_truncation':compare.check(truncated,expected,scene),
            'time_integration_refinement':compare.check(expected,tighter,scene),
            'modal_truncation_counts':[args.truncation,len(reference['frequency_hz'])],
            'nonlinear_comsol_comparison_completed':False,'native_comsol_time_dependent_solve':False,
            'reference_convergence_certified':False,'long_record_accuracy_certified':False}
    if args.check_reference:
        check=read_modes(args.check_reference);_,refined,_=response(check,scene)
        report['mesh_refinement']=compare.check(expected,refined,scene)
        report['refinement_identity']=check['identity']
    atomic(directory/'metrics.json',report)
    atomic(directory/'progress.json',dict(progress,phase='completed'))
    print(json.dumps({'case_id':args.case,'completed':True,'linear_comparison_passed':report['linear_comparison']['passed'],
                      'long_record_accuracy_certified':False}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True)
    p.add_argument('--plan',type=Path,default=Path('research/long-validation/plan.json'))
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--temporal',type=Path,default=Path('offline/model/time-coefficients.json'))
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--check-reference',type=Path)
    p.add_argument('--truncation',type=int,default=48);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
