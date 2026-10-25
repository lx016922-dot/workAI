"""Full nonlinear graph-modal prediction followed by separate COMSOL diagnostics."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from graphwave.graded_mesh import make_graded_mesh
from graphwave.modal_reduction import attach_basis,axial_reflection_subspace,propagate
from graphwave.modal_calibration import digest,save


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--step-us',type=float,required=True)
    p.add_argument('--modes',type=int,default=200);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path);p.add_argument('--check-reference',type=Path)
    p.add_argument('--native-reference',type=Path);p.add_argument('--reference-truncation',type=int,default=160)
    args=p.parse_args();repo=Path(__file__).resolve().parents[1]
    config=json.loads((args.model/'model.json').read_text());plan=json.loads(args.plan.read_text())
    for name,sha in plan['source_sha256'].items():
        if digest(repo/name)!=sha:raise ValueError('Frozen modal implementation changed: '+name)
    for name,key in [('basis.npz','basis_sha256'),('weights.npz','weights_sha256')]:
        if digest(args.model/name)!=config[key]:raise ValueError('Candidate artifact changed: '+name)
    case=next((c for c in plan['cases'] if c['id']==args.case),None)
    if case is None:raise ValueError('Case is not in the frozen signal/cycle plan')
    scene=case['scene']
    for key in ['geometry','material','boundary','constitutive']:
        if scene[key]!=config['scene'][key]:raise ValueError('Case is outside the compiled physical problem: '+key)
    if args.modes>config['exported_modes'] or args.modes<8:raise ValueError('Invalid candidate modal truncation')
    with np.load(args.model/'weights.npz',allow_pickle=False) as saved:weights=dict(saved)
    if config.get('quarter_domain'):
        from graphwave.quarter_symmetry import make_quarter_mesh
        mesh=make_quarter_mesh(scene,config['degree'],config['h_mm'],config['grading_ratios'],config['axis_elements'])
    else:mesh=make_graded_mesh(scene,config['degree'],config['h_mm'],config['grading_ratios'],config['axis_elements'])
    with np.load(args.model/'basis.npz',allow_pickle=False) as saved:
        frequency,fields=saved['frequency_hz'][:args.modes],saved['fields'][:args.modes]
    if config.get('quarter_domain'):
        selected=np.arange(len(frequency));symmetry=dict(config['symmetry'],retained_modes=len(frequency))
    else:selected,symmetry=axial_reflection_subspace(mesh,scene,weights,fields)
    model=attach_basis(mesh,scene,weights,frequency[selected],fields[selected])
    directory=args.output/args.case;directory.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    identity={'model_sha256':digest(args.model/'model.json'),'weights_sha256':config['weights_sha256'],
        'basis_sha256':config['basis_sha256'],'plan_sha256':digest(args.plan),'case':case,
        'source_sha256':plan['source_sha256'],'runner_sha256':digest(__file__),
        'step_us':args.step_us,'candidate_mode_limit':args.modes,'symmetry':symmetry,
        'reference_arrays_loaded_before_prediction':False}
    save(directory/'identity.json',identity)
    def progress(value):
        value.update(phase='running',elapsed_s=time.monotonic()-started)
        save(directory/'progress.json',value)
        if round(value['completed_us']/scene['acquisition']['sample_step_us'])%1000==0:print(json.dumps(value),flush=True)
    composition=np.asarray(json.loads((repo/'offline/model/time-coefficients.json').read_text())['weights'])
    result=propagate(model,scene,step_us=args.step_us,composition=composition,keep_coordinates=True,progress=progress)
    np.savez_compressed(directory/'prediction.npz',times_us=result['times_us'],traces=result['traces'],
                         energy=result['energy'],coordinates=result['coordinates'],selected_indices=selected)
    prediction={'completed':True,'duration_us':float(result['times_us'][-1]),'frames':len(result['times_us']),
        'controls':['linear','stvk','full'],'elapsed_s':time.monotonic()-started,
        'prediction_sha256':digest(directory/'prediction.npz'),'energy_relative_error':result['energy_relative_error'].tolist(),
        'maximum_nonlinear_strain':result['maximum_nonlinear_strain'].tolist(),'minimum_nonlinear_j':result['minimum_nonlinear_j'].tolist(),
        'maximum_fixed_displacement_m':0.,'mode_limit':args.modes,'active_modes':len(selected),'step_us':args.step_us,
        'mode_truncation_verified':False,'time_refinement_verified':False,'scientific_accuracy_certified':False}
    save(directory/'prediction.json',prediction)
    # No reference arrays are read until the actual three-control record is saved.
    spec=importlib.util.spec_from_file_location('matrix',Path(__file__).with_name('validate-comsol-linear-matrix.py'))
    compare=importlib.util.module_from_spec(spec);spec.loader.exec_module(compare)
    report={'schema':'workai.graph_modal_case_validation.v1','case_id':args.case,'prediction':prediction,
            'nonlinear_comsol_comparison_completed':False,'native_comsol_time_dependent_solve':False,
            'reference_convergence_certified':False,'scientific_accuracy_certified':False}
    if args.reference:
        from graphwave.modal_calibration import read_modes
        from tools.comsol_modal_response import response
        reference=read_modes(args.reference)
        if not 0<args.reference_truncation<len(reference['frequency_hz']):raise ValueError('Independent reference truncation must be smaller')
        times,expected,method=response(reference,scene)
        if not np.array_equal(times,result['times_us']):raise ValueError('Reference time grid differs')
        _,truncated,_=response(reference,scene,count=args.reference_truncation)
        _,tighter,_=response(reference,scene,rtol=1e-12)
        report.update(reference_identity=reference['identity'],reference_method=method,
            linear_comparison=compare.check(result['traces'][0],expected,scene),
            reference_modal_truncation=compare.check(truncated,expected,scene),
            reference_time_refinement=compare.check(expected,tighter,scene))
        np.savez_compressed(directory/'comsol-linear-reference.npz',times_us=times,traces=expected)
        if args.check_reference:
            check=read_modes(args.check_reference);_,refined,_=response(check,scene)
            report.update(reference_mesh_refinement=compare.check(expected,refined,scene),refinement_identity=check['identity'])
    if args.native_reference:
        native=compare.comparison
        raw,receipt,expected=native.load_case(args.native_reference)
        if raw['scene']!=scene:raise ValueError('Native reference scene differs from the declared prediction case')
        actual=native.parts({i:result['traces'][i] for i in range(3)});expected=native.parts(expected)
        report.update(nonlinear_comsol_comparison_completed=True,native_comsol_time_dependent_solve=True,
            native_reference_job_id=receipt['job_id'],native_reference_duration_us=scene['acquisition']['duration_us'],
            native_comparison={name:compare.check(actual[name],expected[name],scene) for name in expected})
    save(directory/'metrics.json',report);save(directory/'progress.json',{'phase':'completed','elapsed_s':time.monotonic()-started})
    print(json.dumps({'case_id':args.case,'completed':True,'scientific_accuracy_certified':False}),flush=True)

if __name__=='__main__':main()
