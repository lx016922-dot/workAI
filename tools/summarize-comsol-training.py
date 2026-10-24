"""Publish inspectable COMSOL training receipts without optimizer checkpoints."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from graphwave.modal_calibration import read_modes,match,digest
from spectral3d.mesh import make_mesh


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--check-reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    config=json.loads((args.run/'config.json').read_text());training=json.loads((args.run/'training.json').read_text())
    ref,check=read_modes(args.reference),read_modes(args.check_reference)
    if ref['identity']!=config['reference_identity'] or check['identity']!=config['refinement_identity']:
        raise ValueError('Training label identity mismatch')
    if digest(args.run/'candidate-weights.npz')!=training['weights_sha256']:
        raise ValueError('Training weights changed')
    snapshot=args.run/'source_snapshot/modal_calibration.py'
    if digest(snapshot)!=config['training_source_sha256']:raise ValueError('Training source changed')
    if config.get('physics_constraints'):
        if digest(args.run/'source_snapshot/nonlinear_constraints.py')!=config['physics_constraints']['physical_equations_source_sha256']:
            raise ValueError('Physical equation source changed')
        for split in ['training','validation']:
            if digest(args.run/f'physics-{split}-patches.npz')!=config['physics_constraints'][f'{split}_patches_sha256']:
                raise ValueError('Physical constraint labels changed')
    mesh=make_mesh(ref['scene'],4,2.5);order,mac=match(ref['fields'],check['fields'],mesh['mass'])
    target=check['frequency_hz'][order]
    if args.output.exists() and (not (args.output/'config.json').exists() or digest(args.output/'config.json')!=digest(args.run/'config.json')):
        raise ValueError('Output already belongs to another training run')
    args.output.mkdir(parents=True,exist_ok=True)
    for name in ['config.json','training.json','history.json','candidate-weights.npz']:
        shutil.copyfile(args.run/name,args.output/name)
    (args.output/'source_snapshot').mkdir(exist_ok=True);shutil.copyfile(snapshot,args.output/'source_snapshot/modal_calibration.py')
    if config.get('physics_constraints'):
        shutil.copyfile(args.run/'source_snapshot/nonlinear_constraints.py',args.output/'source_snapshot/nonlinear_constraints.py')
        for name in ['physics-initial.json','physics-training.json','physics-validation.json','physics-training-patches.npz','physics-validation-patches.npz']:
            shutil.copyfile(args.run/name,args.output/name)
    if config.get('postfit_feasibility_restoration'):
        restoration=config['postfit_feasibility_restoration']
        for name,key in [('finalize-comsol-training.py','source_sha256'),('feasibility-restoration.py','algorithm_source_sha256')]:
            if digest(args.run/'source_snapshot'/name)!=restoration[key]:raise ValueError('Restoration source changed')
            shutil.copyfile(args.run/'source_snapshot'/name,args.output/'source_snapshot'/name)
        shutil.copyfile(args.run/'pre-restoration-training.json',args.output/'pre-restoration-training.json')
    rows=[]
    for i,freq in enumerate(target):
        before=training['initial_frequency_errors_hz'][i];after=training['final_frequency_errors_hz'][i]
        rows.append({'mode_index':i,'used_for_training':i in training['training_mode_indices'],
                     'comsol_reference_hz':float(freq),'reference_refinement_difference_hz':float(abs(freq-ref['frequency_hz'][i])),
                     'initial_error_hz':before,'candidate_error_hz':after,
                     'initial_phase_drift_degrees_at_5ms':before*1.8,'candidate_phase_drift_degrees_at_5ms':after*1.8})
    with (args.output/'frequencies.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'schema':'workai.comsol_training_summary.v1','weights_sha256':training['weights_sha256'],
             'actual_comsol_development_jobs':[ref['identity']['job_id'],check['identity']['job_id']],
             'training_mode_indices':training['training_mode_indices'],
             'training_initial_frequency_error_norm_hz':training['initial_training_error_norm_hz'],
             'training_final_frequency_error_norm_hz':training['final_training_error_norm_hz'],
             'optimizer_success':training['optimizer_success'],
             'feasibility_restoration':training.get('feasibility_restoration'),
             'polynomial_consistency_order':config.get('polynomial_consistency_order',2),
             'material_energy_coefficients_changed':False,'transient_test_waveforms_used_for_training':False,
             'nonlinear_longitudinal_transverse_physics_in_loss':training.get('nonlinear_longitudinal_transverse_physics_in_loss',False),
             'physics_training_passed':training.get('physics_training_passed'),
             'hard_training_limit':training.get('hard_training_limit'),
             'hard_training_limit_satisfied':training.get('hard_training_limit_satisfied'),
             'physics_training_maxima':training.get('physics_training_maxima'),
             'physics_validation_passed':training.get('physics_validation_passed'),
             'physics_validation_maxima':training.get('physics_validation_maxima'),
             'reference_long_phase_converged':config['long_phase_reference_verified'],
             'accepted_for_offline_release':False,'long_record_accuracy_certified':False,
             'scope':'P4 graph filter research candidate for the fixed-end 8 x 3 x 1.5 mm aluminum plate with h=2.5 mm; not a general geometry calibration',
             'files_sha256':{str(path.relative_to(args.output)):digest(path) for path in args.output.rglob('*') if path.is_file() and path.name!='summary.json'}}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
