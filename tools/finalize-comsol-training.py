"""Restore strict training feasibility without rerunning or hiding an optimizer."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from graphwave import nonlinear_constraints as physics
from graphwave.modal_calibration import digest,read_modes,filter_basis,project_filter_consistency,restore_physical_feasibility,modes,match,save
from spectral3d.mesh import make_mesh


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=Path('offline/model/continued-weights.npz'))
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--check-reference',type=Path,required=True)
    p.add_argument('--validation-seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();config=json.loads((args.run/'config.json').read_text());original=json.loads((args.run/'training.json').read_text())
    if args.output.exists():raise ValueError('Use a new output directory')
    if digest(args.baseline)!=config['baseline_weights_sha256'] or digest(args.run/'candidate-weights.npz')!=original['weights_sha256']:
        raise ValueError('Input weights changed')
    constraints=config['physics_constraints']
    if digest(Path(physics.__file__))!=constraints['physical_equations_source_sha256']:
        raise ValueError('Restore with the same physical equations')
    reference,check=read_modes(args.reference),read_modes(args.check_reference)
    if reference['identity']!=config['reference_identity'] or check['identity']!=config['refinement_identity']:
        raise ValueError('COMSOL identity changed')
    scene=reference['scene'];mesh=make_mesh(scene,4,2.5)
    with np.load(args.baseline,allow_pickle=False) as f:weights=dict(f)
    baseline=weights['derivative_4'];basis=filter_basis(mesh['gll'],config['polynomial_consistency_order'])
    if config.get('initial_filter_consistency_projection'):
        baseline=project_filter_consistency(baseline,mesh['gll'],config['polynomial_consistency_order'])
    theta=np.array(original['parameters'])
    with np.load(args.run/'candidate-weights.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(f['derivative_4'],baseline+np.einsum('a,aij->ij',theta,basis))
    dataset=physics.make_dataset(scene['material'],constraints['training_samples'],constraints['seed'])
    if digest(args.run/'physics-training-patches.npz')!=constraints['training_patches_sha256']:
        raise ValueError('Stored training physics changed')
    with np.load(args.run/'physics-training-patches.npz',allow_pickle=False) as saved:
        for key in saved.files:np.testing.assert_array_equal(saved[key],np.stack([item[key] for item in dataset]))
    final,restoration=restore_physical_feasibility(theta,baseline,basis,weights,dataset,constraints['hard_training_limit'])
    derivative=baseline+np.einsum('a,aij->ij',final,basis)
    # Only after selection is complete, generate a fresh independent check set.
    if args.validation_seed in [constraints['seed'],constraints['seed']+1]:raise ValueError('Use a fresh validation seed')
    validation=physics.make_dataset(scene['material'],constraints['validation_samples'],args.validation_seed)
    _,training_physics=physics.evaluate(derivative,weights,dataset,details=True)
    _,validation_physics=physics.evaluate(derivative,weights,validation,details=True)
    mapping,_=match(reference['fields'],check['fields'],mesh['mass']);target=check['frequency_hz'][mapping]
    frequencies,fields=modes(mesh,scene,derivative,weights['quadratic_tensor'],len(reference['frequency_hz']))
    order,mac=match(reference['fields'],fields,mesh['mass']);errors=frequencies[order]-target
    args.output.mkdir(parents=True)
    for name in ['config.json','training.json','history.json','physics-initial.json','physics-training-patches.npz']:
        shutil.copyfile(args.run/name,args.output/name)
    shutil.copytree(args.run/'source_snapshot',args.output/'source_snapshot')
    shutil.copyfile(args.run/'training.json',args.output/'pre-restoration-training.json')
    shutil.copyfile(Path(__file__),args.output/'source_snapshot/finalize-comsol-training.py')
    shutil.copyfile(Path(__file__).parents[1]/'graphwave/modal_calibration.py',args.output/'source_snapshot/feasibility-restoration.py')
    weights['derivative_4']=derivative;np.savez(args.output/'candidate-weights.npz',**weights)
    numeric={key:np.stack([item[key] for item in validation]) for key in validation[0] if isinstance(validation[0][key],np.ndarray)}
    np.savez_compressed(args.output/'physics-validation-patches.npz',**numeric)
    constraints['validation_seed']=args.validation_seed
    constraints['validation_patches_sha256']=digest(args.output/'physics-validation-patches.npz')
    config['postfit_feasibility_restoration']={**restoration,'parent_weights_sha256':original['weights_sha256'],
        'source_sha256':digest(Path(__file__)),'algorithm_source_sha256':digest(args.output/'source_snapshot/feasibility-restoration.py')}
    original.update(parameters=final.tolist(),raw_optimizer_parameters=theta.tolist(),feasibility_restoration=restoration,
        weights_sha256=digest(args.output/'candidate-weights.npz'),final_frequency_errors_hz=errors.tolist(),
        final_training_error_norm_hz=float(np.linalg.norm(errors[original['training_mode_indices']])),final_mode_mac=mac.tolist(),
        physics_training_passed=training_physics['all_physical_checks_within_5_percent'],
        hard_training_limit_satisfied=max(training_physics['maximum_normalized_residual'].values())<=constraints['hard_training_limit'],
        physics_training_maxima=training_physics['maximum_normalized_residual'],
        physics_validation_passed=validation_physics['all_physical_checks_within_5_percent'],
        physics_validation_maxima=validation_physics['maximum_normalized_residual'])
    save(args.output/'config.json',config);save(args.output/'training.json',original)
    save(args.output/'physics-training.json',training_physics);save(args.output/'physics-validation.json',validation_physics)
    print(json.dumps({'output':str(args.output),'raw_optimizer_success':original['optimizer_success'],
                      'restoration':restoration,'independent_physics_passed':original['physics_validation_passed'],
                      'frequency_error_norm_hz':original['final_training_error_norm_hz']}),flush=True)


if __name__=='__main__':main()
