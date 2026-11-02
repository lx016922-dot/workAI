"""Independent analytic momentum/energy checks on a complete candidate graph.

This does not test quadrature convergence or replace native COMSOL validation.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphwave.modal_calibration import digest, save
from graphwave.modal_reduction import attach_basis, _nonlinear_stresses
from graphwave.modal_sensitivity import modal_gradients
from graphwave.quarter_symmetry import make_quarter_mesh
from tools.comsol_nodal_reference import stresses

spec = importlib.util.spec_from_file_location('quadrature_checks', Path(__file__).with_name('validate-modal-quadrature.py'))
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--fields', type=int, default=64)
    args = p.parse_args()
    if args.fields < 64:
        p.error('At least 64 independent fields required')
    started = time.monotonic()
    repo = Path(__file__).resolve().parents[1]
    names = ['tools/validate-modal-full-physics.py', 'tools/validate-modal-quadrature.py',
             'tools/comsol_nodal_reference.py', 'graphwave/modal_reduction.py',
             'graphwave/modal_sensitivity.py', 'graphwave/buffered_inference.py',
             'graphwave/quarter_symmetry.py', 'graphwave/graded_mesh.py', 'spectral3d/mesh.py']
    hashes = {name: digest(repo/name) for name in names}
    config = json.loads((args.model/'model.json').read_text())
    training = config.get('nonlinear_basis_training', {})
    if args.seed in [training.get('train_seed'), training.get('holdout_seed')]:
        raise ValueError('A fresh analytic admission seed is required')
    for name, key in [('basis.npz', 'basis_sha256'), ('weights.npz', 'weights_sha256')]:
        if digest(args.model/name) != config[key]:
            raise ValueError('Candidate identity changed')
    if not config.get('quarter_domain'):
        raise ValueError('Explicit quarter domain required')
    with np.load(args.model/'weights.npz', allow_pickle=False) as saved:
        weights = dict(saved)
    with np.load(args.model/'basis.npz', allow_pickle=False) as saved:
        frequency, fields = saved['frequency_hz'], saved['fields']
    scene = config['scene']
    mesh = make_quarter_mesh(scene, config['degree'], config['h_mm'], config['grading_ratios'], config['axis_elements'])
    model = attach_basis(mesh, scene, weights, frequency, fields)
    reference_gradient = np.ascontiguousarray(modal_gradients(mesh, mesh['derivative'], fields).reshape(len(fields), -1))
    constants = np.array([scene['material'][k] for k in ['lambda_pa', 'mu_pa', 'l_pa', 'm_pa', 'n_pa']])
    laws = np.array([1, 2], dtype=np.int64)
    count = len(frequency)
    qscale = (abs(model['load']) + .01*np.linalg.norm(model['load'])/np.sqrt(count)) / model['omega']**2
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, states = [], []
    for index in range(args.fields):
        q = rng.normal(size=count) * qscale
        gradient = (q @ reference_gradient).reshape(-1, 3, 3)
        q *= rng.uniform(2e-5, 8e-4) / np.linalg.norm(gradient, axis=(1, 2)).max()
        states.append(q)
        g = np.stack([(q @ reference_gradient).reshape(-1, 3, 3)]*2)
        stress, energy, strain, jac = stresses(g, model['volume'], constants, laws)
        expected = checks.weak_parts(stress, energy, reference_gradient, count)
        g = np.stack([(q @ model['gradient']).reshape(-1, 3, 3)]*2)
        stress, energy, _, _ = _nonlinear_stresses(g, model['volume'], model['quadratic'], model['cubic'], laws)
        row = checks.compare(checks.weak_parts(stress, energy, model['gradient'], count), expected, model['omega'])
        row.update(field=index, maximum_reference_strain=float(strain.max()), minimum_reference_j=float(jac.min()))
        errors = list(row['relative_errors'].values())
        row['passed'] = bool(np.isfinite(errors).all() and max(errors) <= .025 and strain.max() <= .001 and jac.min() > 0)
        rows.append(row)
        if (index+1) % 8 == 0:
            print(json.dumps({'validated_fields': index+1, 'elapsed_s': time.monotonic()-started}), flush=True)
    if hashes != {name: digest(repo/name) for name in names}:
        raise ValueError('Physical admission source changed')
    maxima = {key: max(row['relative_errors'][key] for row in rows) for key in rows[0]['relative_errors']}
    report = {'model_sha256': digest(args.model/'model.json'), 'basis_sha256': config['basis_sha256'],
              'source_sha256': hashes, 'seed': args.seed, 'fields': args.fields,
              'gradient_norm_range': [2e-5, 8e-4], 'constraints_per_field': len(maxima),
              'relative_limit': .025, 'rows': rows, 'maxima': maxima,
              'passed': all(row['passed'] for row in rows),
              'comsol_data_used': False, 'quadrature_convergence_verified': False,
              'scientific_accuracy_certified': False, 'elapsed_s': time.monotonic()-started}
    np.savez_compressed(args.output/'independent-states.npz', coordinates=np.asarray(states))
    save(args.output/'report.json', report)
    shutil.copyfile(__file__, args.output/'executed-validator.py')
    print(json.dumps({'passed': report['passed'], 'maximum_relative_error': max(maxima.values()),
                      'constraints_checked': len(maxima)*args.fields, 'elapsed_s': report['elapsed_s']}), flush=True)


if __name__ == '__main__':
    main()
