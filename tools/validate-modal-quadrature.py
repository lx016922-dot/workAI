"""Independent analytic admission for research nonlinear modal quadrature.

Uses manufactured modal states, never COMSOL/test trajectories. Successful
physical admission does not establish waveform accuracy or model convergence.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphwave.modal_calibration import digest, save
from graphwave.modal_reduction import attach_basis, _nonlinear_stresses
from graphwave.modal_sensitivity import modal_gradients
from graphwave.quarter_symmetry import make_quarter_mesh
from spectral3d.mesh import gll
from tools.comsol_nodal_reference import stresses as analytic_stresses


def weak_parts(stress, energy, gradient, modes):
    shaped = gradient.reshape(modes, -1, 3, 3)
    moments = np.einsum('lpab,mpab->lam', stress, shaped, optimize=True)
    moments = np.concatenate([moments, moments.sum(axis=1, keepdims=True)], axis=1)
    return (np.stack([moments[0], moments[1] - moments[0], moments[1]]),
            np.array([energy[0], energy[1] - energy[0], energy[1]]))


def compare(actual, expected, omega):
    errors, absolute, reference_norm = {}, {}, {}
    for part, label in enumerate(['geometric', 'material', 'total']):
        for axis, name in enumerate(['x', 'y', 'z', 'sum']):
            for scaling, scale in [('raw', np.ones(len(omega))),
                                   ('inverse_omega', 1 / omega),
                                   ('inverse_omega_squared', 1 / omega**2)]:
                key = f'{label}.{name}.{scaling}'
                numerator = np.linalg.norm((actual[0][part, axis] - expected[0][part, axis]) * scale)
                denominator = np.linalg.norm(expected[0][part, axis] * scale)
                absolute[key], reference_norm[key] = float(numerator), float(denominator)
                errors[key] = float(numerator / max(denominator, 1e-30))
        key = f'{label}.energy'
        numerator, denominator = abs(actual[1][part] - expected[1][part]), abs(expected[1][part])
        absolute[key], reference_norm[key] = float(numerator), float(denominator)
        errors[key] = float(numerator / max(denominator, 1e-30))
    return {'relative_errors': errors, 'absolute_errors': absolute, 'reference_norms': reference_norm}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--gauss', type=int, choices=range(3, 10))
    group.add_argument('--cubature', type=Path)
    parser.add_argument('--fields', type=int, default=64)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.fields < 64:
        parser.error('At least 64 fresh physical fields are required')
    started = time.monotonic()
    config = json.loads((args.model / 'model.json').read_text())
    if not config.get('quarter_domain'):
        raise ValueError('This admission requires an explicitly identified quarter-domain model')
    source_names = ['graphwave/modal_calibration.py', 'graphwave/modal_reduction.py',
                    'graphwave/modal_sensitivity.py', 'graphwave/quarter_symmetry.py',
                    'graphwave/graded_mesh.py', 'graphwave/buffered_inference.py',
                    'graphwave/inference.py', 'spectral3d/mesh.py',
                    'tools/comsol_nodal_reference.py', 'tools/validate-modal-quadrature.py']
    repo = Path(__file__).resolve().parents[1]
    hashes = {name: digest(repo / name) for name in source_names}
    for name, key in [('basis.npz', 'basis_sha256'), ('weights.npz', 'weights_sha256')]:
        if digest(args.model / name) != config[key]:
            raise ValueError('Candidate identity changed: ' + name)
    with np.load(args.model / 'weights.npz', allow_pickle=False) as archive:
        weights = dict(archive)
    with np.load(args.model / 'basis.npz', allow_pickle=False) as archive:
        frequency, fields = archive['frequency_hz'], archive['fields']
    scene, degree = config['scene'], config['degree']
    mesh = make_quarter_mesh(scene, degree, config['h_mm'], config['grading_ratios'], config['axis_elements'])
    full = attach_basis(mesh, scene, weights, frequency, fields)
    modes = len(frequency)
    candidate = dict(full)
    if args.gauss:
        x, w = leggauss(args.gauss)
        nodes = gll(degree)[0]
        interpolation = []
        for i, node in enumerate(nodes):
            others = np.delete(nodes, i)
            interpolation.append((Polynomial.fromroots(others) / np.prod(node - others))(x))
        interpolation = np.array(interpolation).T
        for power in range(degree + 1):
            np.testing.assert_allclose(interpolation @ nodes**power, x**power, rtol=0, atol=1e-12)
        gradient = full['gradient'].reshape(modes, len(mesh['elements']), *([degree + 1] * 3), 3, 3)
        candidate['gradient'] = np.ascontiguousarray(np.einsum(
            'ai,bj,ck,meijkuv->meabcuv', interpolation, interpolation, interpolation,
            gradient, optimize=True).reshape(modes, -1))
        candidate['volume'] = (np.prod(mesh['size'], axis=1)[:, None, None, None] / 8 *
                               np.einsum('i,j,k->ijk', w, w, w)[None]).ravel()
        selection = {'method': 'Gauss-Legendre polynomial lift of learned gradient', 'points_per_axis': args.gauss}
    else:
        training = json.loads(args.cubature.with_name('report.json').read_text())
        for key in ['basis_sha256', 'weights_sha256']:
            if training[key] != config[key]:
                raise ValueError('Cubature belongs to a different model: ' + key)
        if args.seed in [training['train_seed'], training['validation_seed']]:
            raise ValueError('Independent admission requires a new random seed')
        with np.load(args.cubature, allow_pickle=False) as archive:
            selected, volume = archive['point_indices'], archive['volume']
        if (selected.ndim != 1 or selected.dtype.kind not in 'iu' or
                len(np.unique(selected)) != len(selected) or
                np.any((selected < 0) | (selected >= len(full['volume']))) or
                volume.shape != selected.shape or not np.isfinite(volume).all() or np.any(volume <= 0)):
            raise ValueError('Invalid positive quadrature')
        candidate['gradient'] = np.ascontiguousarray(full['gradient'].reshape(modes, -1, 9)[:, selected].reshape(modes, -1))
        candidate['volume'] = volume
        selection = {'method': 'trained positive quadrature', 'artifact_sha256': digest(args.cubature),
                     'training_report_sha256': digest(args.cubature.with_name('report.json'))}
    np.testing.assert_allclose(candidate['volume'].sum(), full['volume'].sum(), rtol=1e-12, atol=0)
    reference_gradient = np.ascontiguousarray(modal_gradients(mesh, mesh['derivative'], fields).reshape(modes, -1))
    del fields
    constants = np.array([scene['material'][k] for k in ['lambda_pa', 'mu_pa', 'l_pa', 'm_pa', 'n_pa']])
    laws = np.array([1, 2], dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    qscale = (abs(full['load']) + .01 * np.linalg.norm(full['load']) / np.sqrt(modes)) / full['omega']**2
    args.output.mkdir(parents=True, exist_ok=False)
    rows, states = [], []
    for index in range(args.fields):
        q = rng.normal(size=modes) * qscale
        g = (q @ reference_gradient).reshape(-1, 3, 3)
        q *= rng.uniform(2e-5, 1e-4) / np.linalg.norm(g, axis=(1, 2)).max()
        states.append(q)
        g = np.stack([(q @ reference_gradient).reshape(-1, 3, 3)] * 2)
        stress, energy, strain, jac = analytic_stresses(g, full['volume'], constants, laws)
        expected = weak_parts(stress, energy, reference_gradient, modes)
        row = {'field': index, 'maximum_reference_strain': float(strain.max()), 'minimum_reference_j': float(jac.min())}
        for name, model in [('full_graph', full), ('candidate', candidate)]:
            g = np.stack([(q @ model['gradient']).reshape(-1, 3, 3)] * 2)
            stress, energy, _, _ = _nonlinear_stresses(g, model['volume'], model['quadratic'], model['cubic'], laws)
            row[name] = compare(weak_parts(stress, energy, model['gradient'], modes), expected, full['omega'])
            errors = list(row[name]['relative_errors'].values())
            row[name]['passed'] = bool(np.isfinite(errors).all() and max(errors) <= .025)
        rows.append(row)
        if (index + 1) % 8 == 0:
            print(json.dumps({'validated_fields': index + 1, 'maximum_candidate_error_so_far':
                max(max(r['candidate']['relative_errors'].values()) for r in rows), 'elapsed_s': time.monotonic() - started}), flush=True)
    if hashes != {name: digest(repo / name) for name in source_names}:
        raise ValueError('Admission source changed while running')
    np.savez_compressed(args.output / 'independent-states.npz', coordinates=np.array(states))
    report = {'schema': 'workai.independent_modal_physics_admission.v2',
              'scope': 'Geometric/material/total nonlinear weak momentum and energy on independent manufactured modal states',
              'selection': selection, 'source_sha256': hashes, 'base_weights_sha256': config['weights_sha256'],
              'base_basis_sha256': config['basis_sha256'], 'validation_seed': args.seed, 'fields': args.fields,
              'constraints_per_field': 39, 'hard_relative_limit': .025,
              'original_points': len(full['volume']), 'candidate_points': len(candidate['volume']),
              'reference_uses_learned_material_tensors': False, 'reference_uses_learned_spatial_derivative': False,
              'full_graph_passed': all(r['full_graph']['passed'] for r in rows),
              'candidate_passed': all(r['candidate']['passed'] for r in rows),
              'maximum_full_graph_error': max(max(r['full_graph']['relative_errors'].values()) for r in rows),
              'maximum_candidate_error': max(max(r['candidate']['relative_errors'].values()) for r in rows),
              'rows': rows, 'elapsed_s': time.monotonic() - started, 'scientific_accuracy_certified': False,
              'independent_states_sha256': digest(args.output / 'independent-states.npz')}
    save(args.output / 'report.json', report)
    print(json.dumps({key: report[key] for key in ['candidate_points', 'full_graph_passed', 'candidate_passed',
          'maximum_full_graph_error', 'maximum_candidate_error', 'elapsed_s']}), flush=True)


if __name__ == '__main__':
    main()
