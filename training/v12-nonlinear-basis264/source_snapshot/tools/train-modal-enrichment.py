"""Learn nonlinear displacement shapes while preserving the complete graph energy."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphwave.buffered_inference import graph_message
from graphwave.inference import inverse_mapping
from graphwave.modal_calibration import digest, save
from graphwave.modal_enrichment import fit_space, diagonalize, static_metrics
from graphwave.modal_reduction import attach_basis
from graphwave.modal_sparse import stiffness
from graphwave.quarter_symmetry import free_dofs, make_quarter_mesh

SOURCES = ['tools/train-modal-enrichment.py', 'graphwave/modal_enrichment.py',
           'graphwave/modal_reduction.py', 'graphwave/modal_sparse.py',
           'graphwave/buffered_inference.py', 'graphwave/inference.py',
           'graphwave/modal_calibration.py', 'graphwave/modal_sensitivity.py',
           'graphwave/quarter_symmetry.py', 'graphwave/graded_mesh.py', 'spectral3d/mesh.py']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--training-fields', type=int, default=48)
    p.add_argument('--holdout-fields', type=int, default=64)
    p.add_argument('--added-modes', type=int, default=64)
    p.add_argument('--train-seed', type=int, default=2026102601)
    p.add_argument('--holdout-seed', type=int, default=2026102602)
    args = p.parse_args()
    if (args.training_fields < 32 or args.holdout_fields < 64 or
            args.train_seed == args.holdout_seed or not 0 < args.added_modes < 6*args.training_fields):
        p.error('Separate seeds, at least 32 training and 64 holdout fields required')
    started = time.monotonic()
    repo = Path(__file__).resolve().parents[1]
    hashes = {name: digest(repo/name) for name in SOURCES}
    config = json.loads((args.model/'model.json').read_text())
    if not config.get('quarter_domain'):
        raise ValueError('An explicit quarter-domain candidate is required')
    for name, key in [('basis.npz', 'basis_sha256'), ('weights.npz', 'weights_sha256')]:
        if digest(args.model/name) != config[key]:
            raise ValueError('Input model changed')
    args.output.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        target = args.output/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo/name, target)
    identity = {'schema': 'workai.nonlinear_basis_training.v1',
                'input_model_sha256': digest(args.model/'model.json'),
                'input_basis_sha256': config['basis_sha256'], 'weights_sha256': config['weights_sha256'],
                'source_sha256': hashes, 'training_fields': args.training_fields,
                'holdout_fields': args.holdout_fields, 'added_modes': args.added_modes,
                'train_seed': args.train_seed, 'holdout_seed': args.holdout_seed,
                'gradient_norm_range': [2e-5, 8e-4], 'comsol_data_used': False,
                'validation_trajectories_used': False, 'quadrature_reduced': False,
                'material_tensors_modified': False, 'online_correction': False,
                'method': 'POD of full candidate nonlinear static responses, then conservative Rayleigh-Ritz rotation'}
    save(args.output/'identity.json', identity)
    with np.load(args.model/'weights.npz', allow_pickle=False) as archive:
        weights = dict(archive)
    with np.load(args.model/'basis.npz', allow_pickle=False) as archive:
        frequency, fields = archive['frequency_hz'], archive['fields']
    scene = config['scene']
    mesh = make_quarter_mesh(scene, config['degree'], config['h_mm'], config['grading_ratios'], config['axis_elements'])
    model = attach_basis(mesh, scene, weights, frequency, fields)
    free = free_dofs(mesh, scene)
    massroot = np.sqrt(np.repeat(mesh['mass'], 3)[free])
    component = np.tile(np.arange(3), len(mesh['points']))[free]
    base = fields.reshape(len(frequency), -1)[:, free].T * massroot[:, None]
    derivative = weights['derivative_'+str(config['degree'])]
    print(json.dumps({'phase': 'assemble_candidate_tangent', 'free_dofs': int(free.sum())}), flush=True)
    raw = stiffness(mesh, derivative, weights['quadratic_tensor'])[free][:, free]
    normalized = (diags(1/massroot) @ raw @ diags(1/massroot)).tocsc()
    del raw
    factor = splu(normalized)
    print(json.dumps({'phase': 'candidate_tangent_factored', 'elapsed_s': time.monotonic()-started}), flush=True)
    inverse = inverse_mapping(mesh)
    qscale = (abs(model['load']) + .02*np.linalg.norm(model['load'])/np.sqrt(len(frequency))) / model['omega']**2
    states = {'training': [], 'holdout': []}

    def response(rng, split):
        q = rng.normal(size=len(frequency)) * qscale
        gradient = (q @ model['gradient']).reshape(-1, 3, 3)
        q *= rng.uniform(2e-5, 8e-4) / np.linalg.norm(gradient, axis=(1, 2)).max()
        states[split].append(q)
        current = (q @ fields.reshape(len(frequency), -1)).reshape(-1, 3)
        forces = []
        for law in range(3):
            force, _, strain, jac = graph_message(current, mesh['elements'], derivative, inverse,
                                                  mesh['volume_weight'], model['quadratic'], model['cubic'], law)
            if strain > .001 or jac <= 0 or not np.isfinite(force).all():
                raise ValueError('Training state violates the original physics guard')
            forces.append(force.ravel()[free])
        # Excite each equation separately; the full coupled K solve retains all
        # longitudinal/transverse responses. Equal snapshot weights prevent the
        # larger axial component from suppressing the transverse equations.
        rhs = np.column_stack([force * (component == axis) / massroot
                               for force in [forces[1]-forces[0], forces[2]-forces[1]]
                               for axis in range(3)])
        exact = factor.solve(rhs)
        relative = np.linalg.norm(normalized @ exact-rhs, axis=0) / np.linalg.norm(rhs, axis=0)
        if np.max(relative) > 1e-7:
            raise ValueError('Full candidate static solve failed its residual guard')
        return rhs, exact

    rng = np.random.default_rng(args.train_seed)
    snapshots = []
    for index in range(args.training_fields):
        _, exact = response(rng, 'training')
        snapshots.append(exact)
        if (index+1) % 8 == 0:
            print(json.dumps({'phase': 'training_fields', 'completed': index+1}), flush=True)
    snapshots = np.column_stack(snapshots)
    space, singular_squared = fit_space(base, snapshots, args.added_modes)
    omega, enriched, skew = diagonalize(space, normalized)
    del snapshots, space
    output_fields = np.zeros((len(omega), len(free)))
    output_fields[:, free] = (enriched / massroot[:, None]).T
    output_fields = output_fields.reshape(len(omega), -1, 3)
    # Verify all fixed and reflection constraints plus mass orthonormality using
    # the same admission that the actual offline predictor uses.
    checked = attach_basis(mesh, scene, weights, omega/(2*np.pi), output_fields)
    orthogonality = checked['mass_orthonormality_error']
    del checked
    rng = np.random.default_rng(args.holdout_seed)
    rows = []
    for index in range(args.holdout_fields):
        rhs, exact = response(rng, 'holdout')
        rows.append({'field': index,
                     'baseline': static_metrics(base, model['omega'], normalized, rhs, exact, component),
                     'enriched': static_metrics(enriched, omega, normalized, rhs, exact, component)})
        if (index+1) % 8 == 0:
            print(json.dumps({'phase': 'independent_static_holdout', 'completed': index+1}), flush=True)
    if hashes != {name: digest(repo/name) for name in SOURCES}:
        raise ValueError('Training implementation changed while running')
    maxima = {name: {key: max(max(row[name][key]) for row in rows) for key in rows[0][name]}
              for name in ['baseline', 'enriched']}
    report = {'identity': identity, 'force_columns': [part+'.'+axis for part in ['geometric', 'material'] for axis in ['x','y','z']],
              'holdout': rows, 'maximum_holdout_errors': maxima,
              'all_static_holdout_checks_below_2_5_percent': bool(max(maxima['enriched'].values()) < .025),
              'pod_eigenvalues': singular_squared.tolist(), 'projected_stiffness_skew': skew,
              'mass_orthonormality_error': orthogonality, 'elapsed_s': time.monotonic()-started,
              'scientific_accuracy_certified': False,
              'scope': 'Candidate-only representation learning; independent analytic and COMSOL waveform checks remain required'}
    np.savez_compressed(args.output/'manufactured-states.npz', **{k: np.asarray(v) for k,v in states.items()})
    np.savez_compressed(args.output/'basis.npz', frequency_hz=omega/(2*np.pi), fields=output_fields)
    shutil.copyfile(args.model/'weights.npz', args.output/'weights.npz')
    save(args.output/'report.json', report)
    updated = dict(config, exported_modes=len(omega), basis_sha256=digest(args.output/'basis.npz'),
                   compilation_source_sha256=hashes, mass_orthonormality_error=orthogonality,
                   compilation_seconds=time.monotonic()-started, nonlinear_basis_training=identity,
                   training_report_sha256=digest(args.output/'report.json'),
                   symmetry=dict(config['symmetry'], original_modes=len(omega), retained_modes=len(omega)),
                   modal_truncation_convergence_verified=False, scientific_accuracy_certified=False)
    save(args.output/'model.json', updated)
    print(json.dumps({'phase': 'completed', 'modes': len(omega), 'maximum_holdout_errors': maxima,
                      'elapsed_s': time.monotonic()-started, 'scientific_accuracy_certified': False}), flush=True)


if __name__ == '__main__':
    main()
