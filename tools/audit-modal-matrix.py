"""Recompute evidence from complete saved graph-modal trajectories.

The COMSOL references in this audit are linear eigenreferences. Nonlinear
trajectories, their modal coordinates, and their energy are retained and checked,
but this audit cannot certify nonlinear accuracy against COMSOL.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphwave.modal_calibration import digest, read_modes, save
from graphwave.modal_reduction import attach_basis, nonlinear_force
from graphwave.quarter_symmetry import make_quarter_mesh
from tools.comsol_modal_response import response

spec = importlib.util.spec_from_file_location(
    'matrix_comparison', Path(__file__).with_name('validate-comsol-linear-matrix.py'))
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


# These bound reproducibility of recorded diagnostics, not scientific error.
# The dimensionless allowance is 50 million times smaller than the 5% gate;
# the displacement allowance matches the existing raw-reference replay check.
METRIC_REPLAY_TOLERANCE = {'relative_rtol': 1e-8, 'relative_atol': 1e-9,
                         'rms_atol_m': 1e-19}


def check_recorded_metrics(recorded, recomputed, path='metrics'):
    """Check every component/band and preserve every acceptance decision.

    Re-solving a nearly identical pair of reference oscillators produces tiny
    residuals whose *relative* difference is large. Use a small absolute
    reproducibility allowance for those diagnostics, never for pass/fail.
    """
    if isinstance(recomputed, dict):
        if not isinstance(recorded, dict) or recorded.keys() != recomputed.keys():
            raise ValueError('Recorded metric schema differs: ' + path)
        for key, value in recomputed.items():
            check_recorded_metrics(recorded[key], value, path + '.' + key)
    elif isinstance(recomputed, list):
        if not isinstance(recorded, list) or len(recorded) != len(recomputed):
            raise ValueError('Recorded metric rows differ: ' + path)
        for index, value in enumerate(recomputed):
            check_recorded_metrics(recorded[index], value, f'{path}[{index}]')
    elif isinstance(recomputed, (bool, np.bool_)) or recomputed is None:
        if type(recorded) is not type(recomputed) and not (
                isinstance(recorded, bool) and isinstance(recomputed, np.bool_)):
            raise ValueError('Recorded metric decision type differs: ' + path)
        if recorded != recomputed:
            raise ValueError('Recomputed acceptance or floor classification differs: ' + path)
    else:
        if isinstance(recorded, bool) or not np.isfinite([recorded, recomputed]).all():
            raise ValueError('Nonfinite or invalid recorded metric: ' + path)
        absolute = (METRIC_REPLAY_TOLERANCE['rms_atol_m'] if path.endswith('_m')
                    else METRIC_REPLAY_TOLERANCE['relative_atol'])
        np.testing.assert_allclose(recorded, recomputed,
                                   rtol=METRIC_REPLAY_TOLERANCE['relative_rtol'],
                                   atol=absolute, err_msg=path)


def check_arrays(times, traces, energy, coordinates, scene, mode_count):
    acquisition = scene['acquisition']
    expected_times = np.arange(round(acquisition['duration_us'] / acquisition['sample_step_us']) + 1) * acquisition['sample_step_us']
    if acquisition['duration_us'] != 5000 or len(expected_times) != 25001:
        raise ValueError('Audit requires the declared complete 5 ms record')
    if not np.array_equal(times, expected_times):
        raise ValueError('Missing, repeated, shifted or shortened output time grid')
    frames, receivers = len(times), len(scene['receivers'])
    for array, shape in [(traces, (3, frames, receivers, 3)),
                         (energy, (3, frames, 3)),
                         (coordinates, (3, frames, mode_count))]:
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError('Incomplete or nonfinite saved trajectory')
        if np.any(array[:, 0] != 0):
            raise ValueError('All controls must start from zero')
    if np.any(energy[:, :, 0] < 0):
        raise ValueError('Negative kinetic energy')
    internal = energy[:, :, :2].sum(axis=-1)
    work = energy[:, :, 2]
    scale = np.maximum(np.max(abs(internal), axis=1), np.max(abs(work), axis=1))
    residual = np.max(abs(internal - work), axis=1) / np.maximum(scale, 1e-30)
    if np.any(residual > .01):
        raise ValueError('Saved record exceeds the original 1% energy guard')
    return residual


def load_model(folder):
    config = json.loads((folder / 'model.json').read_text())
    for name, key in [('basis.npz', 'basis_sha256'), ('weights.npz', 'weights_sha256')]:
        if digest(folder / name) != config[key]:
            raise ValueError('Compiled candidate artifact changed')
    if not config.get('quarter_domain'):
        raise ValueError('This audit expects the explicit quarter-domain candidate')
    with np.load(folder / 'weights.npz', allow_pickle=False) as archive:
        weights = dict(archive)
    with np.load(folder / 'basis.npz', allow_pickle=False) as archive:
        frequency, fields = archive['frequency_hz'], archive['fields']
    mesh = make_quarter_mesh(config['scene'], config['degree'], config['h_mm'],
                             config['grading_ratios'], config['axis_elements'])
    return config, attach_basis(mesh, config['scene'], weights, frequency, fields)


def case_folder(root, identifier):
    candidates = [root / identifier, root / ('graph-modal-' + identifier) / identifier]
    present = [path for path in candidates if (path / 'prediction.json').is_file()]
    if len(present) != 1:
        raise ValueError('Expected exactly one complete prediction for ' + identifier)
    return present[0]


def audit_case(folder, case, plan, plan_path, model_folder, config, model, reference, refinement):
    identity = json.loads((folder / 'identity.json').read_text())
    prediction = json.loads((folder / 'prediction.json').read_text())
    if (identity['case'] != case or identity['source_sha256'] != plan['source_sha256']
            or identity['plan_sha256'] != digest(plan_path)
            or identity['model_sha256'] != digest(model_folder / 'model.json')
            or identity['weights_sha256'] != config['weights_sha256']
            or identity['basis_sha256'] != config['basis_sha256']
            or identity['reference_arrays_loaded_before_prediction'] is not False):
        raise ValueError('Frozen prediction provenance mismatch')
    if (not prediction['completed'] or prediction['controls'] != ['linear', 'stvk', 'full']
            or prediction['active_modes'] != len(model['omega'])
            or prediction['duration_us'] != 5000 or prediction['frames'] != 25001
            or prediction['prediction_sha256'] != digest(folder / 'prediction.npz')):
        raise ValueError('Incomplete or altered three-control prediction')
    with np.load(folder / 'prediction.npz', allow_pickle=False) as archive:
        times, traces, energy, coordinates = [archive[name] for name in
                                               ['times_us', 'traces', 'energy', 'coordinates']]
        if not np.array_equal(archive['selected_indices'], np.arange(len(model['omega']))):
            raise ValueError('Unexpected candidate mode selection')
    scene = case['scene']
    residual = check_arrays(times, traces, energy, coordinates, scene, len(model['omega']))
    np.testing.assert_allclose(residual, prediction['energy_relative_error'], rtol=1e-12, atol=1e-14)
    if scene['receivers'] != config['scene']['receivers']:
        raise ValueError('Receiver geometry differs from this frozen model')
    reconstructed = (coordinates.reshape(-1, len(model['omega'])) @
                     model['receiver_fields'].reshape(len(model['omega']), -1)).reshape(traces.shape)
    np.testing.assert_allclose(traces, reconstructed, rtol=1e-11, atol=1e-20)
    # Re-evaluate conservative potential at a declared deterministic sample of
    # actual saved states. No velocity is saved, so kinetic energy is not
    # presented as independently reconstructed.
    samples = np.unique(np.linspace(0, len(times) - 1, 11).astype(int))
    potential_error = 0.
    for index in samples:
        states = coordinates[:, index]
        _, nonlinear, _, _ = nonlinear_force(model, states[1:], np.array([1, 2]))
        potential = .5 * np.sum((states * model['omega']) ** 2, axis=1)
        potential[1:] += nonlinear
        scale = np.maximum(np.max(abs(energy[:, :, 1]), axis=1), 1e-30)
        potential_error = max(potential_error, float(np.max(abs(potential - energy[:, index, 1]) / scale)))
    if potential_error > 1e-9:
        raise ValueError('Saved potential energy differs from the candidate energy')
    reference_times, expected, method = response(reference, scene)
    if not np.array_equal(times, reference_times):
        raise ValueError('COMSOL-derived reference time grid differs')
    with np.load(folder / 'comsol-linear-reference.npz', allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['times_us'], reference_times)
        np.testing.assert_allclose(archive['traces'], expected, rtol=1e-10, atol=1e-19)
    _, truncated, _ = response(reference, scene, count=160)
    _, tighter, _ = response(reference, scene, rtol=1e-12)
    _, refined, _ = response(refinement, scene)
    metrics = {'linear_comparison': comparison.check(traces[0], expected, scene),
               'reference_modal_truncation': comparison.check(truncated, expected, scene),
               'reference_time_refinement': comparison.check(expected, tighter, scene),
               'reference_mesh_refinement': comparison.check(expected, refined, scene)}
    recorded = json.loads((folder / 'metrics.json').read_text())
    for name, value in metrics.items():
        check_recorded_metrics(recorded[name], value, name)
    result = {'case_id': case['id'], 'duration_us': 5000, 'frames': len(times),
              'controls': prediction['controls'], 'source_identity': identity,
              'energy_relative_error': residual.tolist(), 'receiver_reconstruction_verified': True,
              'potential_sample_indices': samples.tolist(), 'maximum_sampled_potential_relative_error': potential_error,
              'kinetic_energy_independently_reconstructed': False,
              'reference_identity': reference['identity'], 'refinement_identity': refinement['identity'],
              'reference_method': method, 'native_nonlinear_comsol_comparison': False,
              'metric_replay_tolerance': METRIC_REPLAY_TOLERANCE,
              'all_component_and_band_acceptance_decisions_reproduced': True,
              'scientific_accuracy_certified': False, **metrics,
              'files_sha256': {p.name: digest(p) for p in sorted(folder.iterdir()) if p.is_file()}}
    compact = {'case_id': case['id'], 'maximum_vector_error_percent': 100 * metrics['linear_comparison']['maximum_vector_waveform_relative_l2'],
               'maximum_complete_error_percent': 100 * metrics['linear_comparison']['maximum_relative_l2'],
               'linear_complete_passed': metrics['linear_comparison']['passed'],
               'linear_vector_passed': all(row['vector']['passed'] for row in metrics['linear_comparison']['waveform']),
               'reference_modal_truncation_passed': metrics['reference_modal_truncation']['passed'],
               'reference_time_refinement_passed': metrics['reference_time_refinement']['passed'],
               'reference_mesh_refinement_passed': metrics['reference_mesh_refinement']['passed'],
               'maximum_energy_error_percent': 100 * float(residual.max()),
               'nonlinear_accuracy_certified': False}
    return result, compact


def main():
    parser = argparse.ArgumentParser()
    for name in ['ci-root', 'model', 'plan', 'reference', 'refinement', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan = json.loads(args.plan.read_text())
    expected = {f'{signal}-{cycles:02d}' for signal in ['hann', 'gaussian', 'rectangular', 'chirp'] for cycles in [1, 3, 5, 10, 20]}
    if len(plan['cases']) != 20 or {case['id'] for case in plan['cases']} != expected:
        raise ValueError('The complete predeclared signal/cycle matrix is required')
    for name, sha in plan['source_sha256'].items():
        if digest(repo / name) != sha:
            raise ValueError('Frozen prediction implementation changed: ' + name)
    config, model = load_model(args.model)
    reference, refinement = read_modes(args.reference), read_modes(args.refinement)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in plan['cases']:
        result, compact = audit_case(case_folder(args.ci_root, case['id']), case, plan, args.plan,
                                     args.model, config, model, reference, refinement)
        save(args.output / (case['id'] + '.json'), result)
        rows.append(compact)
        print(json.dumps(compact), flush=True)
    summary = {'schema': 'workai.graph_modal_matrix_audit.v1', 'cases': len(rows),
               'plan_sha256': digest(args.plan), 'model_sha256': digest(args.model / 'model.json'),
               'auditor_sha256': digest(__file__), 'rows': rows,
               'linear_vector_passes': sum(row['linear_vector_passed'] for row in rows),
               'linear_complete_passes': sum(row['linear_complete_passed'] for row in rows),
               'nonlinear_accuracy_certified': False, 'scientific_accuracy_certified': False,
               'remaining_requirements': ['independent converged full 5 ms nonlinear COMSOL reference',
                                          'candidate full-duration modal and temporal convergence',
                                          'all declared waveform, component and spectral checks below 5%']}
    save(args.output / 'summary.json', summary)
    save(args.output / 'files.json', {p.name: digest(p) for p in sorted(args.output.iterdir()) if p.is_file()})


if __name__ == '__main__':
    main()
