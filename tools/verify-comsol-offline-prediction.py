"""Verify a fresh offline prediction against an independently audited CI record.

This checks runtime reproducibility, not agreement with physical truth. The
fixed numerical tolerance never replaces the original scientific verdicts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import scipy
import numba


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def load_record(folder):
    identity = read(folder / 'identity.json')
    metrics = read(folder / 'metrics.json')
    prediction = read(folder / 'prediction.json')
    progress = read(folder / 'progress.json')
    if prediction != metrics['prediction']:
        raise ValueError('Prediction receipt differs from metrics')
    if (not prediction['completed'] or prediction['frames'] != 25001
            or prediction['duration_us'] != 5000
            or progress['phase'] != 'completed'
            or progress['completed_frames'] != 25001
            or prediction['controls'] != ['linear', 'stvk', 'full']):
        raise ValueError('Complete 5 ms three-control prediction required')
    arrays, energies, first = [], [], 0
    for block in prediction['probe_blocks']:
        name = block['file']
        if Path(name).name != name or block['first_frame'] != first:
            raise ValueError('Invalid or discontinuous probe block')
        path = folder / name
        if sha(path) != block['sha256']:
            raise ValueError('Probe checksum mismatch')
        count = block['last_frame'] - first + 1
        with np.load(path, allow_pickle=False) as saved:
            value, energy = saved['predicted'], saved['energy']
            if value.shape != (3, count, 8, 3) or energy.shape != (3, count, 4):
                raise ValueError('Wrong probe or energy dimensions')
            if not np.isfinite(value).all() or not np.isfinite(energy).all():
                raise ValueError('Nonfinite prediction')
            arrays.append(value)
            energies.append(energy)
        first = block['last_frame'] + 1
    if first != 25001 or len(arrays) != 98:
        raise ValueError('Missing samples')
    probes = np.concatenate(arrays, axis=1)
    energy = np.concatenate(energies, axis=1)
    # Match the frozen integrator's operation order exactly: internal step
    # number * dt in seconds * 1e6. Multiplying the output index by .2 instead
    # can differ by two floating-point units without changing the time grid.
    subdivisions = round(.2 / prediction['internal_step_us'])
    dt = .2 * 1e-6 / subdivisions
    if dt * 1e6 != prediction['internal_step_us']:
        raise ValueError('Internal step does not divide the fixed sample interval')
    times = np.arange(25001) * subdivisions * dt * 1e6
    np.testing.assert_array_equal(energy[:, :, 0], np.broadcast_to(times, (3, 25001)))
    if np.any(probes[:, 0]) or prediction['maximum_fixed_displacement_m'] != 0:
        raise ValueError('Initial or fixed boundary displacement differs from zero')
    total, work = energy[:, :, 1] + energy[:, :, 2], energy[:, :, 3]
    balance = np.max(abs(total - work), axis=1) / np.maximum(
        np.maximum(np.max(abs(total), axis=1), np.max(abs(work), axis=1)), 1e-30)
    np.testing.assert_allclose(balance, prediction['energy_balance_relative'],
                               rtol=1e-8, atol=1e-20)
    if bool(np.all(balance <= .01)) != prediction['energy_within_1_percent']:
        raise ValueError('Energy verdict differs')
    return identity, metrics, probes, energy


def verdicts(value, prefix=''):
    """Include weak-signal classification and every Boolean scientific check."""
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(verdicts(child, prefix + '.' + key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(verdicts(child, prefix + '[' + str(index) + ']'))
    elif isinstance(value, bool) or value is None:
        result[prefix] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ['offline', 'ci', 'audit', 'output', 'source-root', 'runtime-root']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    offline = load_record(args.offline)
    ci = load_record(args.ci)
    audit = read(args.audit / 'summary.json')
    if audit['completed_cases'] != 20 or not audit['metrics_independently_recomputed']:
        raise ValueError('A complete independent CI audit is required')
    case = offline[0]['case']['id']
    rows = [row for row in audit['rows'] if row['case_id'] == case]
    if len(rows) != 1 or offline[0] != ci[0]:
        raise ValueError('Frozen prediction identities differ')
    for name, key in [('identity.json', 'identity_sha256'),
                      ('metrics.json', 'metrics_sha256'),
                      ('comsol-modal-reference.npz', 'comsol_modal_reference_sha256')]:
        if sha(args.ci / name) != rows[0][key]:
            raise ValueError('CI record differs from audited evidence')
    if offline[0]['weights_sha256'] != audit['weights_sha256']:
        raise ValueError('Wrong audited weights')
    if args.source_root.resolve() == Path(__file__).resolve().parents[1]:
        raise ValueError('An independent source directory is required')
    for name, digest in offline[0]['inference_source_sha256'].items():
        if sha(args.source_root / name) != digest:
            raise ValueError('Independent inference source differs: ' + name)
    if sha(args.source_root / 'tools/run-comsol-linear-case.py') != offline[0]['runner_sha256']:
        raise ValueError('Independent prediction runner differs')
    runtime = args.runtime_root.resolve()
    if not Path(sys.executable).resolve().is_relative_to(runtime / 'runtime'):
        raise ValueError('Use the specified portable Python runtime')
    manifest_path = runtime / 'MANIFEST.json'
    manifest = read(manifest_path)
    runtime_entries = [entry for entry in manifest['files'] if entry['path'].startswith('runtime/')]
    if manifest['platform'] != 'linux' or not runtime_entries:
        raise ValueError('Linux portable runtime manifest required')
    for entry in runtime_entries:
        path = runtime / entry['path']
        if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError('Portable runtime integrity check failed: ' + entry['path'])
    checks = ['linear_comparison', 'modal_truncation',
              'time_integration_refinement', 'mesh_refinement']
    for key in checks:
        if verdicts(offline[1][key]) != verdicts(ci[1][key]):
            raise ValueError('Scientific verdict or weak-signal classification changed: ' + key)
    for key in ['reference_identity', 'refinement_identity', 'modal_truncation_counts']:
        if offline[1][key] != ci[1][key]:
            raise ValueError('Different COMSOL reference: ' + key)

    # Fixed before comparing either output: match the existing independent
    # audit's 1e-7 relative floating-point allowance. RMS absolute floors are
    # far below the unchanged scientific weak-signal threshold (1e-13 m).
    relative_tolerance, displacement_floor, energy_floor = 1e-7, 1e-23, 1e-23
    error = np.sqrt(np.mean((offline[2] - ci[2]) ** 2, axis=1))
    scale = np.sqrt(np.mean(ci[2] ** 2, axis=1))
    if np.any(error > relative_tolerance * scale + displacement_floor):
        raise ValueError('Offline displacement differs from CI beyond roundoff allowance')
    energy_error = np.sqrt(np.mean((offline[3][:, :, 1:] - ci[3][:, :, 1:]) ** 2, axis=1))
    energy_scale = np.sqrt(np.mean(ci[3][:, :, 1:] ** 2, axis=1))
    if np.any(energy_error > relative_tolerance * energy_scale + energy_floor):
        raise ValueError('Offline energy differs from CI beyond roundoff allowance')
    if args.output.exists():
        raise ValueError('Use a new archive output directory')
    args.output.mkdir(parents=True)
    for name in ['identity.json', 'prediction.json', 'progress.json', 'metrics.json',
                 'comsol-modal-reference.npz']:
        shutil.copyfile(args.offline / name, args.output / name)
    np.savez_compressed(args.output / 'traces.npz',
                        times_us=offline[3][0, :, 0],
                        predicted=offline[2], energy=offline[3])
    receipt = {
        'schema': 'workai.offline_fresh_prediction_verification.v1',
        'case_id': case, 'frames': 25001, 'duration_us': 5000,
        'controls': ['linear', 'stvk', 'full'], 'receivers': 8,
        'original_probe_blocks_verified_per_run': 98,
        'full_time_grid_matches_frozen_internal_step_formula': True,
        'fresh_prediction_completed': True,
        'code_loaded_from_independent_directory': True,
        'runtime_kind': 'Linux portable build using release-pinned dependencies',
        'release_archive_downloaded_for_this_check': False,
        'runtime_manifest_sha256': sha(manifest_path),
        'runtime_manifest_files_verified': len(runtime_entries),
        'network_required': False,
        'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__,
                    'scipy': scipy.__version__, 'numba': numba.__version__},
        'elapsed_s': offline[1]['prediction']['elapsed_s'],
        'weights_sha256': offline[0]['weights_sha256'],
        'ci_workflow_run_url': audit['workflow_run_url'],
        'ci_audit_sha256': sha(args.audit / 'summary.json'),
        'ci_identity_sha256': rows[0]['identity_sha256'],
        'ci_metrics_sha256': rows[0]['metrics_sha256'],
        'relative_roundoff_tolerance': relative_tolerance,
        'displacement_rms_absolute_floor_m': displacement_floor,
        'energy_rms_absolute_floor_j': energy_floor,
        'all_72_displacement_component_checks_passed': True,
        'maximum_component_difference_rms_m': float(error.max()),
        'maximum_component_relative_l2': float(np.max(error / np.maximum(scale, 1e-30))),
        'weak_component_count_at_scientific_1e_13_m_floor': int(np.sum(scale <= 1e-13)),
        'maximum_nonweak_component_relative_l2': (float(np.max(error[scale > 1e-13] / scale[scale > 1e-13]))
                                                  if np.any(scale > 1e-13) else None),
        'all_energy_columns_match': True,
        'maximum_energy_column_relative_l2': float(np.max(energy_error / np.maximum(energy_scale, 1e-30))),
        'original_scientific_verdicts_and_weak_signal_classifications_match': True,
        'energy_within_1_percent': offline[1]['prediction']['energy_within_1_percent'],
        'linear_comparison_passed': offline[1]['linear_comparison']['passed'],
        'scientific_accuracy_certified': False,
        'scope': 'Fresh offline neural rollout compared with the audited CI rollout; runtime reproducibility is separate from COMSOL accuracy.',
        'verification_source_sha256': sha(Path(__file__)),
        'retained_files': {path.name: {'bytes': path.stat().st_size, 'sha256': sha(path)}
                           for path in sorted(args.output.iterdir())},
    }
    (args.output / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
