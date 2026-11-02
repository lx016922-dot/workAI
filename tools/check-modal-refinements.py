"""Compare modal-count and time refinements from one immutable candidate basis."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphwave.modal_calibration import digest, save


def load_case(folder):
    identity = json.loads((folder / 'identity.json').read_text())
    prediction = json.loads((folder / 'prediction.json').read_text())
    metrics = json.loads((folder / 'metrics.json').read_text())
    if not prediction['completed'] or digest(folder / 'prediction.npz') != prediction['prediction_sha256']:
        raise ValueError('Incomplete or modified prediction')
    if prediction['controls'] != ['linear', 'stvk', 'full'] or metrics['prediction'] != prediction:
        raise ValueError('Prediction controls or recorded metrics changed')
    if (prediction['mode_limit'] != identity['candidate_mode_limit'] or
            prediction['step_us'] != identity['step_us'] or
            prediction['active_modes'] != identity['symmetry']['retained_modes']):
        raise ValueError('Prediction mode/time identity changed')
    scene = identity['case']['scene']
    acquisition = scene['acquisition']
    step, duration = acquisition['sample_step_us'], acquisition['duration_us']
    expected = np.arange(round(duration / step) + 1) * step
    with np.load(folder / 'prediction.npz', allow_pickle=False) as archive:
        times, traces, energy, coordinates = [archive[key] for key in ['times_us', 'traces', 'energy', 'coordinates']]
        np.testing.assert_allclose(times, expected, rtol=0, atol=1e-10)
        if len(times) != prediction['frames'] or prediction['duration_us'] != duration:
            raise ValueError('Incomplete prediction duration')
        for array, shape in [(traces, (3, len(times), len(scene['receivers']), 3)),
                             (energy, (3, len(times), 3)),
                             (coordinates, (3, len(times), prediction['active_modes']))]:
            if array.shape != shape or not np.isfinite(array).all() or np.any(array[:, 0] != 0):
                raise ValueError('Incomplete or nonfinite trajectory, or nonzero initial state')
    internal, work = energy[:, :, :2].sum(axis=-1), energy[:, :, 2]
    scale = np.maximum(np.max(abs(internal), axis=1), np.max(abs(work), axis=1))
    residual = np.max(abs(internal - work), axis=1) / np.maximum(scale, 1e-30)
    np.testing.assert_allclose(residual, prediction['energy_relative_error'], rtol=1e-12, atol=1e-14)
    if np.any(energy[:, :, 0] < 0) or np.any(residual > .01):
        raise ValueError('Prediction fails its energy-balance gate')
    return identity, prediction, metrics, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lower-modes', type=Path, required=True)
    parser.add_argument('--higher-modes', type=Path, required=True)
    parser.add_argument('--finer-time', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folders = [args.lower_modes, args.higher_modes, args.finer_time]
    cases = [load_case(folder) for folder in folders]
    low, high, fine = [case[0] for case in cases]
    for key in ['case', 'model_sha256', 'basis_sha256', 'weights_sha256', 'source_sha256', 'plan_sha256']:
        if not (low[key] == high[key] == fine[key]):
            raise ValueError('Refinement identities differ: ' + key)
    if not (low['candidate_mode_limit'] < high['candidate_mode_limit'] == fine['candidate_mode_limit'] and
            low['step_us'] == high['step_us'] == 2 * fine['step_us']):
        raise ValueError('Expected a mode refinement followed by a halved time step')
    job_ids = {case[2].get('native_reference_job_id') for case in cases}
    if len(job_ids) != 1 or None in job_ids or not all(case[2]['native_comsol_time_dependent_solve'] for case in cases):
        raise ValueError('An identical native COMSOL reference is required')
    spec = importlib.util.spec_from_file_location('checks', Path(__file__).with_name('validate-comsol-linear-matrix.py'))
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    parts = [checks.comparison.parts(dict(enumerate(case[3]))) for case in cases]
    scene = low['case']['scene']
    changes = {}
    for label, a, b in [('modal', 0, 1), ('temporal', 1, 2)]:
        changes[label] = {name: checks.check(parts[a][name], parts[b][name], scene) for name in parts[a]}
    report = {'schema': 'workai.same_basis_modal_refinements.v1',
              'duration_us': scene['acquisition']['duration_us'],
              'candidate_mode_counts': [low['candidate_mode_limit'], high['candidate_mode_limit']],
              'step_sizes_us': [high['step_us'], fine['step_us']],
              'basis_sha256': high['basis_sha256'], 'weights_sha256': high['weights_sha256'],
              'changes': changes,
              'native_comparisons': [case[2]['native_comparison'] for case in cases],
              'native_reference_job_id': next(iter(job_ids)),
              'modal_check_passed': all(value['passed'] for value in changes['modal'].values()),
              'temporal_check_passed': all(value['passed'] for value in changes['temporal'].values()),
              'reference_convergence_certified': False, 'scientific_accuracy_certified': False,
              'full_5ms_nonlinear_accuracy_verified': False,
              'files_sha256': {str(index) + '/' + name: digest(folder / name)
                  for index, folder in enumerate(folders)
                  for name in ['identity.json', 'prediction.json', 'prediction.npz', 'metrics.json']}}
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'summary.json', report)
    print(json.dumps({'duration_us': report['duration_us'], 'modal_check_passed': report['modal_check_passed'],
                      'temporal_check_passed': report['temporal_check_passed'],
                      'maximum_changes': {name: max(row['maximum_relative_l2'] for row in value.values())
                                          for name, value in changes.items()},
                      'scientific_accuracy_certified': False}), flush=True)


if __name__ == '__main__':
    main()
