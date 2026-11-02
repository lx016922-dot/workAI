"""Verify saved offline/cloud predictions from the same immutable candidate."""
import argparse
import importlib.util
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphwave.modal_calibration import digest, save


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', type=Path, required=True)
    parser.add_argument('--cloud', type=Path, required=True)
    parser.add_argument('--minimum-duration-us', type=float, default=5000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    admission = load_module('admission', 'check-modal-refinements.py')
    comparison = load_module('comparison', 'validate-comsol-linear-matrix.py')
    first, second = [admission.load_case(folder) for folder in [args.offline, args.cloud]]
    if first[0] != second[0]:
        raise ValueError('Offline and cloud runs do not have the same complete prediction identity')
    scene = first[0]['case']['scene']
    duration = scene['acquisition']['duration_us']
    if duration < args.minimum_duration_us:
        raise ValueError('A shorter record cannot replace the requested complete replay')
    a, b = [comparison.comparison.parts(dict(enumerate(case[3]))) for case in [first, second]]
    metrics = {key: comparison.check(a[key], b[key], scene) for key in a}
    passed = True
    maximum_relative, maximum_absolute = 0., 0.
    for group in metrics.values():
        for category in ['waveform', 'spectral']:
            for receiver in group[category]:
                for value in receiver.values():
                    maximum_absolute = max(maximum_absolute, value['error_rms_m'])
                    if value['relative_l2'] is None:
                        passed &= value['error_rms_m'] <= 1e-18
                    else:
                        maximum_relative = max(maximum_relative, value['relative_l2'])
                        passed &= value['relative_l2'] <= 1e-4
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'workai.modal_offline_cloud_replay.v1', 'duration_us': duration,
              'prediction_identity': first[0], 'metrics': metrics,
              'relative_replay_limit': 1e-4, 'small_signal_absolute_replay_limit_m': 1e-18,
              'maximum_relative_replay_error': maximum_relative, 'maximum_absolute_replay_error_m': maximum_absolute,
              'offline_replay_passed': bool(passed), 'complete_5ms_reproduced': bool(passed and duration >= 5000),
              'offline_prediction_sha256': digest(args.offline / 'prediction.npz'),
              'cloud_prediction_sha256': digest(args.cloud / 'prediction.npz'),
              'scientific_accuracy_certified': False,
              'scope': 'Reproducibility only; this does not compare against an independent nonlinear COMSOL reference'}
    save(args.output / 'summary.json', report)
    print(json.dumps({key: report[key] for key in ['duration_us', 'maximum_relative_replay_error',
          'maximum_absolute_replay_error_m', 'offline_replay_passed', 'complete_5ms_reproduced']}), flush=True)


if __name__ == '__main__':
    main()
