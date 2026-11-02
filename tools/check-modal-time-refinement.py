"""Compare saved predictions at two time steps without changing scientific gates."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphwave.modal_calibration import digest, save


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coarse', type=Path, required=True)
    parser.add_argument('--fine', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    loader = module('case_loader', 'check-modal-refinements.py')
    comparison = module('case_comparison', 'validate-comsol-linear-matrix.py')
    cases = [loader.load_case(path) for path in [args.coarse, args.fine]]
    coarse, fine = [case[0] for case in cases]
    for key in ['case', 'model_sha256', 'weights_sha256', 'basis_sha256',
                'plan_sha256', 'source_sha256', 'runner_sha256',
                'candidate_mode_limit', 'symmetry', 'reference_arrays_loaded_before_prediction']:
        if coarse[key] != fine[key]:
            raise ValueError('Time refinement changed candidate identity: ' + key)
    if coarse['reference_arrays_loaded_before_prediction'] is not False:
        raise ValueError('Candidate-only predictions are required')
    if not 0 < fine['step_us'] < coarse['step_us']:
        raise ValueError('A strictly finer positive time step is required')
    ratio = coarse['step_us'] / fine['step_us']
    if ratio not in [2, 4, 8]:
        raise ValueError('Expected one or more exact step halvings')
    scene = coarse['case']['scene']
    parts = [comparison.comparison.parts(dict(enumerate(case[3]))) for case in cases]
    changes = {name: comparison.check(parts[0][name], parts[1][name], scene)
               for name in parts[0]}
    maximum = max(value['maximum_relative_l2'] for value in changes.values()
                  if value['maximum_relative_l2'] is not None)
    passed = all(value['passed'] for value in changes.values())
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        'schema': 'workai.fixed_candidate_time_refinement.v1',
        'duration_us': scene['acquisition']['duration_us'],
        'step_sizes_us': [coarse['step_us'], fine['step_us']],
        'step_ratio': ratio, 'coarse_identity': coarse, 'fine_identity': fine,
        'changes': changes, 'maximum_relative_change': maximum,
        'passed': passed, 'relative_limit': .05,
        'small_signal_absolute_limit_m': 1e-13,
        'observed_convergence_order_established': False,
        'scientific_accuracy_certified': False,
        'full_5ms_nonlinear_accuracy_verified': False,
        'analysis_source_sha256': {path.name: digest(path) for path in [
            Path(__file__), Path(loader.__file__), Path(comparison.__file__),
            Path(comparison.comparison.__file__)]},
        'files_sha256': {label + '/' + name: digest(folder / name)
            for label, folder in [('coarse', args.coarse), ('fine', args.fine)]
            for name in ['identity.json', 'prediction.json', 'prediction.npz', 'metrics.json']},
    }
    save(args.output / 'summary.json', report)
    print(json.dumps({key: report[key] for key in [
        'duration_us', 'step_sizes_us', 'maximum_relative_change', 'passed']}), flush=True)


if __name__ == '__main__':
    main()
