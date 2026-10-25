"""Compare an unchanged modal trajectory with later native COMSOL refinements."""
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
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--references', type=Path, nargs='+', required=True,
                        help='References ordered from the coarser mesh through the finer time step')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    admission = module('admission', 'check-modal-refinements.py')
    checks = module('checks', 'validate-comsol-linear-matrix.py')
    identity, prediction, _, traces = admission.load_case(args.prediction)
    if identity['reference_arrays_loaded_before_prediction']:
        raise ValueError('Reference-independent saved prediction required')
    scene = identity['case']['scene']
    actual = checks.comparison.parts(dict(enumerate(traces)))
    comparisons, reference_parts = [], []
    for folder in args.references:
        collection = json.loads((folder / 'collection.json').read_text())
        if digest(folder / 'status.json') != collection['signed_status_sha256']:
            raise ValueError('Signed collection status changed')
        status = json.loads((folder / 'status.json').read_text())
        if not status.get('server_signature_verified') or status['body']['job_id'] != collection['job_id']:
            raise ValueError('Verified native collection required')
        raw, receipt, values = checks.comparison.load_case(folder)
        if raw['scene'] != scene:
            raise ValueError('Physical scene or output grid differs')
        expected = checks.comparison.parts(values)
        reference_parts.append(expected)
        comparisons.append({'reference': folder.name, 'job_id': receipt['job_id'],
                            'identity': collection['identity'],
                            'collection_sha256': digest(folder / 'collection.json'),
                            'native_comparison': {key: checks.check(actual[key], value, scene)
                                                  for key, value in expected.items()}})
    refinements = []
    for i, (coarse, fine) in enumerate(zip(reference_parts[:-1], reference_parts[1:])):
        refinements.append({'from': args.references[i].name, 'to': args.references[i + 1].name,
                            'comparison': {key: checks.check(coarse[key], fine[key], scene) for key in coarse}})
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'workai.frozen_modal_native_recomparison.v1',
              'duration_us': scene['acquisition']['duration_us'],
              'unchanged_prediction_identity': identity, 'unchanged_prediction': prediction,
              'reference_arrays_loaded_before_prediction': False,
              'prediction_identity_sha256': digest(args.prediction / 'identity.json'),
              'candidate_recomputed_or_corrected': False,
              'native_comparisons': comparisons, 'reference_refinements': refinements,
              'full_5ms_nonlinear_accuracy_verified': False, 'scientific_accuracy_certified': False}
    save(args.output / 'summary.json', report)
    print(json.dumps({'duration_us': report['duration_us'], 'references': len(comparisons),
                      'maximum_complete_errors': {row['reference']: max(value['maximum_relative_l2']
                          for value in row['native_comparison'].values()) for row in comparisons},
                      'scientific_accuracy_certified': False}), flush=True)


if __name__ == '__main__':
    main()
