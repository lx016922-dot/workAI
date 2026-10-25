"""Add a new analytic degree filter as an explicit, untrained research baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectral3d.mesh import gll
from graphwave.modal_calibration import sbp_checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--degree', type=int, choices=[3, 4, 5, 6], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.weights, allow_pickle=False) as saved:
        weights = dict(saved)
    name = 'derivative_' + str(args.degree)
    if name in weights:
        raise ValueError('The selected filter already exists; do not overwrite learned parameters')
    nodes, quadrature, derivative, _ = gll(args.degree)
    check = sbp_checks(derivative, quadrature)
    if not check['passed']:
        raise ValueError('Analytic initialization must satisfy discrete integration by parts')
    if args.output.exists():
        raise ValueError('Use a new baseline output directory')
    args.output.mkdir(parents=True)
    target = args.output / 'candidate-weights.npz'
    np.savez(target, **weights, **{name: derivative})
    with np.load(target, allow_pickle=False) as saved:
        for key, value in weights.items():
            np.testing.assert_array_equal(saved[key], value)
        np.testing.assert_array_equal(saved[name], derivative)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        'schema': 'workai.new_degree_initialization.v1',
        'parent_weights_sha256': sha(args.weights),
        'weights_sha256': sha(target),
        'new_filter': name, 'degree': args.degree,
        'initialization': 'Analytic GLL interpolation derivative; not yet trained against COMSOL',
        'all_parent_arrays_preserved_exactly': True,
        'preserved_parent_arrays': sorted(weights),
        'polynomial_consistency_max_absolute': float(max(
            np.max(abs(derivative @ nodes**k - (k*nodes**(k-1) if k else nodes*0)))
            for k in range(args.degree + 1))),
        'summation_by_parts': check,
        'initialization_source_sha256': sha(Path(__file__)),
        'accepted_for_offline_release': False,
        'scientific_accuracy_certified': False,
    }
    (args.output / 'initialization.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
