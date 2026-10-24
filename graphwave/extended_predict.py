"""Research-only p3/p4/p5 graph recurrence for a continued frozen checkpoint.

The v1 inference module and its published model remain unchanged. This module
adds degree-five admission; it never imports or invokes a reference solver.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from spectral3d.mesh import make_mesh, boundary_data, interpolation
from .inference import inverse_mapping, source_values, recurrent_forward
from .predict import digest


def predict(scene, mesh, weights, cfl=.3, callback=None):
    degree = int(mesh['degree'])
    if degree not in (3, 4, 5) or 'derivative_' + str(degree) not in weights:
        raise ValueError('Continued checkpoint supports trained graph degrees 3, 4 and 5.')
    if not .05 <= cfl <= .3:
        raise ValueError('Forward layer spacing outside validated bounds.')
    sample = scene['acquisition']['sample_step_us'] * 1e-6
    subdivisions = int(np.ceil(sample / (cfl * float(mesh['min_node_spacing']) / 10000.)))
    dt = sample / subdivisions
    frames = round(scene['acquisition']['duration_us'] / scene['acquisition']['sample_step_us']) + 1
    steps = (frames - 1) * subdivisions
    if steps > 150000 or frames * len(mesh['mass']) * 3 * 8 > 512 * 1024**2:
        raise ValueError('Bounded neural forward budget exceeded.')
    load, fixed, _ = boundary_data(mesh, scene)
    inverse = inverse_mapping(mesh)
    pulse = source_values(scene['source'], np.arange(steps + 1) * dt * 1e6)
    fields, diagnostics = [], []
    for law in range(3):
        started = time.monotonic()
        field, energy, strain, jac = recurrent_forward(
            mesh['elements'], weights['derivative_' + str(degree)], inverse,
            mesh['volume_weight'], mesh['mass'], load, fixed,
            weights['quadratic_tensor'], weights['cubic_tensor'], weights['state_weights'],
            law, dt, subdivisions, frames, pulse)
        scale = max(np.max(abs(energy[:, 3])), np.max(abs(energy[:, 1] + energy[:, 2])), 1e-30)
        balance = float(np.max(abs(energy[:, 1] + energy[:, 2] - energy[:, 3])) / scale)
        fields.append(field)
        diagnostics.append({'law': ['linear', 'stvk', 'murnaghan'][law], 'degree': degree,
                            'elapsed_s': time.monotonic() - started, 'recurrent_layers': steps,
                            'layer_step_us': dt * 1e6, 'max_strain': float(strain), 'min_J': float(jac),
                            'external_work_balance_relative': balance,
                            'energy_balance_within_1_percent': balance <= .01,
                            'energy_samples': energy.tolist(), 'neural_forward_only': True,
                            'numerical_correction': False, 'reference_responses_read': False})
        if callback:
            callback(diagnostics[-1])
    return fields, diagnostics


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    request = json.loads(args.input.read_text(encoding='utf-8'))
    scene = request['scene']
    assert digest(args.weights) == request['weights_sha256']
    with np.load(args.weights, allow_pickle=False) as saved:
        weights = dict(saved)
    expected = {'quadratic_tensor': (6, 6), 'cubic_tensor': (6, 6, 6), 'state_weights': (2,),
                'derivative_3': (4, 4), 'derivative_4': (5, 5), 'derivative_5': (6, 6)}
    if set(weights) != set(expected) or any(weights[k].shape != shape or not np.isfinite(weights[k]).all()
                                           for k, shape in expected.items()):
        raise ValueError('Invalid continued network weights.')
    mesh = make_mesh(scene, request['degree'], request['h_mm'])
    fields, diagnostics = predict(scene, mesh, weights, request.get('cfl', .3),
        callback=lambda row: print(json.dumps({k: v for k, v in row.items() if k != 'energy_samples'}), flush=True))
    np.savez_compressed(args.output / 'mesh.npz', **mesh)
    np.savez_compressed(args.output / 'fields.npz', linear=fields[0], stvk=fields[1], murnaghan=fields[2],
                        times_us=np.arange(len(fields[0])) * scene['acquisition']['sample_step_us'])
    probes = []
    for receiver in scene['receivers']:
        ids, w = interpolation(mesh, receiver['point_mm'])
        probes.append(np.stack([np.einsum('n,tnc->tc', w, field[:, ids]) for field in fields], 1))
    np.savez_compressed(args.output / 'probes.npz', controls=np.stack(probes, 1))
    forbidden = ['spectral3d.solver', 'spectral3d.curved_solver', 'spectral3d.implicit', 'solid.fem']
    if any(name in sys.modules for name in forbidden):
        raise RuntimeError('A numerical solver entered the inference process.')
    report = {'schema': 'workai.graphwave_extended_prediction.v1', 'weights_sha256': digest(args.weights),
              'input_sha256': digest(args.input), 'field_sha256': digest(args.output / 'fields.npz'),
              'mesh_sha256': digest(args.output / 'mesh.npz'), 'probe_sha256': digest(args.output / 'probes.npz'),
              'diagnostics': diagnostics, 'neural_forward_only': True, 'numerical_correction': False,
              'reference_responses_read_for_inference': False, 'solver_modules_absent': forbidden}
    (args.output / 'prediction.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
