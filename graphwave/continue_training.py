"""Extend a frozen GraphWave checkpoint with a learned degree-five filter.

The 120 parent parameters remain bit-identical. Only the new 36-parameter
filter is optimized, using polynomial field/derivative pairs generated offline.
No transient test field is loaded or used for checkpoint selection.
"""
import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial.legendre import Legendre

from .model import GraphWave


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    parent = torch.load(args.parent / 'best.pt', weights_only=True, map_location='cpu')
    parent_state = parent['model_state']
    model = GraphWave(degrees=(3, 4, 5))
    result = model.load_state_dict(parent_state, strict=False)
    assert result.missing_keys == ['filters.5'] and not result.unexpected_keys
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == 'filters.5')
    rng = np.random.default_rng(args.seed)
    nodes = np.r_[-1., Legendre.basis(5).deriv().roots(), 1.]
    basis = np.stack([Legendre.basis(k)(nodes) for k in range(6)], axis=1)
    derivative = np.stack([Legendre.basis(k).deriv()(nodes) for k in range(6)], axis=1)

    def sample(count):
        coefficients = rng.normal(size=(6, count)) / np.arange(1, 7)[:, None]
        return torch.tensor(basis @ coefficients), torch.tensor(derivative @ coefficients)

    train_x, train_y = sample(4096)
    valid_x, valid_y = sample(2048)
    np.savez_compressed(args.output / 'training-patches.npz', train_x=train_x.numpy(), train_y=train_y.numpy(),
                        validation_x=valid_x.numpy(), validation_y=valid_y.numpy())
    repo = Path(__file__).resolve().parents[1]
    sources = {}
    for module in ['graphwave', 'spectral3d', 'solid', 'operator3d']:
        directory = args.output / 'source_snapshot' / module
        directory.mkdir(parents=True)
        for path in (repo / module).glob('*.py'):
            shutil.copyfile(path, directory / path.name)
            sources[module + '/' + path.name] = digest(path)
    config = {'schema': 'workai.graphwave_continuation.v1', 'seed': args.seed,
              'parent_checkpoint_sha256': digest(args.parent / 'best.pt'),
              'parent_weights_sha256': digest(args.parent / 'weights.npz'),
              'total_parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'degrees': [3, 4, 5], 'frozen_parent_parameters': 120,
              'training_examples': 4096, 'validation_examples': 2048,
              'training_data_sha256': digest(args.output / 'training-patches.npz'),
              'training_data': 'Random Legendre field patches and their analytic derivatives; no transient responses.',
              'parent_retrained': False, 'test_fields_used': False,
              'source_sha256': sources, 'created_at_utc': now()}
    atomic(args.output / 'config.json', config)

    def loss(x=train_x, y=train_y):
        return ((model.derivative(5) @ x - y) ** 2).mean() / y.square().mean()

    initial = float(loss().detach())
    parameters = [model.filters['5']]
    optimizer = torch.optim.Adam(parameters, lr=.03)
    history = []
    for step in range(1, 2401):
        optimizer.zero_grad(set_to_none=True)
        value = loss()
        value.backward()
        optimizer.step()
        if step == 1 or step % 200 == 0:
            row = {'stage': 'adam', 'step': step, 'training_loss': float(value.detach()),
                   'validation_loss': float(loss(valid_x, valid_y).detach()), 'elapsed_s': time.monotonic() - started}
            history.append(row)
            print(json.dumps(row), flush=True)
    optimizer = torch.optim.LBFGS(parameters, lr=1., max_iter=600, tolerance_grad=1e-17,
                                 tolerance_change=1e-28, history_size=100, line_search_fn='strong_wolfe')
    calls = 0

    def closure():
        nonlocal calls
        optimizer.zero_grad(set_to_none=True)
        value = loss()
        value.backward()
        calls += 1
        return value

    optimizer.step(closure)
    final = float(loss(valid_x, valid_y).detach())
    for key, value in parent_state.items():
        assert torch.equal(model.state_dict()[key], value), key
    assert final < 1e-20, 'New filter did not converge; candidate is not frozen.'
    checkpoint = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                  'optimizer': 'LBFGS_after_Adam', 'adam_steps': 2400, 'lbfgs_function_evaluations': calls,
                  'seed': args.seed, 'degrees': [3, 4, 5], 'parent_checkpoint_sha256': config['parent_checkpoint_sha256']}
    torch.save(checkpoint, args.output / 'best.pt')
    exported = model.export()
    weights = {'quadratic_tensor': exported['quadratic_tensor'], 'cubic_tensor': exported['cubic_tensor'],
               'state_weights': exported['state_weights'],
               **{'derivative_' + p: v for p, v in exported['derivatives'].items()}}
    np.savez(args.output / 'weights.npz', **weights)
    with np.load(args.parent / 'weights.npz', allow_pickle=False) as original:
        for key in original.files:
            np.testing.assert_array_equal(weights[key], original[key])
    history.append({'stage': 'lbfgs', 'function_evaluations': calls, 'training_loss': float(loss().detach()),
                    'validation_loss': final, 'elapsed_s': time.monotonic() - started})
    atomic(args.output / 'history.json', history)
    audit = {'schema': 'workai.graphwave_continuation_training.v1', 'initial_training_loss': initial,
             'final_validation_loss': final, 'adam_steps': 2400, 'lbfgs_function_evaluations': calls,
             'elapsed_s': time.monotonic() - started, 'parent_parameters_bit_identical': True,
             'parent_export_arrays_bit_identical': True, 'test_fields_used': False,
             'weights_sha256': digest(args.output / 'weights.npz'), 'checkpoint_sha256': digest(args.output / 'best.pt'),
             'frozen_at_utc': now(), 'transient_accuracy_validated': False, 'accepted_for_offline_release': False}
    atomic(args.output / 'training.json', audit)
    atomic(args.output / 'checkpoint-freeze.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026102407)
    run(parser.parse_args())
