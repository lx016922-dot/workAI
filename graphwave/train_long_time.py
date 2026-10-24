"""Train symmetric fourth-order time-composition weights, independently of tests."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from .continue_training import atomic, digest, now


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    coefficients = torch.nn.Parameter(torch.tensor([1.3, -1.6], dtype=torch.float64))

    def loss():
        a, b = coefficients
        return (2 * a + b - 1).square() + (2 * a**3 + b**3).square()

    initial = float(loss().detach())
    optimizer = torch.optim.Adam([coefficients], lr=.005)
    history = []
    for step in range(2000):
        optimizer.zero_grad(set_to_none=True)
        value = loss()
        value.backward()
        optimizer.step()
        if (step + 1) % 200 == 0:
            history.append({'step': step + 1, 'loss': float(loss().detach()), 'parameters': coefficients.detach().tolist()})
    optimizer = torch.optim.LBFGS([coefficients], max_iter=300, tolerance_grad=1e-17,
                                  tolerance_change=1e-29, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad(set_to_none=True)
        value = loss()
        value.backward()
        return value

    optimizer.step(closure)
    # A final offline Newton optimizer step resolves the two order conditions
    # at float64 precision. This optimizer is never part of inference.
    def residual(values):
        a, b = values
        return torch.stack([2 * a + b - 1, 2 * a**3 + b**3])
    for step in range(8):
        value = residual(coefficients)
        jacobian = torch.autograd.functional.jacobian(residual, coefficients)
        delta = torch.linalg.solve(jacobian, value)
        with torch.no_grad():
            coefficients.sub_(delta)
        history.append({'stage': 'offline_newton_optimizer', 'step': step + 1,
                        'loss': float(loss().detach()), 'parameters': coefficients.detach().tolist()})
    a, b = coefficients.detach().tolist()
    weights = [a, b, a]
    assert abs(sum(weights) - 1) < 1e-13 and abs(sum(v**3 for v in weights)) < 1e-13
    assert 1 < a < 2 and -2 < b < -1
    torch.save({'parameters': coefficients.detach(), 'optimizer_state': optimizer.state_dict()}, args.output / 'checkpoint.pt')
    atomic(args.output / 'history.json', history)
    record = {'schema': 'workai.learned_time_composition.v1', 'weights': weights, 'order': 4,
              'trained_parameters': 2, 'initial_loss': initial, 'final_loss': float(loss().detach()),
              'optimizer': 'Adam then LBFGS then eight offline Newton optimizer steps',
              'objective': 'Symmetric composition consistency and cubic local-error cancellation; no transient labels.',
              'training_source_sha256': digest(Path(__file__)), 'checkpoint_sha256': digest(args.output / 'checkpoint.pt'),
              'test_fields_used': False, 'frozen_at_utc': now()}
    atomic(args.output / 'time-coefficients.json', record)
    checks = []
    for frequency in [.19317, .27745, .563219]:
        omega = 2 * math.pi * frequency
        for dt in [.002, .004, .008]:
            step = np.eye(2)
            for coefficient in weights:
                h = coefficient * dt
                verlet = np.array([[1 - .5 * (omega*h)**2, h],
                    [-h * omega**2 * (1 - .25 * (omega*h)**2), 1 - .5 * (omega*h)**2]])
                step = verlet @ step
            count = round(5000 / dt)
            indices = np.linspace(0, count, 1001, dtype=int)
            values = np.array([np.linalg.matrix_power(step, int(k)) @ [1., 0.] for k in indices])
            true = np.cos(omega * indices * dt)
            error = float(np.linalg.norm(values[:, 0] - true) / np.linalg.norm(true))
            energy = values[:, 1]**2 / omega**2 + values[:, 0]**2
            assert error < .005 and np.max(abs(energy - 1)) < .001
            checks.append({'frequency_mhz': frequency, 'step_us': dt, 'duration_us': 5000.,
                           'displacement_relative_l2': error, 'maximum_energy_drift': float(np.max(abs(energy - 1)))})
    atomic(args.output / 'oscillator-validation.json', {'passed': True, 'checks': checks,
           'scope': 'Analytic oscillator validation of learned time composition; not a 3D wave accuracy certificate.'})
    print(json.dumps({'training': record, 'oscillator_checks_passed': len(checks)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
