import importlib.util
from pathlib import Path
import unittest

import numpy as np

from tools.comsol_nodal_reference import cell_fields, nonlinear_force


class NativeFieldReconstruction(unittest.TestCase):
    def test_arbitrary_q4_polynomial_and_physical_derivatives(self):
        local = np.array([-1, -np.sqrt(3 / 7), 0, np.sqrt(3 / 7), 1])
        size = np.array([.0031, .00027, .00063])
        origin = np.array([.002, .00011, .0003])
        xyz = np.meshgrid(*[origin[i] + .5 * size[i] * (local + 1) for i in range(3)], indexing='ij')
        x, y, z = xyz
        fields = np.stack([x**4 * y**3, x * z**4, y**2 * z**3], axis=-1)[None]
        points = np.array([-.91, -.27, .38, .87])
        values, gradient = cell_fields(fields, size, points)
        x, y, z = np.meshgrid(*[origin[i] + .5 * size[i] * (points + 1) for i in range(3)], indexing='ij')
        expected = np.stack([x**4 * y**3, x * z**4, y**2 * z**3], axis=-1)
        derivatives = np.zeros(expected.shape + (3,))
        derivatives[..., 0, 0] = 4 * x**3 * y**3
        derivatives[..., 0, 1] = 3 * x**4 * y**2
        derivatives[..., 1, 0] = z**4
        derivatives[..., 1, 2] = 4 * x * z**3
        derivatives[..., 2, 1] = 2 * y * z**3
        derivatives[..., 2, 2] = 3 * y**2 * z**2
        np.testing.assert_allclose(values[0], expected, rtol=5e-13, atol=1e-30)
        # Derivatives that are analytically zero still incur cancellation at
        # the scale of field magnitude divided by the smallest cell size.
        roundoff = 100 * np.finfo(float).eps * np.max(abs(fields)) / size.min()
        np.testing.assert_allclose(gradient[0], derivatives, rtol=5e-12, atol=roundoff)

    def test_material_force_is_the_derivative_of_its_own_energy(self):
        random = np.random.default_rng(2026102553)
        gradient = random.normal(size=(8, 17 * 9)) * 1e3
        model = {'omega': np.arange(1, 9), 'gradient': gradient,
                 'volume': random.uniform(1e-10, 2e-10, 17),
                 'material_constants': np.array([58.1e9, 26.1e9, -143e9, -297e9, -345e9])}
        state = random.normal(size=8) * 1e-8
        direction = random.normal(size=8); direction /= np.linalg.norm(direction)
        for law in [0, 1, 2]:
            force, _, _, _ = nonlinear_force(model, state, [law])
            delta = 2e-11
            upper = nonlinear_force(model, state + delta * direction, [law])[1][0]
            lower = nonlinear_force(model, state - delta * direction, [law])[1][0]
            actual = (upper - lower) / (2 * delta)
            expected = float(force[0] @ direction)
            self.assertLess(abs(actual - expected), max(1e-10, abs(expected) * 2e-5))

    def test_sixth_order_nonlinear_flow_against_adaptive_duffing_reference(self):
        from scipy.integrate import solve_ivp
        from graphwave.signals import pulse_values
        spec = importlib.util.spec_from_file_location('reference_flow', Path(__file__).resolve().parents[1] / 'tools/run-comsol-nodal-transient.py')
        flow = importlib.util.module_from_spec(spec); spec.loader.exec_module(flow)
        scene = {'source': {'frequency_mhz': .22, 'cycles': 1, 'delay_us': .2,
                            'phase_deg': 0., 'amplitude_mpa': 1.},
                 'signal': {'kind': 'hann'}, 'receivers': [{}],
                 'acquisition': {'duration_us': 8., 'sample_step_us': .2}}
        frequency = np.array([.16, .47]) * 1e6; omega = 2 * np.pi * frequency
        scale = 1e-9
        model = {'frequency_hz': frequency, 'omega': omega, 'load': np.array([.001, -.0007]),
                 'receiver_fields': np.array([[[1., 0., 0.]], [[0., 1., 0.]]])}
        def force(model, position, laws):
            coefficient = np.array(laws)[:, None] * .002
            residual = coefficient * omega**2 * position**3 / scale**2
            energy = np.sum(.25 * coefficient * omega**2 * position**4 / scale**2, axis=1)
            return residual, energy, np.zeros(len(laws)), np.ones(len(laws))
        def right_hand_side(time, value):
            state = value.reshape(3, 2, 2); position, velocity = state[:, 0], state[:, 1]
            acceleration = -omega * (position + np.arange(3)[:, None] * .002 * position**3)
            acceleration += model['load'] * (1e6 * pulse_values(scene, time * 1e6)) / (scale * omega)
            return np.stack([omega * velocity, acceleration], axis=1).ravel()
        times = np.arange(41) * .2e-6
        reference = solve_ivp(right_hand_side, [0., 8e-6], np.zeros(12), t_eval=times,
                              method='DOP853', rtol=2e-12, atol=1e-13, max_step=.025e-6)
        self.assertTrue(reference.success)
        expected = reference.y.reshape(3, 2, 2, -1)[:, 0].transpose(0, 2, 1) * scale
        errors = []
        for step in [.2, .1]:
            actual = flow.evolve(model, scene, step, force=force)['coordinates']
            errors.append(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
        self.assertLess(errors[1], 1e-6)
        self.assertGreater(errors[0] / max(errors[1], 1e-15), 15.)


if __name__ == '__main__':
    unittest.main()
