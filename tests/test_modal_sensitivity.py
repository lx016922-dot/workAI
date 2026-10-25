import json
from pathlib import Path
import unittest

import numpy as np

from graphwave.graded_mesh import make_graded_mesh
from graphwave.modal_calibration import filter_basis, modes, stiffness
from graphwave.modal_sensitivity import frequency_jacobian, mandel_strain, modal_gradients
from spectral3d.mesh import boundary_data


class ModalSensitivityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.scene = json.loads((root/'research/comsol-validation/data/eigen-030-v7/scene.json').read_text())['scene']
        cls.mesh = make_graded_mesh(cls.scene, 5, 2.5, (8., 3., 3.), counts=(2, 1, 1))
        with np.load(root/'offline/model/continued-weights.npz', allow_pickle=False) as saved:
            cls.quadratic = saved['quadratic_tensor']
        cls.basis = filter_basis(cls.mesh['gll'], 2, cls.mesh['weights_1d'])

    def test_gradient_energy_matches_actual_assembled_graph_tangent(self):
        mesh = self.mesh
        fields = np.random.default_rng(2026102524).normal(size=(3, len(mesh['points']), 3))*1e-9
        _, fixed, _ = boundary_data(mesh, self.scene)
        fields[:, fixed] = 0.
        strain = mandel_strain(modal_gradients(mesh, mesh['derivative'], fields))
        actual = np.einsum('meijkr,rs,meijks,eijk->m', strain, 2e11*self.quadratic,
                           strain, mesh['volume_weight'], optimize=True)
        matrix = stiffness(mesh, mesh['derivative'], self.quadratic)
        expected = np.array([x.ravel() @ (matrix @ x.ravel()) for x in fields])
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-22)

    def test_eigenfrequency_jacobian_matches_independent_eigensolves(self):
        mesh = self.mesh
        derivative = mesh['derivative'] + np.einsum('a,aij->ij', [.003, -.002], self.basis)
        frequency, fields = modes(mesh, self.scene, derivative, self.quadratic, 8)
        actual = frequency_jacobian(mesh, derivative, self.quadratic, frequency, fields, self.basis)
        step = 2e-4
        numeric = []
        for direction in self.basis:
            plus, _ = modes(mesh, self.scene, derivative+step*direction, self.quadratic, 8)
            minus, _ = modes(mesh, self.scene, derivative-step*direction, self.quadratic, 8)
            numeric.append((plus-minus)/(2*step))
        np.testing.assert_allclose(actual, np.stack(numeric, axis=-1), rtol=3e-4, atol=.003)
        # The derivative must be invariant to arbitrary eigenvector scaling.
        scaled = fields*np.arange(1, 9)[:, None, None]
        np.testing.assert_allclose(frequency_jacobian(mesh, derivative, self.quadratic, frequency, scaled, self.basis),
                                   actual, rtol=1e-11, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
