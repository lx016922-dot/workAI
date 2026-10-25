import json
from pathlib import Path
import unittest

import numpy as np

from graphwave.graded_mesh import make_graded_mesh
from spectral3d.mesh import boundary_data, interpolation, make_mesh


class GradedPlateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.scene = json.loads((root/'research/comsol-validation/data/eigen-030-v7/scene.json').read_text())['scene']
        with np.load(root/'offline/model/continued-weights.npz', allow_pickle=False) as saved:
            cls.weights = dict(saved)

    def test_default_reproduces_existing_mesh_exactly(self):
        old = make_mesh(self.scene, 4, 1.6)
        new = make_graded_mesh(self.scene, 4, 1.6)
        self.assertEqual(set(old), set(new))
        for key in old:
            np.testing.assert_array_equal(old[key], new[key])

    def test_geometry_mass_and_physical_load_survive_grading(self):
        mesh = make_graded_mesh(self.scene, 3, 1.6, (20., 20., 20.), counts=(4, 3, 3))
        self.assertEqual(mesh['axis_element_counts'], [4, 3, 3])
        x, y, z = [np.diff(edge) for edge in mesh['axis_edges_m']]
        self.assertGreater(x[0], x[-1])
        np.testing.assert_allclose(y, y[::-1], rtol=1e-13)
        np.testing.assert_allclose(z, z[::-1], rtol=1e-13)
        self.assertTrue(np.all(mesh['mass'] > 0))
        self.assertAlmostEqual(mesh['volume_weight'].sum() / (8*3*1.5e-9), 1., places=13)
        self.assertAlmostEqual(mesh['mass'].sum() / (2700*8*3*1.5e-9), 1., places=13)
        load, fixed, area = boundary_data(mesh, self.scene)
        np.testing.assert_allclose(load.sum(axis=0), [4.5, 0., 0.], rtol=1e-13, atol=1e-15)
        self.assertAlmostEqual(area / 4.5e-6, 1., places=13)
        np.testing.assert_array_equal(load[fixed], 0.)

    def test_polynomial_interpolation_on_graded_cells(self):
        mesh = make_graded_mesh(self.scene, 3, 1.6, (10., 5., 5.), counts=(4, 3, 3))
        def field(x):
            return np.stack([x[..., 0]**3, x[..., 0]*x[..., 1], x[..., 2]**2], axis=-1)
        values = field(mesh['points'])
        for point in np.random.default_rng(2026102521).uniform([0, 0, 0], [8, 3, 1.5], size=(30, 3)):
            ids, basis = interpolation(mesh, point)
            np.testing.assert_allclose(basis @ values[ids], field(point*1e-3), rtol=3e-13, atol=2e-20)

    def test_actual_graph_energy_and_modal_tangent_on_graded_geometry(self):
        from graphwave.buffered_inference import energy_activation, graph_message
        from graphwave.inference import inverse_mapping
        from graphwave.modal_calibration import stiffness
        mesh = make_graded_mesh(self.scene, 3, 1.6, (10., 5., 5.), counts=(3, 3, 3))
        gradient = np.array([[2., -1., .3], [.2, -3., .5], [.1, -.4, 1.]])*1e-5
        field = mesh['points'] @ gradient.T
        quadratic, cubic = self.weights['quadratic_tensor'], self.weights['cubic_tensor']
        for law in [0, 1, 2]:
            _, energy, _, _ = graph_message(field, mesh['elements'], mesh['derivative'], inverse_mapping(mesh),
                                            mesh['volume_weight'], quadratic, cubic, law)
            exact = energy_activation(gradient, quadratic, cubic, law)[1] * mesh['volume_weight'].sum()
            self.assertAlmostEqual(energy/exact, 1., places=12)
        random = np.random.default_rng(2026102521).normal(size=field.shape)*1e-11
        actual = graph_message(random, mesh['elements'], mesh['derivative'], inverse_mapping(mesh),
                               mesh['volume_weight'], quadratic, cubic, 0)[0]
        matrix = stiffness(mesh, mesh['derivative'], quadratic)
        self.assertLess(np.linalg.norm(matrix@random.ravel()-actual.ravel())/np.linalg.norm(actual), 1e-12)


if __name__ == '__main__':
    unittest.main()
