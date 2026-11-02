import copy
import json
from pathlib import Path
import unittest

import numpy as np

from graphwave.buffered_inference import graph_message
from graphwave.graded_mesh import make_graded_mesh
from graphwave.inference import inverse_mapping
from graphwave.modal_reduction import (compile_basis, attach_basis, axial_reflection_subspace,
                                      nonlinear_force, nonlinear_force_factored, oscillator_flow, propagate)


class ModalReductionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.scene = json.loads((root/'research/comsol-validation/data/eigen-030-v7/scene.json').read_text())['scene']
        cls.mesh = make_graded_mesh(cls.scene, 3, 2.5, (2., 1., 1.), counts=(3, 1, 1))
        with np.load(root/'offline/model/continued-weights.npz', allow_pickle=False) as saved:
            cls.weights = dict(saved)
        cls.weights['derivative_3'] = cls.mesh['derivative']
        cls.model = compile_basis(cls.mesh, cls.scene, cls.weights, 16)

    def test_projected_full_nonlinear_force_and_energy_match_graph(self):
        mesh, model = self.mesh, self.model
        q = np.random.default_rng(2026102527).normal(size=(3, 16))*1e-10
        force, energy, strain, jac = nonlinear_force(model, q, [0, 1, 2])
        for law in [0, 1, 2]:
            field = np.einsum('m,mnc->nc', q[law], model['fields'])
            args = (field, mesh['elements'], mesh['derivative'], inverse_mapping(mesh),
                    mesh['volume_weight'], self.weights['quadratic_tensor'], self.weights['cubic_tensor'])
            full = graph_message(*args, law)
            linear = graph_message(*args, 0)
            expected = np.einsum('mnc,nc->m', model['fields'], full[0]-linear[0])
            np.testing.assert_allclose(force[law], expected, rtol=2e-7, atol=1e-13)
            self.assertAlmostEqual(energy[law]/(full[1]-linear[1]), 1., places=7) if law else self.assertEqual(energy[law], 0.)
            self.assertAlmostEqual(strain[law], full[2], places=15)
            self.assertAlmostEqual(jac[law], full[3], places=15)

    def test_reduced_force_remains_the_derivative_of_energy(self):
        model = self.model
        q = np.random.default_rng(2026102528).normal(size=(2, 16))*1e-10
        direction = np.random.default_rng(2026102529).normal(size=q.shape)*1e-10
        force = nonlinear_force(model, q, [1, 2])[0]
        step = .001
        plus = nonlinear_force(model, q+step*direction, [1, 2])[1]
        minus = nonlinear_force(model, q-step*direction, [1, 2])[1]
        expected = np.sum(force*direction, axis=1)
        np.testing.assert_allclose((plus-minus)/(2*step), expected, rtol=3e-6, atol=1e-23)

    def test_factored_full_tensor_increment_matches_pointwise_model(self):
        q=np.random.default_rng(2026102531).normal(size=(3,16))*1e-10
        pointwise=nonlinear_force(self.model,q,[0,1,2])
        factored=nonlinear_force_factored(self.model,q,[0,1,2])
        for a,b in zip(pointwise,factored):
            np.testing.assert_allclose(a,b,rtol=2e-7,atol=1e-22)

    def test_reflection_selection_preserves_transverse_displacement(self):
        indices,report=axial_reflection_subspace(self.mesh,self.scene,self.weights,self.model['fields'])
        self.assertLess(len(indices),len(self.model['fields']))
        self.assertLess(report['relative_excluded_load'],1e-8)
        self.assertGreater(np.linalg.norm(self.model['fields'][indices,:,1:]),0.)
        scene=copy.deepcopy(self.scene);scene['source']['direction']=[1.,.1,0.]
        with self.assertRaises(ValueError):axial_reflection_subspace(self.mesh,scene,self.weights,self.model['fields'])

    def test_exact_linear_flow_preserves_hamiltonian_and_reverses(self):
        model = self.model
        rng = np.random.default_rng(2026102530)
        q, v = rng.normal(size=(2, 16))*1e-10, rng.normal(size=(2, 16))*1e-4
        a, b = oscillator_flow(q, v, model['omega'], .0037)
        np.testing.assert_allclose(np.sum(a*a*model['omega']**2+b*b, axis=1),
                                   np.sum(q*q*model['omega']**2+v*v, axis=1), rtol=3e-15)
        a, b = oscillator_flow(a, b, model['omega'], -.0037)
        np.testing.assert_allclose(a, q, rtol=2e-14, atol=1e-24)
        np.testing.assert_allclose(b, v, rtol=2e-14, atol=1e-18)

    def test_three_controls_converge_with_time_step(self):
        scene = copy.deepcopy(self.scene)
        scene['acquisition'] = {'duration_us':20., 'sample_step_us':.2}
        scene['source']['cycles'] = 1
        values = [propagate(self.model, scene, step_us=step) for step in [.1, .05, .025]]
        self.assertEqual(values[0]['traces'].shape, (3, 101, 8, 3))
        self.assertEqual(values[0]['times_us'][-1], 20.)
        # Check both physical increments as well as total displacement; a
        # large linear field must not hide a bad nonlinear temporal layer.
        def branches(x):return np.stack([x[0],x[1]-x[0],x[2]-x[1],x[2]])
        a, b, c = [branches(row['traces']) for row in values]
        errors = np.linalg.norm((a-b).reshape(4, -1), axis=1)
        refined = np.linalg.norm((b-c).reshape(4, -1), axis=1)
        self.assertLess(refined[0]/np.linalg.norm(c[0]),1e-10)
        self.assertTrue(np.all(refined[1:] < errors[1:]*.3), (errors, refined))
        self.assertTrue(np.all(refined/np.linalg.norm(c.reshape(4, -1), axis=1) < .002))
        for row in values:self.assertLess(row['energy_relative_error'].max(), .01)


if __name__ == '__main__':
    unittest.main()
