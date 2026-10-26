"""Nodal contraction must preserve the complete learned nonlinear operator."""
import json
from pathlib import Path
import unittest

import numpy as np
from graphwave.modal_reduction import attach_basis, nonlinear_force as dense_force
from graphwave.modal_nodal_force import prepare, nonlinear_force
from graphwave.quarter_symmetry import make_quarter_mesh, free_dofs


class ExactNodalContraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        scene = json.loads((root/'research/comsol-validation/plans/v11-quarter-native-driven.json').read_text())['cases'][0]['scene']
        with np.load(root/'offline/model/continued-weights.npz') as archive:
            weights = dict(archive)
        mesh = make_quarter_mesh(scene,4,1.,[1,1,1],[2,1,1])
        free = free_dofs(mesh,scene)
        rng = np.random.default_rng(514)
        vectors = np.linalg.qr(rng.normal(size=(free.sum(),8)),mode='reduced')[0]
        fields = np.zeros((8,len(free)))
        fields[:,free] = (vectors/np.sqrt(np.repeat(mesh['mass'],3)[free])[:,None]).T
        cls.dense = attach_basis(mesh,scene,weights,np.arange(1,9)*1e5,fields.reshape(8,-1,3))
        cls.nodal = prepare(cls.dense,mesh,weights)

    def test_full_force_and_energy_match_for_both_nonlinear_parts(self):
        rng = np.random.default_rng(515)
        for scale in [2e-5,1e-4,8e-4]:
            q = rng.normal(size=8)
            q *= scale/np.linalg.norm((q@self.dense['gradient']).reshape(-1,3,3),axis=(1,2)).max()
            states = np.stack([q]*3); laws = np.arange(3)
            expected = dense_force(self.dense,states,laws)
            actual = nonlinear_force(self.nodal,states,laws)
            for a,b in zip(actual,expected):
                np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-24)
            for a,b in [(actual[0][2]-actual[0][1],expected[0][2]-expected[0][1]),
                        (actual[1][2]-actual[1][1],expected[1][2]-expected[1][1])]:
                np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-24)

    def test_zero_state_and_invalid_constitutive_branch(self):
        force,energy,_,jac = nonlinear_force(self.nodal,np.zeros((3,8)),np.arange(3))
        np.testing.assert_array_equal(force,0);np.testing.assert_array_equal(energy,0)
        np.testing.assert_array_equal(jac,1)
        with self.assertRaises(ValueError):
            nonlinear_force(self.nodal,np.zeros((1,8)),[3])

    def test_conservative_trajectory_replays_all_three_controls(self):
        from graphwave.modal_reduction import compile_basis,propagate
        from graphwave.modal_nodal_runtime import propagate_nodal
        root = Path(__file__).resolve().parents[1]
        scene = json.loads((root/'research/comsol-validation/plans/v11-quarter-native-driven.json').read_text())['cases'][0]['scene']
        with np.load(root/'offline/model/continued-weights.npz') as archive:
            weights = dict(archive)
        mesh = make_quarter_mesh(scene,4,1.,[1,1,1],[2,1,1])
        dense = compile_basis(mesh,scene,weights,8)
        expected = propagate(dense,scene,step_us=.025)
        actual = propagate_nodal(prepare(dense,mesh,weights),scene,step_us=.025)
        np.testing.assert_array_equal(actual['times_us'],expected['times_us'])
        np.testing.assert_allclose(actual['traces'],expected['traces'],rtol=1e-8,atol=1e-22)
        np.testing.assert_allclose(actual['energy'],expected['energy'],rtol=1e-8,atol=1e-24)


if __name__ == '__main__':
    unittest.main()
