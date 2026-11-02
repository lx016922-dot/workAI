import json
from pathlib import Path
import unittest
import numpy as np

from graphwave.graded_mesh import make_graded_mesh
from graphwave.quarter_symmetry import make_quarter_mesh,modes as quarter_modes,sampling,material_symmetry
from graphwave.modal_sparse import modes as full_modes
from graphwave.modal_calibration import match
from graphwave.modal_reduction import axial_reflection_subspace
from graphwave.buffered_inference import graph_message
from graphwave.inference import inverse_mapping
from spectral3d.mesh import boundary_data


class QuarterSymmetryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        cls.scene=json.loads((root/'research/comsol-validation/data/eigen-030-v7/scene.json').read_text())['scene']
        with np.load(root/'offline/model/continued-weights.npz',allow_pickle=False) as f:cls.weights=dict(f)
        cls.full=make_graded_mesh(cls.scene,3,2.,(4.,5.,6.),(3,4,4))
        cls.quarter=make_quarter_mesh(cls.scene,3,2.,(4.,5.,6.),(3,2,2))
        cls.qfrequency,cls.qfields=quarter_modes(cls.quarter,cls.scene,cls.quarter['derivative'],cls.weights['quadratic_tensor'],12)

    def test_mirrored_mass_load_and_all_nonlinear_energies(self):
        full,quarter=self.full,self.quarter
        self.assertAlmostEqual(full['mass'].sum()/quarter['mass'].sum(),4.,places=13)
        a=boundary_data(full,self.scene)[0].sum(axis=0);b=boundary_data(quarter,self.scene)[0].sum(axis=0)
        np.testing.assert_allclose(a,4*b,rtol=1e-13,atol=1e-14)
        rng=np.random.default_rng(2026102535);coefficients=rng.normal(size=12)*1e-10
        q=np.einsum('m,mnc->nc',coefficients,self.qfields)
        f=sampling(quarter,full['points'])(q[None])[0]
        for law in [0,1,2]:
            energies=[]
            for mesh,field in [(full,f),(quarter,q)]:
                energies.append(graph_message(field,mesh['elements'],mesh['derivative'],inverse_mapping(mesh),
                    mesh['volume_weight'],self.weights['quadratic_tensor'],self.weights['cubic_tensor'],law)[1])
            self.assertAlmostEqual(energies[0]/energies[1],4.,places=11)

    def test_quarter_spectrum_matches_full_graph_reflection_class(self):
        f,fields=full_modes(self.full,self.scene,self.full['derivative'],self.weights['quadratic_tensor'],52)
        selected,_=axial_reflection_subspace(self.full,self.scene,self.weights,fields)
        self.assertGreaterEqual(len(selected),10)
        sampled=sampling(self.quarter,self.full['points'])(self.qfields)
        mapping,mac=match(fields[selected[:10]],sampled,self.full['mass'])
        np.testing.assert_allclose(f[selected[:10]],self.qfrequency[mapping],rtol=2e-10,atol=.0001)
        self.assertGreater(mac.min(),1-1e-10)

    def test_asymmetric_material_and_invalid_grading_are_rejected(self):
        q=self.weights['quadratic_tensor'].copy();t=self.weights['cubic_tensor'].copy()
        self.assertLess(material_symmetry(q,t),1e-12)
        q[0,3]+=.001;q[3,0]+=.001
        with self.assertRaisesRegex(ValueError,'symmetry'):material_symmetry(q,t)
        q=self.weights['quadratic_tensor'].copy();t[0,0,3]+=.001
        with self.assertRaisesRegex(ValueError,'symmetry'):material_symmetry(q,t)
        for ratios in [(4.,float('nan'),6.),(4.,5.,101.)]:
            with self.assertRaises(ValueError):make_quarter_mesh(self.scene,3,2.,ratios,(3,2,2))


if __name__=='__main__':unittest.main()
