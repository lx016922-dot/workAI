"""Physical invariants of the learned network, independent of transient labels."""
import os,unittest
from pathlib import Path
import numpy as np
import torch
from .model import GraphWave,mandel,symmetric,STRAIN_SCALE,PRESSURE_SCALE,PAIRS,TRIPLES
from operator3d.physics import piola


class NetworkPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(2026102404)
        cls.model=GraphWave();cls.model.load_state_dict(torch.load(Path(os.environ['WORKAI_GRAPH_MODEL'])/'candidate.pt',weights_only=True)['model_state'])
        cls.material={'lambda_pa':58.1e9,'mu_pa':26.1e9,'l_pa':-143e9,'m_pa':-297e9,'n_pa':-345e9}

    def test_zero_stress_and_translation_filter(self):
        for law in range(3):torch.testing.assert_close(self.model.stress(torch.zeros(3,3,dtype=torch.float64),law),torch.zeros(3,3,dtype=torch.float64),rtol=0,atol=0)
        for p in [3,4]:torch.testing.assert_close(self.model.derivative(p).sum(-1),torch.zeros(p+1,dtype=torch.float64),rtol=0,atol=4e-15)

    def test_unseen_strains_against_constitutive_teacher(self):
        gradient=torch.randn(137,3,3,dtype=torch.float64)*1e-4
        for law,name in enumerate(['linear','stvk','murnaghan']):
            expected=piola(gradient,self.material,name)[0]
            torch.testing.assert_close(self.model.stress(gradient,law),expected,rtol=1e-11,atol=2e-7)

    def test_objectivity_for_finite_rigid_rotation(self):
        q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
        if torch.linalg.det(q)<0:q[:,0]*=-1
        g=torch.randn(53,3,3,dtype=torch.float64)*1e-4;eye=torch.eye(3,dtype=torch.float64)
        for law in [1,2]:torch.testing.assert_close(self.model.stress(q@(eye+g)-eye,law),q@self.model.stress(g,law),rtol=2e-8,atol=1e-4)

    def test_polynomial_output_is_gradient_of_energy(self):
        z=(torch.randn(29,6,dtype=torch.float64)*.3).requires_grad_(True)
        energy=sum(w*torch.prod(z[:,list(pair)],dim=1) for w,pair in zip(self.model.quadratic,PAIRS))
        energy=energy+STRAIN_SCALE*sum(w*torch.prod(z[:,list(term)],dim=1) for w,term in zip(self.model.cubic,TRIPLES))
        gradient=torch.autograd.grad(energy.sum(),z)[0];a,b=self.model.stress_parts(z)
        torch.testing.assert_close(gradient,a+STRAIN_SCALE*b,rtol=1e-12,atol=1e-12)

    def test_parameters_actually_control_predictions(self):
        model=GraphWave();model.load_state_dict(self.model.state_dict());g=torch.randn(17,3,3,dtype=torch.float64)*1e-4
        original=model.stress(g,2).detach()
        with torch.no_grad():model.quadratic.mul_(1.1)
        self.assertGreater(float((torch.linalg.vector_norm(model.stress(g,2)-original)/torch.linalg.vector_norm(original)).detach()),.09)


if __name__=='__main__':unittest.main()
