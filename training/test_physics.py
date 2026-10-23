import json
from pathlib import Path
import unittest
import torch
from .physics import first_piola, energy, derivative, momentum, boundary_residual, pulse, EPS, L0, T0
from .model import GuidedWaveField

card=json.loads((Path(__file__).parents[1]/'data/materials/aluminum-polycrystal-blaschke2017.json').read_text())
material={k:card[k] for k in ['lambda_pa','mu_pa']} | {k:card['third_order'][k] for k in ['l_pa','m_pa','n_pa']}
scene={'geometry':{'length_mm':160,'thickness_mm':1},'source':{'frequency_mhz':.4,'cycles':5,'delay_us':0,'phase_deg':0,'amplitude_pa':5e6},'acquisition':{'duration_us':26},'material':{'rho_kg_m3':2700,'cl_m_s':((material['lambda_pa']+2*material['mu_pa'])/2700)**.5,'ct_m_s':(material['mu_pa']/2700)**.5},'nonlinear':material}

class PhysicsTests(unittest.TestCase):
    def test_piola_is_energy_gradient_with_murnaghan_convention(self):
        torch.manual_seed(7)
        g=(torch.randn(8,3,3,dtype=torch.float64)*1e-4).requires_grad_()
        expected=torch.autograd.grad(energy(g,material).sum(),g)[0]
        actual,_,_=first_piola(g,material)
        torch.testing.assert_close(actual,expected,rtol=2e-12,atol=1e-6)

    def test_rigid_rotation_is_stress_free(self):
        angle=torch.tensor(.73,dtype=torch.float64);c,s=angle.cos(),angle.sin()
        f=torch.tensor([[c,-s,0],[s,c,0],[0,0,1]],dtype=torch.float64)
        p,_,j=first_piola((f-torch.eye(3,dtype=torch.float64))[None],material)
        self.assertLess(float(p.abs().max()),1e-4);self.assertAlmostEqual(float(j),1.,places=12)

    def test_linear_bulk_wave_satisfies_scaled_momentum(self):
        q=torch.tensor([[.3,.01,2.],[.7,-.02,4.]],dtype=torch.float64,requires_grad=True)
        omega=2*torch.pi*.4; k=omega/(scene['material']['cl_m_s']*T0/L0)
        u=torch.cos(k*q[:,0]-omega*q[:,2]); zero=q[:,0]*0
        linear=torch.stack([u,zero],1);parts=torch.stack([linear,linear*0,linear*0],1)
        r,_,_=momentum(parts,q,scene)
        self.assertLess(float(r[:,0].detach().abs().max()),1e-12)

    def test_left_traction_has_correct_normal_sign_and_units(self):
        q=torch.tensor([[0.,0.,3.],[0.,.02,5.]],dtype=torch.float64,requires_grad=True)
        physical_gradient=-scene['source']['amplitude_pa']*pulse(q[:,2],scene)/(material['lambda_pa']+2*material['mu_pa'])
        displacement=torch.stack([physical_gradient/EPS*q[:,0],q[:,1]*0],1)
        parts=torch.stack([displacement,displacement*0,displacement*0],1)
        r,_,_=boundary_residual(parts,q,torch.tensor([[-1.,0.],[-1.,0.]],dtype=torch.float64),torch.ones(2,dtype=torch.float64),scene)
        self.assertLess(float(r[:,0].detach().abs().max()),1e-12)

    def test_hard_initial_conditions_parity_and_amplitude_order(self):
        torch.manual_seed(8); model=GuidedWaveField(scene,[[1.,.2],[.002,.0003],[.007,.001]])
        q=torch.tensor([[2.,.03,0.],[2.,-.03,0.]],dtype=torch.float64,requires_grad=True)
        p=model(q);self.assertEqual(float(p.detach().abs().max()),0.)
        self.assertEqual(float(derivative(p[:,0,0],q)[:,2].detach().abs().max()),0.)
        q=torch.tensor([[2.,.03,8.],[2.,-.03,8.]],dtype=torch.float64)
        p=model(q);torch.testing.assert_close(p[0,:,0],p[1,:,0]);torch.testing.assert_close(p[0,:,1],-p[1,:,1])
        controls=model.controls(q)
        torch.testing.assert_close(controls[:,4]-controls[:,3],4*(controls[:,2]-controls[:,1]),rtol=1e-11,atol=1e-14)

if __name__=='__main__':unittest.main()
