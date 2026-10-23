import itertools,unittest
import numpy as np
import torch
from solid.mesh import prepare_mesh
from solid.schema import MATERIAL,default_scene
from operator3d.physics import piola,internal,U0
from operator3d.data import tensors
from .basis import assemble,build_basis
from .response import oscillators,quadratic_stresses
from .energy import EnergyCoupling
from .propagator import linear_force_oscillators
from .nonlinear import stress

def cube():
    p=np.array([[i&1,(i>>1)&1,(i>>2)&1] for i in range(8)],float)*.001
    tet=np.array([[0,1<<a,(1<<a)+(1<<b),7] for a,b,c in itertools.permutations(range(3))]);m=prepare_mesh(p,tet);s=default_scene();s['source']['center_mm']=[0.,.5,.5];s['geometry'].update(length_mm=1.,width_mm=1.,thickness_mm=1.);return m,s

class ModalTests(unittest.TestCase):
    def test_stiffness_matches_independent_variational_force(self):
        mesh,scene=cube();k,m=assemble(mesh,MATERIAL);rng=np.random.default_rng(73);u=rng.normal(size=(8,3))*1e-9;parts=torch.zeros((8,3,3),dtype=torch.float64);parts[:,0]=torch.tensor(u/U0);force,_,_=internal(parts,tensors(scene,mesh))
        np.testing.assert_allclose(k@u.ravel(),force[:,0].numpy().ravel(),rtol=1e-12,atol=1e-14);np.testing.assert_allclose(k.toarray(),k.toarray().T,rtol=0,atol=1e-10);self.assertAlmostEqual(m.sum(),3*2700e-9,places=15)

    def test_modes_and_residual_flexibility_recover_static_force(self):
        mesh,scene=cube();scene['boundary']['fixed_face']='xmax';basis=build_basis(mesh,scene,8);rng=np.random.default_rng(92);f=rng.normal(size=(24,2));f[basis['fixed']]=0.;phi=basis['phi'];low=phi@((phi.T@f)*1e-12/basis['lambda_us2'][:,None]);rest=basis['residual_flexibility'](f);total=low+rest;free=~basis['fixed'];force=basis['stiffness']@total
        np.testing.assert_allclose(force[free],f[free],rtol=5e-5,atol=5e-5);np.testing.assert_allclose(phi.T@(basis['mass'][:,None]*rest),0,atol=1e-16);self.assertTrue((total[basis['fixed']]==0).all())

    def test_duhamel_matches_rigid_acceleration_and_oscillator(self):
        time=np.arange(1001)*.001;lambdas=np.array([0.,3.]);force=np.ones((len(time),2))*2.;q=oscillators(force,time,lambdas)
        np.testing.assert_allclose(q[:,0],time*time,atol=1e-13);np.testing.assert_allclose(q[:,1],2/3*(1-np.cos(np.sqrt(3)*time)),rtol=3e-7,atol=1e-12);self.assertTrue((q[0]==0).all())

    def test_piecewise_linear_integrator_handles_high_modes_without_bias(self):
        time=np.arange(401)*.025;lambdas=np.array([0.,3.,10000.]);forces=np.broadcast_to(2+3*time[:,None],(len(time),3));q=linear_force_oscillators(forces,time,lambdas);omega=np.sqrt(lambdas[1:])
        expected=2/lambdas[None,1:]*(1-np.cos(time[:,None]*omega))+3/lambdas[None,1:]*(time[:,None]-np.sin(time[:,None]*omega)/omega)
        np.testing.assert_allclose(q[:,0],time**2+.5*time**3,rtol=2e-13,atol=1e-13);np.testing.assert_allclose(q[:,1:],expected,rtol=1e-11,atol=2e-12)

    def test_quadratic_energy_basis_matches_even_stress_expansion(self):
        rng=np.random.default_rng(79);g=rng.normal(size=(8,3,3))*1e-4;geo,a,b,c=quadratic_stresses(g,MATERIAL);coeff=np.array([MATERIAL['n_pa'],MATERIAL['m_pa']-MATERIAL['n_pa']/2,MATERIAL['l_pa']-MATERIAL['m_pa']+MATERIAL['n_pa']/2])/1e11
        even={}
        for law in ['stvk','murnaghan']:even[law]=sum(piola(torch.tensor(sign*g),MATERIAL,law)[0].numpy() for sign in [-1,1])/2
        np.testing.assert_allclose(even['stvk'],geo,rtol=2e-10,atol=1e-7);np.testing.assert_allclose(even['murnaghan']-even['stvk'],coeff[0]*a+coeff[1]*b+coeff[2]*c,rtol=3e-6,atol=.01)

    def test_learned_cubic_energy_is_objective_and_differentiable(self):
        model=EnergyCoupling();g=torch.tensor([[.0002,.0001,0.],[0.,-.0001,.00003],[.00001,0.,.00005]],dtype=torch.float64,requires_grad=True)
        w=model.cubic_energy(g,MATERIAL);stress=torch.autograd.grad(w,g,create_graph=True)[0];full=piola(g,MATERIAL,'murnaghan')[0];geo=piola(g,MATERIAL,'stvk')[0];torch.testing.assert_close(stress,full-geo,rtol=2e-10,atol=1e-8)
        angle=.43;rotation=torch.tensor([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]],dtype=torch.float64);rotated=rotation@(torch.eye(3,dtype=torch.float64)+g)-torch.eye(3,dtype=torch.float64);torch.testing.assert_close(w,model.cubic_energy(rotated,MATERIAL),rtol=2e-11,atol=1e-12)
        derivative=torch.autograd.grad(w,model.raw)[0];self.assertTrue(torch.isfinite(derivative).all());self.assertGreater(float(derivative.abs().sum()),0)
        with torch.no_grad():model.raw.copy_(torch.tensor([-100.,0.,100.]))
        self.assertTrue(((model()>=.9)&(model()<=1.1)).all())

    def test_full_feedback_stress_matches_independent_hyperelastic_law(self):
        g=np.random.default_rng(98).normal(size=(8,3,3))*1e-4
        for law,alpha in [('stvk',None),('murnaghan',[1.,1.,1.])]:
            expected=piola(torch.tensor(g),MATERIAL,law)[0].numpy();np.testing.assert_allclose(stress(g,MATERIAL,alpha),expected,rtol=1e-12,atol=1e-7)

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
