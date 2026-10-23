import copy
import unittest
import numpy as np
from solid.schema import default_scene,MATERIAL
from solid.fem import stress_energy
from .mesh import gll,make_mesh,boundary_data,interpolation
from .solver import constitutive,internal_force

C=np.array([MATERIAL[k] for k in ['lambda_pa','mu_pa','l_pa','m_pa','n_pa']])

def force(mesh,u,law=2):
    return internal_force(u,mesh['elements'],mesh['derivative'],mesh['size'],mesh['volume_weight'],C,law)


class SpectralPhysicsTests(unittest.TestCase):
    def test_derivatives_and_summation_by_parts(self):
        for p in range(2,7):
            x,w,d,_=gll(p)
            for k in range(p+1):
                np.testing.assert_allclose(d@x**k,k*x**(k-1) if k else x*0,atol=2e-13)
            boundary=np.zeros((p+1,p+1));boundary[0,0]=-1;boundary[-1,-1]=1
            np.testing.assert_allclose(w[:,None]*d+d.T*w[None,:],boundary,atol=2e-13)

    def test_geometry_loading_and_void(self):
        for kind in ['plate','notch']:
            scene=default_scene(kind);mesh=make_mesh(scene,3,2);load,fixed,area=boundary_data(mesh,scene)
            g=scene['geometry'];volume=g['length_mm']*g['width_mm']*g['thickness_mm']
            if kind=='notch': volume-=g['notch_width_mm']*g['width_mm']*g['notch_depth_mm']
            self.assertAlmostEqual(mesh['volume_weight'].sum()*1e9,volume,places=10)
            self.assertAlmostEqual(mesh['mass'].sum(),volume*1e-9*MATERIAL['rho_kg_m3'])
            self.assertAlmostEqual(area*1e6,8.)
            np.testing.assert_allclose(load.sum(0),[8.,0,0],atol=1e-12)
            if kind=='notch':
                with self.assertRaises(ValueError):interpolation(mesh,[6.,2.,1.9])

    def test_polynomial_receiver_reproduction(self):
        mesh=make_mesh(default_scene(),4,3.);ids,w=interpolation(mesh,[5.271,2.431,1.273]);p=mesh['points']*1000
        f=lambda x:x[...,0]**4+x[...,1]**3-3*x[...,2]**2+2
        self.assertAlmostEqual(w@f(p[ids]),f(np.array([5.271,2.431,1.273])),places=10)

    def test_independent_constitutive_and_rotation(self):
        g=np.array([[2e-4,1e-4,-3e-5],[4e-5,-1e-4,2e-5],[1e-5,3e-5,4e-5]])
        for law in range(3):
            a=constitutive(g,C,law);b=stress_energy(g,*C,law)
            np.testing.assert_allclose(a[0],b[0],rtol=2e-15,atol=1e-7)
            np.testing.assert_allclose(a[1:],b[1:],rtol=2e-15)
        angle=.5;r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        mesh=make_mesh(default_scene(),3,4.);u=mesh['points']@(r-np.eye(3)).T
        f,energy,strain,j=force(mesh,u)
        self.assertLess(np.max(abs(f)),1e-8);self.assertLess(abs(energy),1e-22);self.assertAlmostEqual(j,1.)

    def test_discrete_energy_gradient_and_conservation(self):
        scene=default_scene();scene['geometry'].update(length_mm=2.,width_mm=2.,thickness_mm=2.)
        mesh=make_mesh(scene,3,2.);rng=np.random.default_rng(481);u=rng.normal(size=mesh['points'].shape)*1e-8;direction=rng.normal(size=u.shape);direction/=np.linalg.norm(direction)
        for law in range(3):
            f,_,_,_=force(mesh,u,law);h=1e-11
            numerical=(force(mesh,u+h*direction,law)[1]-force(mesh,u-h*direction,law)[1])/(2*h)
            self.assertAlmostEqual(float((f*direction).sum()),numerical,delta=abs(numerical)*3e-7)
            np.testing.assert_allclose(f.sum(0),0,atol=1e-13)
            if law: np.testing.assert_allclose(np.cross(mesh['points']+u,f).sum(0),0,atol=1e-15)

    def test_exact_pure_bending_energy(self):
        lam,mu=C[:2];young=mu*(3*lam+2*mu)/(lam+mu);nu=lam/(2*(lam+mu));kappa=.03
        for p in [2,3,4]:
            scene=default_scene();mesh=make_mesh(scene,p,4.);x,y,z=(mesh['points']-np.array([0,.002,.001])).T
            u=np.array([-kappa*x*z,nu*kappa*y*z,.5*kappa*(x*x+nu*(z*z-y*y))]).T
            _,energy,_,_=force(mesh,u,0)
            expected=.5*young*kappa*kappa*.012*.004*.002**3/12
            self.assertAlmostEqual(energy/expected,1.,places=11)

    def test_exactly_six_rigid_modes_no_hourglass(self):
        scene=default_scene();scene['geometry'].update(length_mm=2.,width_mm=2.,thickness_mm=2.)
        mesh=make_mesh(scene,2,2.);n=mesh['points'].size;k=np.empty((n,n))
        for i in range(n):
            u=np.zeros(n);u[i]=1e-8;k[:,i]=force(mesh,u.reshape(-1,3),0)[0].ravel()/1e-8
        np.testing.assert_allclose(k,k.T,rtol=1e-13,atol=1e-7)
        mass=np.repeat(mesh['mass'],3);eigen=np.linalg.eigvalsh(k/np.sqrt(mass[:,None]*mass[None,:]))
        self.assertLess(np.max(abs(eigen[:6]))/eigen[-1],1e-14)
        self.assertGreater(eigen[6]/eigen[-1],1e-4)

    def test_small_transverse_error_cannot_hide_in_vector_acceptance(self):
        from .compare import compare,metric
        scene=default_scene();times=np.linspace(0,14,281);parts=np.zeros((len(times),2,3,3))
        parts[:,:,0,0]=np.sin(times[:,None])*1e-7;parts[:,:,0,2]=np.sin(times[:,None])*1e-9
        parts[:,:,2,0]=np.sin(2*times[:,None])*1e-10;parts[:,:,2,2]=np.sin(2*times[:,None])*1e-12
        different=parts.copy();different[:,:,:,2]*=1.4
        left={'case':{'scene':scene,'times':times,'parts':different,'diagnostics':[]}};right={'case':{'scene':scene,'times':times,'parts':parts,'diagnostics':[]}}
        result=compare(left,right);probe=result['cases'][0]['probes'][0]
        self.assertTrue(probe['checks']['full_vector']['pass']);self.assertFalse(probe['checks']['full_z']['pass']);self.assertFalse(result['all_probe_checks_pass'])
        tiny=metric(np.array([1e-15]),np.array([2e-15]));self.assertIsNone(tiny['relative_l2']);self.assertFalse(tiny['above_reporting_floor'])


if __name__=='__main__':unittest.main()
