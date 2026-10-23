import unittest
import numpy as np
from scipy.linalg import eigh
from .safe import rectangle,assemble,modes,extensional_branch,shape_fit

RHO=2700.;LAM=58.1e9;MU=26.1e9;YOUNG=MU*(3*LAM+2*MU)/(LAM+MU)

class SafeTests(unittest.TestCase):
    def system(self,w=.004,t=.001,h=.00025):
        p,tri=rectangle(w,t,h);return assemble(p,tri,RHO,LAM,MU)

    def test_consistent_mass_and_independent_affine_energy(self):
        s=self.system();p=s['points_m'];area=.004*.001
        np.testing.assert_allclose(s['axial']@(s['M']@s['axial']),RHO*area,rtol=1e-14)
        u=np.column_stack([2*p[:,0]+3*p[:,1],4*p[:,0]+5*p[:,1],6*p[:,0]+7*p[:,1]]).ravel()
        expected=area*(LAM*(4+7)**2+2*MU*(4**2+7**2)+MU*((5+6)**2+3**2+2**2))
        np.testing.assert_allclose(u@(s['K0']@u),expected,rtol=1e-13)
        k=100.;np.testing.assert_allclose(s['axial']@(k*k*s['K2']@s['axial']),area*(LAM+2*MU)*k*k,rtol=1e-14)
        rotation=np.column_stack([np.zeros(len(p)),-p[:,1],p[:,0]]).ravel()
        self.assertLess(abs(rotation@(s['K0']@rotation)),expected*1e-13)

    def test_hermiticity_and_sparse_modes_agree_with_dense_eigenproblem(self):
        s=self.system(h=.0005);k=350.;K=s['K0']-1j*k*s['G']+k*k*s['K2'];a=K.toarray()
        np.testing.assert_allclose(a,a.conj().T,atol=1e-5)
        expected=eigh(a,s['M'].toarray(),eigvals_only=True);r=modes(s,k,count=8)
        np.testing.assert_allclose(r['frequency_hz'],np.sqrt(expected[:8])/(2*np.pi),rtol=1e-10)
        self.assertLess(max(r['relative_residual']),1e-9)

    def test_long_wavelength_limit_and_rotated_rectangle(self):
        a=extensional_branch(self.system(),10000.,YOUNG)
        self.assertLess(abs(a['phase_velocity_m_s']/np.sqrt(YOUNG/RHO)-1),1e-4)
        b=extensional_branch(self.system(.001,.004),300000.,YOUNG);c=extensional_branch(self.system(),300000.,YOUNG)
        np.testing.assert_allclose(b['wavenumber_rad_m'],c['wavenumber_rad_m'],rtol=1e-8)
        self.assertGreater(c['minimum_tracking_MAC'],.99)

    def test_complex_shape_fit_is_scale_invariant_not_an_energy_claim(self):
        s=self.system();r=modes(s,350);v=r['vectors'][:,3];u=(2+3j)*v
        a=shape_fit(s,v,u);self.assertLess(a['relative_mass_L2_shape_mismatch'],1e-14);np.testing.assert_allclose(a['complex_coefficient'],[2,3],atol=1e-13)
        a=shape_fit(s,v,u+r['vectors'][:,4]);self.assertAlmostEqual(a['relative_mass_L2_shape_mismatch'],1/np.sqrt(14),places=12);self.assertTrue(a['not_a_modal_energy_fraction'])

if __name__=='__main__':unittest.main()
