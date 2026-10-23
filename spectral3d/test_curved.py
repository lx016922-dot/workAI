import unittest
import numpy as np
from solid.schema import default_scene,MATERIAL
from solid.fem import stress_energy
from .mesh import make_mesh,boundary_data,interpolation
from .curved_solver import internal_force

C=np.array([MATERIAL[k] for k in ['lambda_pa','mu_pa','l_pa','m_pa','n_pa']])


class CurvedTests(unittest.TestCase):
    def mesh(self,kind,degree=3):return make_mesh(default_scene(kind),degree,2.)
    def force(self,m,u,law=2):return internal_force(u,m['elements'],m['derivative'],m['inverse_jacobian'],m['volume_weight'],C,law)

    def test_curved_geometry_mass_and_planar_traction(self):
        for kind in ['pipe','hole']:
            s=default_scene(kind);g=s['geometry'];m=self.mesh(kind);load,fixed,area=boundary_data(m,s)
            volume=np.pi*(g['outer_radius_mm']**2-g['inner_radius_mm']**2)*g['length_mm'] if kind=='pipe' else (g['length_mm']*g['width_mm']-np.pi*g['hole_radius_mm']**2)*g['thickness_mm']
            self.assertLess(abs(m['volume_weight'].sum()*1e9/volume-1),2e-4)
            np.testing.assert_allclose(load.sum(0),np.array([1.,0,0])*area*1e6,atol=1e-11)
            self.assertTrue(np.all(m['mass']>0))
            exact_area=np.pi*(3**2-2**2) if kind=='pipe' else 8.
            self.assertLess(abs(area*1e6/exact_area-1),2e-4)

    def test_isoparametric_affine_patch(self):
        gradient=np.array([[1e-4,2e-5,0],[1e-5,-3e-5,4e-5],[3e-5,0,2e-5]])
        for kind in ['pipe','hole']:
            m=self.mesh(kind);u=m['points']@gradient.T;f,energy,strain,j=self.force(m,u)
            p,w,expected_strain,expected_j=stress_energy(gradient,*C,2)
            self.assertAlmostEqual(energy/(w*m['volume_weight'].sum()),1.,places=10)
            self.assertAlmostEqual(strain/expected_strain,1.,places=10)
            np.testing.assert_allclose(f.sum(0),0,atol=1e-11)
            np.testing.assert_allclose(np.cross(m['points']+u,f).sum(0),0,atol=1e-13)

    def test_rigid_rotation_and_discrete_energy_derivative(self):
        m=self.mesh('pipe');angle=.4;r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        f,energy,strain,j=self.force(m,m['points']@(r-np.eye(3)).T)
        self.assertLess(np.max(abs(f)),2e-7);self.assertLess(abs(energy),1e-20);self.assertAlmostEqual(j,1.,places=10)
        rng=np.random.default_rng(23);u=rng.normal(size=m['points'].shape)*1e-9;direction=rng.normal(size=u.shape);direction/=np.linalg.norm(direction);h=1e-12
        f,*_=self.force(m,u);numerical=(self.force(m,u+h*direction)[1]-self.force(m,u-h*direction)[1])/(2*h)
        self.assertAlmostEqual(float((f*direction).sum()),numerical,delta=abs(numerical)*2e-6)

    def test_arbitrary_curved_point_and_void(self):
        for kind,point,void in [('pipe',[4.71,2.31,.32],[5,0,0]),('hole',[3.71,1.13,.83],[6,2,1])]:
            m=self.mesh(kind);ids,w=interpolation(m,point)
            np.testing.assert_allclose(w@m['points'][ids],np.asarray(point)*.001,atol=1e-12)
            with self.assertRaises(ValueError):interpolation(m,void)


if __name__=='__main__':unittest.main()
