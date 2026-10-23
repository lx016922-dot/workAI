import unittest
import numpy as np
from .schema import default_scene,validate_scene,MATERIAL
from .mesh import make_mesh,prepare_mesh,locate,nodal_load
from .fem import stress_energy,internal_force,lumped_mass

C=np.array([MATERIAL[k] for k in ['lambda_pa','mu_pa','l_pa','m_pa','n_pa']])

class PhysicsTests(unittest.TestCase):
    def test_energy_gradient_and_rotation(self):
        g=np.array([[2e-4,1e-4,-3e-5],[4e-5,-1e-4,2e-5],[1e-5,3e-5,4e-5]])
        for law in [0,1,2]:
            p,w,_,_=stress_energy(g,*C,law);numerical=np.zeros((3,3));h=1e-8
            for i in range(3):
                for j in range(3):
                    perturb=np.zeros((3,3));perturb[i,j]=h
                    numerical[i,j]=(stress_energy(g+perturb,*C,law)[1]-stress_energy(g-perturb,*C,law)[1])/(2*h)
            np.testing.assert_allclose(p,numerical,rtol=2e-7,atol=.01)
        a=.53;r=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
        p,w,strain,j=stress_energy(r-np.eye(3),*C,2)
        self.assertLess(np.linalg.norm(p),1e-4);self.assertLess(abs(w),1e-15);self.assertAlmostEqual(j,1.)

    def test_affine_interpolation_and_internal_balance(self):
        p=np.array([[0,0,0],[.001,0,0],[0,.001,0],[0,0,.001]])
        mesh=prepare_mesh(p,[[0,1,2,3]]);index,weights=locate(mesh,[.2,.3,.1])
        np.testing.assert_allclose(weights@[1+2*x-3*y+z for x,y,z in p],1+2*.0002-3*.0003+.0001)
        gradient=np.array([[1e-4,2e-5,0],[0,-5e-5,1e-5],[3e-5,0,2e-5]])
        u=p@gradient.T;f,w,_,_=internal_force(u,mesh['tetra'],mesh['grad'],mesh['volume'],C,2)
        np.testing.assert_allclose(f.sum(axis=0),0,atol=1e-12)
        np.testing.assert_allclose(np.cross(p+u,f).sum(axis=0),0,atol=1e-14)
        self.assertGreater(w,0);self.assertAlmostEqual(lumped_mass(mesh,2700).sum(),2700e-9/6)

    def test_linear_eigenmode_integrator(self):
        # Independent exact semidiscrete vibration of a free single tetrahedron.
        from scipy.linalg import eigh
        p=np.array([[0,0,0],[.001,0,0],[0,.001,0],[0,0,.001]])
        m=prepare_mesh(p,[[0,1,2,3]]);mass=np.repeat(lumped_mass(m,2700),3)
        stiffness=np.zeros((12,12))
        for column in range(12):
            u=np.zeros(12);u[column]=1e-8
            stiffness[:,column]=internal_force(u.reshape(4,3),m['tetra'],m['grad'],m['volume'],C,0)[0].ravel()/1e-8
        np.testing.assert_allclose(stiffness,stiffness.T,rtol=1e-13,atol=1e-8)
        eigen,mode=eigh(stiffness,np.diag(mass));omega=np.sqrt(eigen[-1]);dt=.02/omega
        initial=mode[:,-1]*1e-12;u=initial.copy();v=np.zeros(12);acc=-stiffness@u/mass
        steps=400
        for _ in range(steps):
            u+=dt*v+.5*dt*dt*acc;next_a=-stiffness@u/mass;v+=.5*dt*(acc+next_a);acc=next_a
        np.testing.assert_allclose(u,initial*np.cos(omega*dt*steps),atol=np.max(abs(initial))*2e-4)

    def test_real_shapes_holes_receivers_and_load_units(self):
        for kind in ['plate','pipe','hole','notch']:
            scene=validate_scene(default_scene(kind));m=make_mesh(scene);force,mask=nodal_load(m,scene['source'])
            self.assertTrue(np.all(m['volume']>0));self.assertTrue(mask.any())
            np.testing.assert_allclose(force.sum(axis=0),m['area'][mask].sum()*1e6*np.array([1.,0,0]),atol=1e-12)
            for rx in scene['receivers']:locate(m,rx['point_mm'])
            if kind=='hole':
                with self.assertRaises(ValueError):locate(m,[6,2,1])
            if kind=='pipe':
                with self.assertRaises(ValueError):locate(m,[6,0,0])
            if kind=='notch':
                with self.assertRaises(ValueError):locate(m,[6,2,1.8])
            if kind=='plate':self.assertAlmostEqual(m['volume'].sum()*1e9,96.)

    def test_invalid_sources_sampling_and_material(self):
        for edit in [lambda s:s['source'].update(direction=[0,0,0]),lambda s:s['material'].update(mu_pa=-1),
                     lambda s:s['acquisition'].update(sample_step_us=1),lambda s:s['boundary'].update(fixed_face='xmin')]:
            s=default_scene();edit(s)
            with self.assertRaises(ValueError):validate_scene(s)

    def test_control_mesh_contract_and_normalization(self):
        import copy
        from .worker import validate_mesh_reference
        from .study import validate_study
        baseline=default_scene();baseline['constitutive']='linear';baseline['source']['direction']=[1,2,-1]
        baseline=validate_scene(baseline)
        self.assertEqual(baseline,validate_scene(baseline))
        control=copy.deepcopy(baseline);control['constitutive']='murnaghan';control['source']['amplitude_mpa']*=2
        validate_mesh_reference(control,{'scene':baseline},'controls')
        with self.assertRaises(ValueError):validate_mesh_reference(control,{'scene':baseline},'time')
        control['source']['frequency_mhz']*=2
        with self.assertRaises(ValueError):validate_mesh_reference(control,{'scene':baseline},'controls')
        too_large=default_scene();too_large['source']['amplitude_mpa']=6
        with self.assertRaises(ValueError):validate_study({'scene':too_large})

    def test_float64_output_avoids_measured_float32_quantization(self):
        from .fem import solve
        mesh=prepare_mesh(np.array([[0,0,0],[.001,0,0],[0,.001,0],[0,0,.001]]),[[0,1,2,3]])
        scene=default_scene();scene['acquisition'].update(duration_us=10.,sample_step_us=.5)
        load,_=nodal_load(mesh,scene['source']);fixed=np.array([],dtype=int)
        single,first=solve(scene,mesh,load,fixed,output_precision='float32')
        double,second=solve(scene,mesh,load,fixed,output_precision='float64')
        np.testing.assert_array_equal(single,double.astype(np.float32))
        self.assertGreater(first['max_export_roundoff_m'],0)
        self.assertEqual(first['max_export_roundoff_m'],np.max(abs(single-double)))
        self.assertEqual(second['max_export_roundoff_m'],0)
        self.assertEqual(double.dtype,np.float64)

if __name__=='__main__':unittest.main()
