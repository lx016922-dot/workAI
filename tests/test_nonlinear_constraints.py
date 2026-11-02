import unittest
import numpy as np
from graphwave.nonlinear_constraints import stress_energy_parts,divergence_parts,make_dataset,evaluate
from pathlib import Path

MATERIAL={'lambda_pa':58.1e9,'mu_pa':26.1e9,'l_pa':-143e9,'m_pa':-297e9,'n_pa':-345e9,'rho_kg_m3':2700}


class NonlinearMomentumTest(unittest.TestCase):
    def test_quartic_fields_expose_an_sbp_consistent_high_order_error(self):
        from numpy.polynomial.legendre import Legendre
        from spectral3d.mesh import gll
        from graphwave.modal_calibration import sbp_checks
        root=Path(__file__).resolve().parents[1]
        with np.load(root/'offline/model/continued-weights.npz',allow_pickle=False) as saved:weights=dict(saved)
        x,w,exact,_=gll(6);a,b=Legendre.basis(5)(x),Legendre.basis(6)(x)
        altered=exact+10*(np.outer(a,w*b)-np.outer(b,w*a))
        self.assertTrue(sbp_checks(altered,w)['passed'])
        cubic=make_dataset(MATERIAL,24,2026102510,degree=6)
        extended=make_dataset(MATERIAL,24,2026102510,degree=6,polynomial_degree=4)
        self.assertEqual({int(p['polynomial_degree']) for p in extended},{3,4})
        _,baseline=evaluate(exact,weights,extended,details=True)
        _,old=evaluate(altered,weights,cubic,details=True)
        _,new=evaluate(altered,weights,extended,details=True)
        self.assertTrue(baseline['all_physical_checks_within_5_percent'])
        self.assertLess(max(old['maximum_normalized_residual'].values()),.025)
        self.assertGreater(max(new['maximum_normalized_residual'].values()),.025)

    def test_sixth_order_tangent_and_full_nonlinear_constraints(self):
        import json
        from graphwave.modal_calibration import stiffness,filter_basis,sbp_checks
        from graphwave.nonlinear_constraints import rigid_motion_checks
        from graphwave.buffered_inference import graph_message
        from graphwave.inference import inverse_mapping
        from spectral3d.mesh import make_mesh,gll
        root=Path(__file__).resolve().parents[1]
        scene=json.loads((root/'research/comsol-validation/data/eigen-030-v7/scene.json').read_text())['scene']
        with np.load(root/'offline/model/continued-weights.npz',allow_pickle=False) as saved:weights=dict(saved)
        nodes,w,exact,_=gll(6);basis=filter_basis(nodes,2,w)
        self.assertEqual(len(basis),4)
        learned=exact+np.einsum('a,aij->ij',[.02,-.015,.03,-.01],basis)
        self.assertTrue(sbp_checks(learned,w)['passed'])
        self.assertTrue(rigid_motion_checks(learned,weights)['passed'])
        rng=np.random.default_rng(2026102510)
        for _ in range(20):
            u,v=rng.normal(size=(2,7))
            self.assertAlmostEqual(float(u@(w*(learned@v))+(learned@u)@(w*v)),
                                   float(u[-1]*v[-1]-u[0]*v[0]),places=12)
        mesh=make_mesh(scene,6,8.)
        field=rng.normal(size=mesh['points'].shape)*1e-9
        matrix=stiffness(mesh,learned,weights['quadratic_tensor'])
        force,_,_,_=graph_message(field,mesh['elements'],learned,inverse_mapping(mesh),mesh['volume_weight'],
                                  weights['quadratic_tensor'],weights['cubic_tensor'],0)
        self.assertLess(np.linalg.norm(matrix@field.ravel()-force.ravel())/np.linalg.norm(force),1e-12)
        patches=make_dataset(MATERIAL,12,2026102510,degree=6)
        _,report=evaluate(exact,weights,patches,details=True)
        self.assertTrue(report['all_physical_checks_within_5_percent'])

    def test_sbp_training_preserves_boundary_flux_for_arbitrary_fields(self):
        from graphwave.modal_calibration import filter_basis,sbp_checks
        from spectral3d.mesh import gll
        x,w,exact,_=gll(5)
        basis=filter_basis(x,2,w)
        self.assertEqual(len(basis),2)
        derivative=exact+np.einsum('a,aij->ij',[.13,-.09],basis)
        self.assertTrue(sbp_checks(derivative,w)['passed'])
        np.testing.assert_allclose(derivative@np.stack([x*0+1,x,x*x],1),
                                  np.stack([x*0,x*0+1,2*x],1),rtol=0,atol=3e-14)
        rng=np.random.default_rng(2026102590)
        for _ in range(20):
            u,v=rng.normal(size=(2,6))
            self.assertAlmostEqual(float(u@(w*(derivative@v))+(derivative@u)@(w*v)),
                                   float(u[-1]*v[-1]-u[0]*v[0]),places=12)
        broken=derivative.copy();broken[0,0]+=.01
        self.assertFalse(sbp_checks(broken,w)['passed'])

    def test_fifth_order_constraints_and_mode_sampling(self):
        from graphwave.nonlinear_constraints import rigid_motion_checks
        from graphwave.modal_calibration import modal_sampling,project_filter_consistency
        from spectral3d.mesh import gll,make_mesh
        root=Path(__file__).resolve().parents[1]
        import json
        scene=json.loads((root/'research/comsol-validation/data/eigen-030-v7/scene.json').read_text())['scene']
        with np.load(root/'offline/model/continued-weights.npz',allow_pickle=False) as saved:weights=dict(saved)
        nodes,_,exact,_=gll(5)
        derivative=project_filter_consistency(weights['derivative_5'],nodes,1)
        self.assertTrue(rigid_motion_checks(derivative,weights)['passed'])
        patches=make_dataset(MATERIAL,12,2026102480,degree=5)
        _,report=evaluate(exact,weights,patches,details=True)
        self.assertTrue(report['all_physical_checks_within_5_percent'])
        broken=exact.copy();broken[0,0]+=.2;broken[0,1]-=.2
        _,failure=evaluate(broken,weights,patches,details=True)
        self.assertFalse(failure['all_physical_checks_within_5_percent'])
        source=make_mesh(scene,5,2.5);target=make_mesh(scene,4,2.5)
        def field(x):return np.stack([x[:,0]**3,x[:,0]*x[:,1],x[:,2]**2],axis=-1)[None]
        sampled=modal_sampling(source,target['points'])(field(source['points']))
        np.testing.assert_allclose(sampled,field(target['points']),rtol=1e-12,atol=1e-18)

    def test_learned_affine_filter_does_not_create_force_under_finite_rigid_rotation(self):
        from graphwave.modal_calibration import filter_basis,project_filter_consistency
        from graphwave.buffered_inference import graph_message
        from spectral3d.mesh import gll
        with np.load(Path(__file__).resolve().parents[1]/'offline/model/continued-weights.npz',allow_pickle=False) as saved:
            weights=dict(saved)
        nodes,_,_,_=gll(4);basis=filter_basis(nodes,1)
        derivative=project_filter_consistency(weights['derivative_4'],nodes,1)+np.einsum('a,aij->ij',np.linspace(-.12,.14,len(basis)),basis)
        patch=make_dataset(MATERIAL,6,2026102430)[0]
        coordinates=np.stack(np.meshgrid(nodes,nodes,nodes,indexing='ij'),-1).reshape(-1,3)*patch['size']/2
        angle=.73;rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        displacement=np.ascontiguousarray(coordinates@(rotation-np.eye(3)).T)
        for law in [1,2]:
            force,energy,_,_=graph_message(displacement,patch['cells'],derivative,patch['inverse'],patch['volume'],
                                          weights['quadratic_tensor'],weights['cubic_tensor'],law)
            self.assertLess(np.linalg.norm(force),1e-7)
            self.assertLess(abs(energy),1e-16)

    def test_all_stress_divergences_against_complex_step(self):
        rng=np.random.default_rng(2026102414)
        g=rng.normal(size=(19,3,3))*4e-5
        h=rng.normal(size=(19,3,3,3))*.03;h=(h+h.swapaxes(-1,-2))/2
        actual=divergence_parts(g,h,MATERIAL);expected=np.zeros_like(actual)
        for j in range(3):
            stress,_=stress_energy_parts(g.astype(complex)+1j*1e-28*h[...,j],MATERIAL)
            expected+=stress.imag[...,j]/1e-28
        for branch in range(5):
            self.assertLess(np.linalg.norm(actual[branch]-expected[branch])/np.linalg.norm(expected[branch]),2e-13)

    def test_shear_displacement_drives_longitudinal_nonlinear_motion(self):
        g=np.zeros((1,3,3));g[0,1,0]=3e-4
        h=np.zeros((1,3,3,3));h[0,1,0,0]=.2
        div=divergence_parts(g,h,MATERIAL)
        self.assertAlmostEqual(div[0,0,1]/(.2*MATERIAL['mu_pa']),1,places=13)
        self.assertEqual(div[0,0,0],0)
        expected=(MATERIAL['lambda_pa']+2*MATERIAL['mu_pa'])*3e-4*.2
        self.assertAlmostEqual(div[3,0,0]/expected,1,places=13)
        self.assertGreater(abs(div[4,0,0]),1e5)

    def test_full_piola_is_energy_gradient_and_obeys_angular_momentum(self):
        rng=np.random.default_rng(1209);g=rng.normal(size=(7,3,3))*1e-4
        stress,_=stress_energy_parts(g,MATERIAL)
        for i in range(3):
            for j in range(3):
                perturbed=g.astype(complex);perturbed[:,i,j]+=1j*1e-28
                _,energy=stress_energy_parts(perturbed,MATERIAL)
                np.testing.assert_allclose(energy.imag/1e-28,stress[...,i,j],rtol=3e-12,atol=1e-7)
        current_stress=stress[2]@(np.eye(3)+g).swapaxes(-1,-2)
        self.assertLess(np.linalg.norm(current_stress-current_stress.swapaxes(-1,-2))/np.linalg.norm(current_stress),1e-13)

    def test_rigid_rotation_has_zero_nonlinear_stress_and_energy(self):
        angle=.7;c,s=np.cos(angle),np.sin(angle)
        rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        stress,energy=stress_energy_parts((rotation-np.eye(3))[None],MATERIAL)
        self.assertLess(np.linalg.norm(stress[1:3])/MATERIAL['mu_pa'],1e-13)
        self.assertLess(np.linalg.norm(energy[1:3])/MATERIAL['mu_pa'],1e-15)

    def test_motion_balance_detects_filter_error_despite_affine_consistency(self):
        path=Path(__file__).resolve().parents[1]/'offline/model/continued-weights.npz'
        with np.load(path,allow_pickle=False) as file:weights=dict(file)
        dataset=make_dataset(MATERIAL,count=6)
        baseline,report=evaluate(weights['derivative_4'],weights,dataset,details=True)
        self.assertTrue(report['all_physical_checks_within_5_percent'])
        change=.1*np.outer([1.,0,0,0,-1],[1.,0,-2,0,1.])
        altered,report=evaluate(weights['derivative_4']+change,weights,dataset,details=True)
        self.assertGreater(np.linalg.norm(altered),100*np.linalg.norm(baseline))
        self.assertFalse(report['all_physical_checks_within_5_percent'])


if __name__=='__main__':unittest.main()
