import unittest
import numpy as np
from graphwave.nonlinear_constraints import stress_energy_parts,divergence_parts,make_dataset,evaluate
from pathlib import Path

MATERIAL={'lambda_pa':58.1e9,'mu_pa':26.1e9,'l_pa':-143e9,'m_pa':-297e9,'n_pa':-345e9,'rho_kg_m3':2700}


class NonlinearMomentumTest(unittest.TestCase):
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
