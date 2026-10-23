import unittest
import numpy as np
from solid.schema import default_scene,MATERIAL
from .mesh import make_mesh
from .implicit import assemble,prepare,solve
from .solver import solve as explicit_solve,internal_force
from .curved_solver import internal_force as curved_force

C=np.array([MATERIAL[k] for k in ['lambda_pa','mu_pa','l_pa','m_pa','n_pa']])


class ImplicitTests(unittest.TestCase):
    def test_assembled_tangent_matches_independent_force(self):
        rng=np.random.default_rng(453)
        for kind in ['plate','pipe']:
            s=default_scene(kind);m=make_mesh(s,2,4.);k=assemble(m,s['material']);u=rng.normal(size=m['points'].shape)*1e-9
            if kind=='pipe':f=curved_force(u,m['elements'],m['derivative'],m['inverse_jacobian'],m['volume_weight'],C,0)[0]
            else:f=internal_force(u,m['elements'],m['derivative'],m['size'],m['volume_weight'],C,0)[0]
            np.testing.assert_allclose(k@u.ravel(),f.ravel(),rtol=2e-12,atol=1e-13)
            self.assertLess(np.max(abs((k-k.T).data),initial=0),1e-6)

    def test_full_nonlinear_newmark_matches_small_step_verlet(self):
        s=default_scene();s['geometry'].update(length_mm=4.,width_mm=2.,thickness_mm=1.);s['source'].update(center_mm=[0,1,.5],frequency_mhz=.5,cycles=2.,direction=[.9950371902099893,0,.09950371902099893]);s['acquisition'].update(duration_us=6.);s['receivers']=[{'name':'P1','point_mm':[2.7,1,.5],'direction':[1,0,0]}];m=make_mesh(s,3,2.);p=prepare(s,m,8);outputs=[]
        for law in ['stvk','murnaghan']:
            s['constitutive']=law;a=solve(s,m,8,True,prepared=p);b=explicit_solve(s,m,.15,True);outputs.append((a['fields_m'],b['fields_m']))
            self.assertLess(np.linalg.norm(a['fields_m']-b['fields_m'])/np.linalg.norm(b['fields_m']),.002)
            self.assertLess(a['diagnostics']['external_work_balance_relative'],.0001)
        da=outputs[1][0]-outputs[0][0];db=outputs[1][1]-outputs[0][1]
        self.assertLess(np.linalg.norm(da-db)/np.linalg.norm(db),.003)


if __name__=='__main__':unittest.main()
