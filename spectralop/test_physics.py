import unittest
import numpy as np
import torch
from solid.schema import default_scene,validate_scene
from spectral3d.mesh import make_mesh
from spectralop.data import tensors
from spectralop.physics import geometry,star,internal,choose_targets,physics_loss
from operator3d.physics import U0,piola
from operator3d.propagation_model import PropagationSeparatedField
from spectralop.data import verify_manifest
import json,tempfile
from pathlib import Path

class SpectralPhysicsTests(unittest.TestCase):
    def case(self,kind):
        s=default_scene(kind);s['geometry'].update(length_mm=4.,width_mm=3.,thickness_mm=1.5,outer_radius_mm=1.8,inner_radius_mm=.9,hole_x_mm=2.,hole_y_mm=1.5,hole_radius_mm=.4)
        s['source']['center_mm']=[0.,0.,0.];s['receivers'][0]['point_mm']=[.8,1.1,0.] if kind=='pipe' else [.8,1.,.7];s['receivers'][1]['point_mm']=[3.,1.1,0.] if kind=='pipe' else [3.,1.,.7]
        return tensors(s,make_mesh(s,3,2.))
    def test_complete_star_matches_global_assembly(self):
        for kind in ['plate','pipe','hole']:
            case=self.case(kind);rng=np.random.default_rng(32);p=torch.tensor(rng.normal(size=(len(case['points']),3,3))*[.1,1e-6,1e-5],dtype=torch.float64)
            allnodes=np.arange(len(p));full=star(case,allnodes);targets=allnodes[::max(1,len(p)//7)];patch=star(case,targets)
            a=internal(p,case,full)[0][targets];b=internal(p[patch['ids']],case,patch)[0]
            torch.testing.assert_close(a,b,rtol=2e-9,atol=2e-12)
    def test_force_is_discrete_energy_gradient(self):
        for kind in ['plate','pipe','hole']:
            case=self.case(kind);mesh=case['mesh'];patch=star(case,np.arange(len(case['points'])));g=geometry(case);rng=np.random.default_rng(21);u=torch.tensor(rng.normal(size=(len(case['points']),3))*1e-9,dtype=torch.float64,requires_grad=True)
            grad=torch.einsum('eai,eqaj->eqij',u[patch['cells']],patch['grad']);e=(grad+grad.transpose(-1,-2)+grad.transpose(-1,-2)@grad)/2;t=e.diagonal(dim1=-2,dim2=-1).sum(-1);e2=e@e;m=case['scene']['material'];a=m['n_pa'];b=m['m_pa']-a/2;c=m['l_pa']-m['m_pa']+a/2
            energy=.5*m['lambda_pa']*t*t+m['mu_pa']*(e*e).sum((-1,-2))+a/3*(e2@e).diagonal(dim1=-2,dim2=-1).sum(-1)+b*t*(e*e).sum((-1,-2))+c/3*t**3
            force=torch.autograd.grad((energy*patch['weights']).sum(),u)[0];parts=torch.stack([u/U0,u*0,u*0],1)
            actual=internal(parts,case,patch)[0].sum(1);torch.testing.assert_close(force,actual,rtol=1e-10,atol=1e-12)
    def test_curved_affine_gradient(self):
        case=self.case('pipe');patch=star(case,np.arange(len(case['points'])));a=torch.tensor([[.0001,.00003,0.],[0.,-.00004,0.],[0.,0.,.00002]],dtype=torch.float64);u=torch.as_tensor(case['mesh']['points'])@a.T
        grad=torch.einsum('eai,eqaj->eqij',u[patch['cells']],patch['grad']);torch.testing.assert_close(grad,a.expand_as(grad),rtol=1e-10,atol=1e-15)
    def test_initial_velocity_and_fixed_boundary_are_exact(self):
        c=self.case('plate');c['scene']['boundary']['fixed_face']='xmax';c=tensors(c['scene'],c['mesh']);model=PropagationSeparatedField([1.,1e-5,1e-4],32);ids=torch.arange(len(c['points']));times=torch.full((len(ids),),c['scene']['source']['delay_us'],dtype=torch.float64,requires_grad=True)
        u=model(c,ids,times);velocity=torch.autograd.grad(u.sum(),times)[0];self.assertEqual(int(torch.count_nonzero(u)),0);self.assertEqual(int(torch.count_nonzero(velocity)),0)
        fixed=model(c,ids,torch.full((len(ids),),3.,dtype=torch.float64))[c['fixed']];self.assertGreater(len(fixed),0);self.assertEqual(int(torch.count_nonzero(fixed)),0)
    def test_geometry_duplicates_cannot_hide_in_unused_parameters(self):
        s=default_scene('plate');other=json.loads(json.dumps(s));other['geometry']['hole_radius_mm']*=1.1
        manifest={'phase':'completed','split_counts':{'train':1,'validation':1,'test':0},'cases':[{'case_id':'one','split':'train','scene':s},{'case_id':'two','split':'validation','scene':other}]}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'manifest.json';path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Repeated geometry'):verify_manifest(path)
    def test_zero_field_cannot_pass_an_active_traction_check(self):
        case=self.case('pipe');targets=choose_targets(case,np.random.default_rng(3));scales=torch.tensor([1.,1e-5,1e-4],dtype=torch.float64)
        def zero_model(c,ids,times):return torch.zeros((len(ids),3,3),dtype=torch.float64)
        a,b,g,_=physics_loss(zero_model,case,case['scene']['source']['delay_us']+1.,scales,targets)
        self.assertEqual(float(a),0.);self.assertGreater(float(b),0.);self.assertEqual(float(g),0.)

if __name__=='__main__':unittest.main()
