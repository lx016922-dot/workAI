import copy,gzip,hashlib,itertools,json,tempfile,unittest,uuid
from pathlib import Path
import numpy as np
import torch
from solid.mesh import prepare_mesh
from solid.schema import MATERIAL,default_scene
from .campaign import case_plan
from .expand_campaign import expand
from .data import check_splits,tensors,load_dataset
from .model import GeometryConditionedField
from .physics import piola,internal,weak_residual,physics_loss,U0
from .propagation_model import PropagationConditionedField,PropagationSeparatedField,setup_features,query_features
from .expand_independent import expand as expand_independent

def cube_case(fixed=False):
    points=np.array([[i&1,(i>>1)&1,(i>>2)&1] for i in range(8)],dtype=float)*.001
    tetra=np.array([[0,1<<a,(1<<a)+(1<<b),7] for a,b,c in itertools.permutations(range(3))]);mesh=prepare_mesh(points,tetra);scene=default_scene();scene['geometry'].update(length_mm=1.,width_mm=1.,thickness_mm=1.)
    scene['source']['center_mm']=[0.,.5,.5]
    if fixed:scene['boundary']['fixed_face']='xmax'
    return tensors(scene,mesh)

class OperatorTests(unittest.TestCase):
    def test_separate_heads_train_all_weak_branches_with_finite_physics_gradients(self):
        case=cube_case(True);model=PropagationSeparatedField([1.,1e-4,1e-4],32);indices=torch.arange(8);times=torch.linspace(.5,3,8,dtype=torch.float64);u=model(case,indices,times)
        self.assertTrue((u[case['fixed']]==0).all());loss=((u/model.scales[None,:,None])**2).mean()+.001*sum(physics_loss(model,case,2.,model.scales)[:3]);loss.backward()
        for head in model.heads:self.assertTrue(torch.isfinite(head.weight.grad).all());self.assertGreater(float(head.weight.grad.abs().sum()),0.)
        t=torch.zeros(8,dtype=torch.float64,requires_grad=True);initial=model(case,indices,t);velocity=torch.autograd.grad(initial.sum(),t)[0];self.assertTrue((initial==0).all());self.assertTrue((velocity==0).all())

    def test_source_aligned_model_keeps_initial_fixed_and_mesh_inputs_intact(self):
        torch.manual_seed(18);case=cube_case(True);mass=case['mass'].clone();surface=case['surface_features'].clone();model=PropagationConditionedField([1.,1e-4,1e-4],32);indices=torch.arange(8)
        case['scene']['source']['delay_us']=.3;t=torch.full((8,),.3,dtype=torch.float64,requires_grad=True);u=model(case,indices,t);v=torch.autograd.grad(u.sum(),t)[0]
        torch.testing.assert_close(u,torch.zeros_like(u),rtol=0,atol=0);torch.testing.assert_close(v,torch.zeros_like(v),rtol=0,atol=0)
        t=torch.linspace(1,4,8,dtype=torch.float64);u=model(case,indices,t);self.assertTrue((u[case['fixed']]==0).all());self.assertTrue((u[~case['fixed']].abs().sum((0,1))>0).all())
        perm=torch.tensor([7,5,3,1,6,4,2,0]);torch.testing.assert_close(model(case,indices[perm],t[perm]),u[perm]);torch.testing.assert_close(case['mass'],mass);torch.testing.assert_close(case['surface_features'],surface)
        before=query_features(case,indices,t);case2=cube_case(True);case2['scene']['source'].update(delay_us=.3,frequency_mhz=.23);self.assertFalse(torch.allclose(before,query_features(case2,indices,t)))
        losses=physics_loss(model,case,3.,model.scales);sum(losses[:3]).backward();self.assertGreater(float(model.geometry[0].weight.grad.abs().sum()),0);self.assertTrue(torch.isfinite(model.input.weight.grad).all())

    def test_independent_plan_decouples_end_and_fixity_without_reusing_test_geometry(self):
        first={'schema':'workai.operator3d.campaign.v1','complete':True,'cases':case_plan(),'case_count':16,'split_counts':{'train':8,'validation':4,'test':4},'discretization_verified':False};second=expand(first);second['complete']=True;third=expand_independent(second);third['complete']=True
        self.assertEqual(check_splits(third),{'train':40,'validation':8,'test':8})
        for family in ['plate','pipe','hole','notch']:
            added=[c for c in third['cases'][32:] if c['scene']['geometry']['kind']==family and c['split']=='train'];self.assertEqual({(c['scene']['source']['face'],c['scene']['boundary']['fixed_face']!='none') for c in added},{(a,b) for a in ['xmin','xmax'] for b in [False,True]})
        self.assertTrue(all(c['split']=='train' for c in third['cases'][:32] if c['previous_v2_split']=='test'))
        self.assertEqual([c['geometry_group_sha256'] for c in second['cases'] if c['split']=='validation'],[c['geometry_group_sha256'] for c in third['cases'] if c['split']=='validation'])

    def test_training_loader_does_not_open_withheld_labels_or_use_their_scale(self):
        from .campaign import geometry_identity
        from .train import final_evaluation
        sha=lambda b:hashlib.sha256(b).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);campaign=root/'campaign';campaign.mkdir();cases=[];base=cube_case();tetra=base['tetra'].numpy();points=base['points'].numpy()*.01
            for k,split in enumerate(['train','validation','test']):
                scene=copy.deepcopy(base['scene']);scene['geometry']['length_mm']=k+1.;scene['receivers']=[{'name':'P','point_mm':[.25,.5,.5],'direction':[1.,0.,0.]}];controls=[]
                for role,law in [('linear','linear'),('geometric','stvk'),('material','murnaghan')]:
                    jid='solid-'+uuid.uuid4().hex;controls.append({'role':role,'job_id':jid,'result_sha256':'unread-test'})
                    if split=='test':continue # No test job directory exists; accessing it would fail.
                    p=root/'solid/jobs'/jid;p.mkdir(parents=True);s=copy.deepcopy(scene);s['constitutive']=law;scene_bytes=json.dumps(s).encode();(p/'scene.json').write_bytes(scene_bytes)
                    np.savez(p/'mesh.npz',points=points,tetra=tetra);raw=np.full((3,8,3),1e-7 if split=='train' else 10.,dtype='<f8').tobytes();packed=gzip.compress(raw);(p/'field.bin.gz').write_bytes(packed)
                    result={'scene':s,'job_id':jid,'scene_sha256':sha(scene_bytes),'source_sha256':{},'field':{'sha256':sha(packed),'compressed_bytes':len(packed),'dtype':'little_endian_float64','uncompressed_sha256':sha(raw),'uncompressed_bytes':len(raw),'shape':[3,8,3]},'nodes_mm':(points*1000).tolist(),'tetra':tetra.tolist(),'times_us':[0.,.05,.1],'diagnostics':{'internal_step_us':.005},'mesh_sha256':sha((p/'mesh.npz').read_bytes())}
                    rb=json.dumps(result).encode();(p/'result.json').write_bytes(rb);controls[-1]['result_sha256']=sha(rb);(p/'status.json').write_text(json.dumps({'phase':'completed','result_sha256':sha(rb)}))
                cases.append({'case_id':split,'split':split,'geometry_group_sha256':geometry_identity(scene),'scene':scene,'controls':controls})
            m={'schema':'workai.operator3d.campaign.v1','complete':True,'case_count':3,'split_counts':{'train':1,'validation':1,'test':1},'cases':cases,'discretization_verified':False,'scope':'fixture'};path=campaign/'manifest.json';path.write_text(json.dumps(m))
            loaded,scales,record=load_dataset(root,path,include_splits=['train','validation']);self.assertEqual([c['case_id'] for c in loaded],['train','validation']);np.testing.assert_allclose(scales,[1.,5e-7,5e-7]);self.assertFalse(record['final_test_fields_loaded'])
            with self.assertRaises(FileNotFoundError):load_dataset(root,path)
            metrics=final_evaluation(None,[],None,root,1);self.assertFalse(metrics['acceptance']['all_gates_passed']);self.assertFalse(metrics['final_test_evaluated'])

    def test_murnaghan_stress_is_independent_energy_gradient_and_objective(self):
        torch.manual_seed(1);g=(torch.randn(4,3,3,dtype=torch.float64)*1e-4).requires_grad_();f=torch.eye(3)+g;e=(f.transpose(-1,-2)@f-torch.eye(3))*.5;tr=torch.diagonal(e,dim1=-2,dim2=-1).sum(-1);e2=e@e
        a=MATERIAL['n_pa'];b=MATERIAL['m_pa']-a/2;c=MATERIAL['l_pa']-MATERIAL['m_pa']+a/2
        w=MATERIAL['lambda_pa']/2*tr**2+MATERIAL['mu_pa']*torch.diagonal(e2,dim1=-2,dim2=-1).sum(-1)+a/3*torch.diagonal(e2@e,dim1=-2,dim2=-1).sum(-1)+b*tr*torch.diagonal(e2,dim1=-2,dim2=-1).sum(-1)+c/3*tr**3
        gradient=torch.autograd.grad(w.sum(),g)[0];p,_,_=piola(g,MATERIAL,'murnaghan');torch.testing.assert_close(p,gradient,rtol=1e-10,atol=1e-4)
        angle=.43;rotation=torch.tensor([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]],dtype=torch.float64)
        rotated,_,_=piola(rotation@f-torch.eye(3),MATERIAL,'murnaghan');torch.testing.assert_close(rotated,rotation@p,rtol=1e-8,atol=1e-4)

    def test_affine_tetra_force_equals_surface_traction_and_mass_is_physical(self):
        case=cube_case();self.assertAlmostEqual(case['mass'].sum().item(),2700e-9,places=15)
        parts=torch.zeros((8,3,3),dtype=torch.float64);parts[:,0,0]=case['points'][:,0]*.01*1e-5/U0
        forces,_,_=internal(parts,case);torch.testing.assert_close(forces.sum(0),torch.zeros((3,3),dtype=torch.float64),atol=1e-12,rtol=0)
        # Constant strain sigma_xx=(lambda+2mu)*epsilon; resultant right-face force.
        right=case['points'][:,0]>.05
        expected=(MATERIAL['lambda_pa']+2*MATERIAL['mu_pa'])*1e-5*1e-6
        self.assertAlmostEqual(forces[right,0,0].sum().item(),expected,places=12)
        parts[:,1:]=0
        # Exact quadratic-in-time uniform translation has zero internal force.
        current=torch.zeros_like(parts);before=torch.ones_like(parts)*.5*.025**2;after=before.clone();case['load']=torch.zeros_like(case['load'])
        residual,_,_=weak_residual(before,current,after,torch.tensor(0.),.025,case)
        torch.testing.assert_close(residual,torch.ones_like(parts),atol=1e-12,rtol=0)

    def test_hard_initial_and_fixed_conditions_allow_all_three_components(self):
        torch.manual_seed(2);case=cube_case(True);model=GeometryConditionedField([1.,1e-4,1e-4],width=16);indices=torch.arange(8);time=torch.zeros(8,dtype=torch.float64,requires_grad=True);u=model(case,indices,time)
        torch.testing.assert_close(u,torch.zeros_like(u),rtol=0,atol=0);v=torch.autograd.grad(u.sum(),time)[0];torch.testing.assert_close(v,torch.zeros_like(v),rtol=0,atol=0)
        u=model(case,indices,torch.ones_like(time));torch.testing.assert_close(u[case['fixed']],torch.zeros_like(u[case['fixed']]),rtol=0,atol=0);self.assertTrue((u[~case['fixed'],:,2].abs()>0).any())
        perm=torch.tensor([2,0,3,1,7,6,5,4]);torch.testing.assert_close(model(case,perm,torch.ones_like(time)),u[perm],rtol=1e-12,atol=1e-16)

    def test_whole_geometry_split_rejects_repeats_and_mislabeling(self):
        manifest={'schema':'workai.operator3d.campaign.v1','complete':True,'cases':case_plan(),'split_counts':{'train':8,'validation':4,'test':4}}
        self.assertEqual(check_splits(manifest),manifest['split_counts']);bad=copy.deepcopy(manifest);bad['cases'][1]['scene']['geometry']=copy.deepcopy(bad['cases'][0]['scene']['geometry']);bad['cases'][1]['geometry_group_sha256']=bad['cases'][0]['geometry_group_sha256']
        with self.assertRaisesRegex(ValueError,'Geometry repeats'):check_splits(bad)
        bad=copy.deepcopy(manifest);bad['cases'][1]['scene']['geometry']['length_mm']+=.1
        with self.assertRaisesRegex(ValueError,'identity'):check_splits(bad)

    def test_physical_loss_backpropagates_to_geometry_encoder_and_decoder(self):
        torch.manual_seed(3);case=cube_case();model=GeometryConditionedField([1.,1e-4,1e-4],width=16);scales=torch.tensor([1.,1e-4,1e-4],dtype=torch.float64)
        interior,boundary,guard,_=physics_loss(model,case,2.,scales);loss=interior+boundary+guard;loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(boundary.detach()),0)
        for block in [model.geometry,model.decoder]:
            gradients=[p.grad for p in block.parameters()];self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients));self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0)

    def test_second_split_has_new_geometry_and_both_boundaries_in_every_family(self):
        original={'schema':'workai.operator3d.campaign.v1','complete':True,'cases':case_plan(),'split_counts':{'train':8,'validation':4,'test':4}}
        expanded=expand(original);expanded['complete']=True;self.assertEqual(check_splits(expanded),{'train':16,'validation':8,'test':8});self.assertEqual(original['cases'][1]['split'],'validation')
        for split in ['validation','test']:
            for family in ['plate','pipe','hole','notch']:
                cases=[c for c in expanded['cases'] if c['split']==split and c['scene']['geometry']['kind']==family];self.assertEqual(len(cases),2);self.assertEqual({c['scene']['boundary']['fixed_face'] for c in cases},{'none','xmin'})

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
