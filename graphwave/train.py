"""Offline physics distillation of a shared recurrent graph network.

All labels are generated here; the predictor does not import this module or
the constitutive teacher. No fresh test trajectory is used in optimization.
"""
import argparse,json,math,time,shutil
from pathlib import Path
import numpy as np
import torch
from numpy.polynomial.legendre import Legendre
from .model import GraphWave,STRAIN_SCALE,PRESSURE_SCALE,mandel,symmetric,derivative_features,PAIRS,TRIPLES
from operator3d.physics import piola
from operator3d.train import atomic,now
from spectralop.data import digest


def run(args):
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed);started=time.monotonic();model=GraphWave();record=json.loads(args.development.read_text())
    # Use only a training scene to define this bounded, fixed aluminum law.
    training=next(c for c in record['cases'] if c['split']=='train')
    base=Path(training['base_manifest']);assert digest(base)==training['base_manifest_sha256']
    material=json.loads(base.read_text())['cases'][0]['scene']['material']
    source={};repo=Path(__file__).resolve().parents[1]
    for module in ['graphwave','operator3d','spectralop','spectral3d','solid']:
        dest=args.output/'source_snapshot'/module;dest.mkdir(parents=True)
        for path in (repo/module).glob('*.py'):
            shutil.copyfile(path,dest/path.name);source[module+'/'+path.name]=digest(path)
    atomic(args.output/'config.json',{'schema':'workai.graphwave_training.v1','architecture':'learned_graph_filters_conservative_polynomial_energy_recurrent_state',
        'trainable_parameters':sum(p.numel() for p in model.parameters()),'seed':args.seed,'development_manifest':str(args.development),'development_manifest_sha256':digest(args.development),
        'source_sha256':source,'material':material,'degrees':[3,4],
        'training_labels':'Offline polynomial derivative, objective material stress and discrete momentum state-update labels; transient development validation follows.',
        'no_final_test_fields_loaded':True,'inference_numerical_correction':False,
        'constraints':['Exact zero initial state and fixed nodes','Graph filter annihilates rigid translations','Transpose-tied energy messages','Objective Green strain','Shared full tensor energy network','No hidden solver or reference lookup'],
        'not_a_mesh_free_or_one_shot_operator':True,'accepted_for_solver_routing':False})
    z=torch.tensor(rng.normal(size=(2048,6)),dtype=torch.float64);z=z/torch.linalg.vector_norm(z,dim=1,keepdim=True)*torch.tensor(rng.uniform(.03,1,size=(2048,1)))
    zval=torch.tensor(rng.normal(size=(1024,6)),dtype=torch.float64);zval=zval/torch.linalg.vector_norm(zval,dim=1,keepdim=True)*torch.tensor(rng.uniform(.01,2,size=(1024,1)))
    def labels(values):
        e=symmetric(values*STRAIN_SCALE)
        # Linear stress is the second Piola stress for this strain. Compute
        # the quadratic material term independently to avoid cancellation.
        tr=e.diagonal(dim1=-2,dim2=-1).sum(-1);e2=e@e;eye=torch.eye(3,dtype=torch.float64)
        a=material['n_pa'];b=material['m_pa']-a/2;c=material['l_pa']-material['m_pa']+a/2
        linear=material['lambda_pa']*tr[:,None,None]*eye+2*material['mu_pa']*e
        nonlinear=(b*e2.diagonal(dim1=-2,dim2=-1).sum(-1)+c*tr**2)[:,None,None]*eye+2*b*tr[:,None,None]*e+a*e2
        return mandel(linear)/(PRESSURE_SCALE*STRAIN_SCALE),mandel(nonlinear)/(PRESSURE_SCALE*STRAIN_SCALE**2)
    y2,y3=labels(z);v2,v3=labels(zval)
    features2=derivative_features(z,PAIRS);features3=derivative_features(z,TRIPLES)
    vfeatures2=derivative_features(zval,PAIRS);vfeatures3=derivative_features(zval,TRIPLES)
    designs={}
    for p in [3,4]:
        poly=Legendre.basis(p);nodes=np.r_[-1.,poly.deriv().roots(),1.]
        values=np.stack([Legendre.basis(k)(nodes) for k in range(1,p+1)],1)
        target=np.stack([Legendre.basis(k).deriv()(nodes) for k in range(1,p+1)],1)
        designs[p]=(torch.tensor(values),torch.tensor(target))
    delta=torch.tensor(rng.normal(size=(4096,)),dtype=torch.float64);message=torch.tensor(rng.normal(size=(4096,)),dtype=torch.float64)
    def losses(validation=False):
        f2,f3,a,b=(vfeatures2,vfeatures3,v2,v3) if validation else (features2,features3,y2,y3)
        linear=((f2@model.quadratic-a)**2).mean();nonlinear=((f3@model.cubic-b)**2).mean()
        graph=sum(((model.derivative(p)@value-target)**2).mean()/target.square().mean() for p,(value,target) in designs.items())
        state=((model.state_weights[0]*delta+model.state_weights[1]*message-delta-message)**2).mean()
        return linear+nonlinear+graph+state,{'linear_mse':float(linear.detach()),'nonlinear_mse':float(nonlinear.detach()),'filter_mse':float(graph.detach()),'state_mse':float(state.detach())}
    initial=float(losses()[0]);optimizer=torch.optim.Adam(model.parameters(),lr=.03);history=[]
    for step in range(1,1201):
        optimizer.zero_grad(set_to_none=True);loss,_=losses();loss.backward();optimizer.step()
        if step==1 or step%100==0:
            value,metrics=losses(True);row={'stage':'adam','step':step,'loss':float(loss.detach()),'validation_loss':float(value.detach()),'elapsed_s':time.monotonic()-started,**metrics};history.append(row);print(json.dumps(row),flush=True)
    torch.save({'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'step':1200},args.output/'adam.pt')
    # Offline optimizer refinement is training, never an inference corrector.
    optimizer=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=600,tolerance_grad=1e-16,tolerance_change=1e-25,history_size=100,line_search_fn='strong_wolfe');calls=0
    def closure():
        nonlocal calls
        optimizer.zero_grad(set_to_none=True);loss,_=losses();loss.backward();calls+=1
        return loss
    optimizer.step(closure);value,metrics=losses(True)
    checkpoint={'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'optimizer':'LBFGS_after_Adam','adam_steps':1200,'lbfgs_function_evaluations':calls,'seed':args.seed,'torch_rng_state':torch.get_rng_state(),'numpy_rng_state':rng.bit_generator.state}
    torch.save(checkpoint,args.output/'candidate.pt');export=model.export()
    np.savez(args.output/'candidate-weights.npz',quadratic_tensor=export['quadratic_tensor'],cubic_tensor=export['cubic_tensor'],state_weights=export['state_weights'],**{'derivative_'+p:d for p,d in export['derivatives'].items()})
    atomic(args.output/'history.json',history)
    audit={'initial_loss':initial,'final_validation_loss':float(value.detach()),**metrics,'adam_steps':1200,'lbfgs_function_evaluations':calls,'elapsed_s':time.monotonic()-started,'candidate_checkpoint_sha256':digest(args.output/'candidate.pt'),'export_sha256':digest(args.output/'candidate-weights.npz'),'weights_frozen_for_development_validation_at_utc':now(),'final_test_fields_loaded':False,'accepted_for_solver_routing':False}
    atomic(args.output/'training.json',audit);print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--development',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=2026102401);run(p.parse_args())
