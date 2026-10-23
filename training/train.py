"""Bounded first training run; no remote calls or COMSOL submissions."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from .data import load_data
from .model import GuidedWaveField
from .physics import momentum, boundary_residual, derivative, L0, T0, U0, EPS

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def tensor(value,grad=False):return torch.tensor(value,dtype=torch.float64,requires_grad=grad)
def relative(pred,truth):
    den=float(np.linalg.norm(truth));return float(np.linalg.norm(pred-truth)/den) if den>1e-15 else None

def collocation(rng,n,scene):
    length=scene['geometry']['length_mm']*.001/L0;half=scene['geometry']['thickness_mm']*.0005/L0;duration=scene['acquisition']['duration_us']
    q=np.column_stack([rng.uniform(0,length,n),rng.uniform(-half,half,n),rng.uniform(0,duration,n)])
    # Half of points resolve the pulse corridor; the rest cover the whole domain.
    cl,ct=scene['material']['cl_m_s'],scene['material']['ct_m_s'];hint=2*ct*np.sqrt(1-ct**2/cl**2)
    count=n//2; collected=[]
    while len(collected)<count:
        x=rng.uniform(0,length);t=x*L0/hint/T0+rng.uniform(-2,16)
        if 0<t<duration:collected.append([x,rng.uniform(-half,half),t])
    q[:count]=collected;return q

def boundary_points(rng,n,scene):
    length=scene['geometry']['length_mm']*.001/L0;half=scene['geometry']['thickness_mm']*.0005/L0
    q=np.column_stack([rng.uniform(0,length,n),rng.uniform(-half,half,n),rng.uniform(0,scene['acquisition']['duration_us'],n)])
    side=np.arange(n)%4;normals=np.zeros((n,2))
    q[side==0,0]=0;normals[side==0,0]=-1
    q[side==1,0]=length;normals[side==1,0]=1
    q[side==2,1]=-half;normals[side==2,1]=-1
    q[side==3,1]=half;normals[side==3,1]=1
    return q,normals,(side==0).astype(float)

def evaluate_data(model,data):
    with torch.no_grad():
        pred=model(tensor(data['val_q'])).numpy();field=model(tensor(data['field_val_q'])).numpy()
    per={}
    for k,name in enumerate(['linear','geometric_increment','material_increment']):
        per[name]={axis:relative(pred[:,k,j],data['val_y'][:,k,j]) for j,axis in enumerate(['u','v'])}
    return {'normalized_mse':float(np.mean(((pred-data['val_y'])/data['scales'])**2)),
            'receiver_40mm_relative_l2':per,'late_field_relative_l2':relative(field,data['field_val_y'])}

def physics_losses(model,q,boundary,scene,pde_scale,traction_scale):
    r,strain,jac=momentum(model(q),q,scene)
    qb,normals,edge=boundary
    bc,bs,bj=boundary_residual(model(qb),qb,normals,edge,scene)
    pde=((r/pde_scale)**2).mean();traction=((bc/traction_scale)**2).mean()
    strain_size=torch.linalg.matrix_norm(strain,dim=(-2,-1));boundary_size=torch.linalg.matrix_norm(bs,dim=(-2,-1))
    guard=(torch.relu(strain_size/.001-1)**2).mean()+(torch.relu(boundary_size/.001-1)**2).mean()+(torch.relu(1e-6-jac)**2).mean()+(torch.relu(1e-6-bj)**2).mean()
    return pde,traction,guard

def final_metrics(model,data,rng,pde_scale,traction_scale):
    result=evaluate_data(model,data)
    with torch.no_grad():prediction=model.controls(tensor(data['scan_q'])).numpy()
    result['held_out_10MPa_relative_l2']={str(shot):{axis:relative(prediction[:,shot,j],data['raw_scan'][:,shot,j]) for j,axis in enumerate(['u','v'])} for shot in [3,4]}
    harmonics=[]
    for row in data['report']['harmonics']['rows']:
        rx=row['rx'];mask=data['scan_rx']==rx;time_us=data['scan_q'][mask,2]
        start,end=row['window_us'];w=np.where((time_us>=start)&(time_us<=end),.5*(1-np.cos(2*np.pi*(time_us-start)/(end-start))),0)
        signal=(prediction[mask,2,0]-prediction[mask,1,0])*U0*1e9
        phase=np.exp(-2j*np.pi*2*data['scene']['source']['frequency_mhz']*time_us)
        coefficient=2*np.sum(w*(signal-np.sum(w*signal)/np.sum(w))*phase)/np.sum(w)
        true=row['levels']['timecheck']['material_second'];truth=complex(true['re_nm'],true['im_nm'])
        harmonics.append({'x_mm':row['x_mm'],'label_use':'validation' if rx==1 else 'training',
                          'predicted_material_a2_nm':float(abs(coefficient)),'comsol_material_a2_nm':float(abs(truth)),
                          'relative_complex_error':float(abs(coefficient-truth)/abs(truth))})
    result['material_harmonics']=harmonics
    residuals=[];boundaries=[];strains=[];jacobians=[]
    for repeat in range(16):
        amplitude=1. if repeat<8 else 2.; factors=tensor([[amplitude],[amplitude**2],[amplitude**2]])
        q=tensor(collocation(rng,64,data['scene']),True);r,s,j=momentum(model(q)*factors,q,data['scene'])
        residuals.append((r/(pde_scale*factors)).detach().numpy());strains.append(float(torch.linalg.matrix_norm(s,dim=(-2,-1)).detach().max()));jacobians.append(float(j.detach().min()))
        b=boundary_points(rng,64,data['scene']);qb=tensor(b[0],True)
        r,s,j=boundary_residual(model(qb)*factors,qb,tensor(b[1]),tensor(b[2]),data['scene'],amplitude)
        boundaries.append((r/(traction_scale*factors)).detach().numpy());strains.append(float(torch.linalg.matrix_norm(s,dim=(-2,-1)).detach().max()));jacobians.append(float(j.detach().min()))
    residuals=np.concatenate(residuals);boundaries=np.concatenate(boundaries)
    result['sampled_pde_normalized_rms']=np.sqrt(np.mean(residuals**2,axis=0)).tolist()
    result['sampled_traction_normalized_rms']=np.sqrt(np.mean(boundaries**2,axis=0)).tolist()
    result['physics_check_amplitudes_mpa']=[5,10]
    result['sampled_max_green_strain']=max(strains);result['sampled_min_J']=min(jacobians)
    error40=harmonics[1]['relative_complex_error']
    waveform_ok=all(v is not None and v<=.05 for errors in result['receiver_40mm_relative_l2'].values() for v in errors.values())
    physics_ok=float(np.sqrt(np.mean(residuals**2,axis=0)).max())<=.05 and float(np.sqrt(np.mean(boundaries**2,axis=0)).max())<=.05
    result['acceptance']={'threshold':.05,'held_out_receiver_waveforms':waveform_ok,'held_out_material_2f_complex':error40<=.05,'sampled_physics_residuals':physics_ok,'strain_and_positive_J':max(strains)<=.001 and min(jacobians)>0}
    result['accepted_for_solver_routing']=False
    result['scope']='Experimental fixed-case field fit. No independent material/frequency cases; no general-purpose solver routing. A² scaling is imposed, not learned evidence.'
    return result,prediction

def run(args):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(args.seed);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(args.seed);validation_rng=np.random.default_rng(args.seed+1)
    data=load_data(args.remote_root);scene=data['scene'];out=Path(args.run_dir)
    out.mkdir(parents=True,exist_ok=False)
    latest=Path(args.remote_root)/'training/latest.json'
    model=GuidedWaveField(scene,data['scales'],args.width);optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    scales=tensor(data['scales']);omega=2*np.pi*scene['source']['frequency_mhz']
    pde_scale=scales*tensor([[omega**2],[4*omega**2],[4*omega**2]])
    base_traction=scene['source']['amplitude_pa']/(scene['material']['rho_kg_m3']*(L0/T0)**2*EPS)
    traction_scale=scales/scales[0,0]*base_traction
    source_files={str(p.relative_to(Path(__file__).parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
    config={'seed':args.seed,'steps':args.steps,'width':args.width,'learning_rate':args.lr,'dtype':'float64','device':'cpu','torch_version':torch.__version__,'numpy_version':np.__version__,'threads':2,
            'physics_weight_final':.15,'traction_weight_final':.15,'warmup_steps':min(400,args.steps//3),'source_files':source_files,
            'scales_displacement_over_100nm':data['scales'].tolist(),'pde_normalization':pde_scale.tolist(),'traction_normalization':traction_scale.tolist(),
            'constraints':['rho0*u_tt=Div_X(P)','E=(F.T*F-I)/2; P=F*dW/dE; Murnaghan l/m/n','left nominal traction; other faces free','hard u(0)=ut(0)=0','hard mirror parity and w=0','hard small-amplitude linear+A^2 response expansion','sampled Green strain <=0.001 and J>0'],
            'checkpoint_selection':'40 mm validation label MSE plus independent collocation residuals; 10 MPa controls excluded from selection',
            'no_new_comsol_jobs':True}
    atomic_json(out/'config.json',config);atomic_json(out/'dataset.json',data['manifest']);atomic_json(out/'scene.json',scene)
    started=time.time();started_at=now();best=float('inf');history=[]
    status={'schema':'workai.physics_training.v1','run_id':out.name,'run_dir':str(out),'status':'training','pid':os.getpid(),'started_at_utc':started_at,'steps_total':args.steps,'step':0,'physical_case_count':1,'constraints':config['constraints'],'accepted_for_solver_routing':False}
    def publish():atomic_json(out/'status.json',status);atomic_json(latest,status)
    publish();print(json.dumps(status,ensure_ascii=False),flush=True)
    fixed_q=collocation(validation_rng,32,scene);fixed_b=boundary_points(validation_rng,32,scene)
    try:
        for step in range(1,args.steps+1):
            warm=min(1,step/config['warmup_steps']);weight=.01+.14*warm
            lr=args.lr*(.15+.85*.5*(1+math.cos(math.pi*step/args.steps)))
            for group in optimizer.param_groups:group['lr']=lr
            batch=rng.choice(len(data['train_q']),size=min(320,len(data['train_q'])),replace=False)
            output=model(tensor(data['train_q'][batch]));data_loss=(((output-tensor(data['train_y'][batch]))/scales)**2).mean()
            q=tensor(collocation(rng,48,scene),True);b=boundary_points(rng,32,scene)
            pde,traction,guard=physics_losses(model,q,(tensor(b[0],True),tensor(b[1]),tensor(b[2])),scene,pde_scale,traction_scale)
            loss=data_loss+weight*pde+weight*traction+guard
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training objective; stopped without publishing a usable model.')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
            if step==1 or step%50==0 or step==args.steps:
                validation=evaluate_data(model,data)
                q=tensor(fixed_q,True);vp,vb,_=physics_losses(model,q,(tensor(fixed_b[0],True),tensor(fixed_b[1]),tensor(fixed_b[2])),scene,pde_scale,traction_scale)
                selection=validation['normalized_mse']+.15*float(vp.detach())+.15*float(vb.detach())
                row={'step':step,'elapsed_s':time.time()-started,'total':float(loss.detach()),'data':float(data_loss.detach()),'pde':float(pde.detach()),'traction':float(traction.detach()),'guard':float(guard.detach()),'physics_weight':weight,'validation_mse':validation['normalized_mse'],'validation_pde':float(vp.detach()),'validation_traction':float(vb.detach()),'selection_score':selection}
                history.append(row)
                with (out/'history.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                payload={'model_state':model.state_dict(),'step':step,'width':args.width,'scales':data['scales'].tolist(),'scene':scene,'optimizer_state':optimizer.state_dict(),'seed':args.seed}
                torch.save(payload,out/'last.pt')
                if step>=config['warmup_steps'] and selection<best:best=selection;torch.save(payload,out/'best.pt')
                status.update(step=step,updated_at_utc=now(),elapsed_s=row['elapsed_s'],losses=row,validation=validation,checkpoint='last.pt');publish()
                print(json.dumps(row),flush=True)
        saved=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['model_state'])
        metrics,prediction=final_metrics(model,data,np.random.default_rng(args.seed+2),pde_scale,traction_scale)
        metrics.update(selected_step=saved['step'],elapsed_s=time.time()-started,completed_at_utc=now(),checkpoint_sha256=hashlib.sha256((out/'best.pt').read_bytes()).hexdigest())
        atomic_json(out/'metrics.json',metrics)
        np.savez_compressed(out/'predictions.npz',q=data['scan_q'],predicted=prediction,comsol=data['raw_scan'],rx=data['scan_rx'])
        status.update(status='completed_experimental',step=args.steps,updated_at_utc=now(),elapsed_s=metrics['elapsed_s'],checkpoint='best.pt',metrics=metrics);publish()
        print(json.dumps({'training_complete':True,'run_dir':str(out),'selected_step':saved['step'],'acceptance':metrics['acceptance'],'material_harmonics':metrics['material_harmonics']},ensure_ascii=False),flush=True)
    except BaseException as error:
        status.update(status='failed',updated_at_utc=now(),error=str(error));publish();raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--remote-root',default='/workspace/workai-remote');parser.add_argument('--run-dir',required=True);parser.add_argument('--steps',type=int,default=1500);parser.add_argument('--width',type=int,default=48);parser.add_argument('--lr',type=float,default=.002);parser.add_argument('--seed',type=int,default=20261022)
    args=parser.parse_args()
    if not 30<=args.steps<=10000 or not 16<=args.width<=128:parser.error('Bounded pilot requires 30–10000 steps and width 16–128.')
    run(args)
