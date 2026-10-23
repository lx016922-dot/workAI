"""Explicit bounded CPU training of a shared geometry-conditioned 3-D field."""
import argparse,datetime,hashlib,json,math,shutil,time
from pathlib import Path
import numpy as np
import torch
from .data import load_dataset,digest
from .model import GeometryConditionedField,make_model
from .physics import physics_loss,U0

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2));temporary.replace(path)
def tensor(value):return torch.as_tensor(value,dtype=torch.float64)

def design(case,rng,nodes=128,frames=17):
    mass=case['mass'].numpy();indices=rng.choice(len(mass),size=min(nodes,len(mass)),replace=False,p=mass/mass.sum());time_index=np.unique(np.linspace(1,len(case['times'])-1,frames,dtype=int))
    return np.tile(indices,len(time_index)),np.repeat(time_index,len(indices))

def sample_score(model,cases,designs,scales,physics=True):
    data=[];pde=[];boundary=[]
    with torch.no_grad():
        for case in cases:
            nodes,frames=designs[case['case_id']];pred=model(case,torch.as_tensor(nodes),tensor(case['times'][frames]));truth=tensor(case['parts'][frames,nodes]);data.append(float((((pred-truth)/scales[None,:,None])**2).mean()))
            if physics:
                a,b,_,_=physics_loss(model,case,case['scene']['acquisition']['duration_us']*.53,scales);pde.append(float(a));boundary.append(float(b))
    return {'normalized_mse':float(np.mean(data)),'pde':float(np.mean(pde)) if pde else None,'traction':float(np.mean(boundary)) if boundary else None}

def predict_case(model,case):
    count=len(case['points']);total=len(case['times'])*count;result=np.empty((total,3,3),dtype=np.float64)
    with torch.no_grad():
        for start in range(0,total,4096):
            ids=np.arange(start,min(start+4096,total));result[start:start+len(ids)]=model(case,torch.as_tensor(ids%count),tensor(case['times'][ids//count])).numpy()
    return result.reshape(len(case['times']),count,3,3)

def relative(a,b,weight=None):
    error=(a-b)**2;reference=b*b
    if weight is not None:error=error*weight;reference=reference*weight
    norm=float(reference.sum());return float(np.sqrt(error.sum()/norm)) if norm>1e-28 else None

def coefficients(times,signal,frequency):
    w=.5*(1-np.cos(2*np.pi*(times-times[0])/(times[-1]-times[0])));mean=np.tensordot(w,signal,axes=(0,0))/w.sum()
    return 2*np.tensordot(w*np.exp(-2j*np.pi*frequency*times),signal-mean,axes=(0,0))/w.sum()

def final_evaluation(model,cases,scales,out,seed):
    rows=[];saved={};rng=np.random.default_rng(seed);physics=[]
    for case in cases:
        predicted=predict_case(model,case);truth=case['parts'];mass=case['mass'].numpy()[None,:,None]
        parts={name:relative(predicted[:,:,k],truth[:,:,k],mass) for k,name in enumerate(['linear','geometric_increment','material_increment'])}
        full=relative(predicted.sum(2),truth.sum(2),mass);probes=[]
        for i,((cell,weights),receiver) in enumerate(zip(case['receivers'],case['scene']['receivers'])):
            nodes=case['tetra'][cell].numpy();pv=np.einsum('takc,a->tkc',predicted[:,nodes],weights);tv=np.einsum('takc,a->tkc',truth[:,nodes],weights);direction=np.array(receiver['direction']);p=np.einsum('tkc,c->tk',pv,direction);t=np.einsum('tkc,c->tk',tv,direction)
            pc=coefficients(case['times'],p[:,2],2*case['scene']['source']['frequency_mhz']);tc=coefficients(case['times'],t[:,2],2*case['scene']['source']['frequency_mhz']);floor=1e-6 # 0.0001 nm in the 100 nm displacement scale.
            probes.append({'name':receiver['name'],'point_mm':receiver['point_mm'],'direction':receiver['direction'],'full_waveform_relative_l2':relative(p.sum(1),t.sum(1)),'material_waveform_relative_l2':relative(p[:,2],t[:,2]),'material_2f_predicted_nm':float(abs(pc)*U0*1e9),'material_2f_fem_nm':float(abs(tc)*U0*1e9),'material_2f_complex_relative_error':float(abs(pc-tc)/abs(tc)) if abs(tc)>floor else None,'material_2f_above_reporting_floor':bool(abs(tc)>floor),'window_us':[float(case['times'][0]),float(case['times'][-1])],'includes_reflections':True})
            saved[case['case_id']+'_P'+str(i+1)+'_prediction']=pv*U0;saved[case['case_id']+'_P'+str(i+1)+'_fem']=tv*U0
        saved[case['case_id']+'_times_us']=case['times']
        with torch.no_grad():
            checks=[]
            for _ in range(3):
                t=float(rng.uniform(.05,case['scene']['acquisition']['duration_us']-.05));a,b,_,guard=physics_loss(model,case,t,scales)
                checks.append({'interior_rms':float(a.sqrt()),'boundary_rms':float(b.sqrt()),'max_strain':float(guard['max_strain']),'min_J':float(guard['min_J'])})
        rows.append({'case_id':case['case_id'],'split':case['split'],'geometry':case['scene']['geometry'],'source':case['scene']['source'],'boundary':case['scene']['boundary'],'full_field_relative_l2':full,'parts_field_relative_l2':parts,'probes':probes,'sampled_physics':checks})
        print(json.dumps({'evaluated':case['case_id'],'split':case['split'],'full_field_relative_l2':full,'material_increment_relative_l2':parts['material_increment']}),flush=True)
    np.savez_compressed(out/'probe-predictions.npz',**saved)
    test=[r for r in rows if r['split']=='test'];threshold=.05
    gates={'held_out_full_fields':all(r['full_field_relative_l2'] is not None and r['full_field_relative_l2']<=threshold for r in test),'held_out_material_increment':all(r['parts_field_relative_l2']['material_increment'] is not None and r['parts_field_relative_l2']['material_increment']<=threshold for r in test),'held_out_material_2f':all(p['material_2f_complex_relative_error'] is not None and p['material_2f_complex_relative_error']<=threshold for r in test for p in r['probes']),'sampled_weak_physics':all(max(c['interior_rms'],c['boundary_rms'])<=threshold for r in test for c in r['sampled_physics']),'strain_and_positive_J':all(c['max_strain']<=.001 and c['min_J']>0 for r in test for c in r['sampled_physics']),'label_discretization_verified':False}
    if not test:gates={k:False for k in gates}
    return {'cases':rows,'acceptance':{'threshold':threshold,**gates,'all_gates_passed':bool(test) and all(gates.values())},'accepted_for_solver_routing':False,'test_used_for_checkpoint_selection':False,'final_test_evaluated':bool(test),'material_harmonic_scope':'Full-record Hann complex difference, including reflections; not a separated nonlinear guided mode.','physics_scope':'Sampled times, tetrahedral weak momentum and natural traction balance; not a continuous-domain proof.'}

def run(args):
    torch.set_num_threads(2);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed);out=Path(args.run_dir);root=Path(args.root);started=time.monotonic();package=Path(__file__).resolve().parent
    hashes={p.name:digest(p.read_bytes()) for p in package.glob('*.py')};prior=None
    if args.resume:
        config=json.loads((out/'config.json').read_text());assert config['source_sha256']==hashes and config['steps']==args.steps and config['seed']==args.seed and config['width']==args.width and config['learning_rate']==args.lr and config['physics_weight_final']==args.physics_weight and config['data_warmup_steps']==args.data_warmup and config['architecture']==args.architecture and config['development_only']==args.development_only and config['batch_cases']==args.batch_cases
        prior=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
    else:
        out.mkdir(parents=True,exist_ok=False);snapshot=out/'source_snapshot';snapshot.mkdir()
        for name in ['operator3d','solid']:
            (snapshot/name).mkdir()
            for p in (package.parent/name).glob('*.py'):shutil.copyfile(p,snapshot/name/p.name)
    latest=root/'training3d/latest.json';latest.parent.mkdir(parents=True,exist_ok=True)
    campaign_header=json.loads(Path(args.campaign).read_text())
    status={'schema':'workai.operator_training.v1','run_id':out.name,'kind':'mesh_conditioned_3d','status':'preparing','step':0,'steps_total':args.steps,'started_at_utc':now(),'physical_case_count':campaign_header['case_count'],'split_counts':campaign_header['split_counts'],'accepted_for_solver_routing':False,'constraints':['3-D objective Murnaghan constitutive law','tetrahedral weak momentum','integrated nominal traction/free boundary','hard fixed nodes and zero initial displacement/velocity'],'discretization_verified':False}
    def publish(**values):
        status.update(values,updated_at_utc=now(),elapsed_s=time.monotonic()-started+(prior.get('elapsed_s',0) if prior else 0));atomic(out/'status.json',status)
        if not args.development_only:atomic(latest,status)
    publish()
    try:
        cases,scale,description=load_dataset(root,args.campaign,include_splits=['train','validation'],require_test=not args.development_only);scales=tensor(scale);atomic(out/'dataset.json',description);dataset_sha=digest((out/'dataset.json').read_bytes())
        model=make_model(scale,args.width,args.architecture);optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
        training=[c for c in cases if c['split']=='train'];validation=[c for c in cases if c['split']=='validation'];vrng=np.random.default_rng(args.seed+1);designs={c['case_id']:design(c,vrng) for c in validation}
        config={'architecture':args.architecture,'development_only':args.development_only,'batch_cases':args.batch_cases,'steps':args.steps,'seed':args.seed,'width':args.width,'learning_rate':args.lr,'threads':2,'dtype':'float64','geometry_groups':description['split_counts'],'physics_weight_final':args.physics_weight,'data_warmup_steps':args.data_warmup,'physics_warmup_steps':400,'physics_time_difference_us':.025,'physics_sampling':'one full tetrahedral mesh at a random time per step; natural boundary and interior balances separated','data_sampling':'batch of physical cases per step, volume-weighted nodes and random recorded times','checkpoint_selection':'validation normalized field MSE + physics_weight_final*(validation interior + boundary weak residual); test fields are not loaded until after freezing the checkpoint','dataset_sha256':dataset_sha,'source_sha256':hashes,'scope':'Bounded cross-geometry aluminum pilot. No claim of arbitrary-shape accuracy.'}
        best=float('inf');start=1
        if prior:
            assert prior['dataset_sha256']==dataset_sha;model.load_state_dict(prior['model_state']);optimizer.load_state_dict(prior['optimizer_state']);rng.bit_generator.state=prior['numpy_rng_state'];torch.set_rng_state(prior['torch_rng_state']);best=prior['best_score'];start=prior['step']+1
        else:atomic(out/'config.json',config)
        baseline=sample_score(model,validation,designs,scales);publish(status='training',validation=baseline,step=start-1)
        if not prior:atomic(out/'initial-validation.json',baseline)
        for step in range(start,args.steps+1):
            data_losses=[]
            for index in rng.choice(len(training),size=args.batch_cases,replace=False):
                case=training[int(index)];mass=case['mass'].numpy();node=rng.choice(len(mass),size=max(192,768//args.batch_cases),p=mass/mass.sum());frame=rng.integers(1,len(case['times']),size=len(node))
                predicted=model(case,torch.as_tensor(node),tensor(case['times'][frame]));truth=tensor(case['parts'][frame,node]);data_losses.append((((predicted-truth)/scales[None,:,None])**2).mean())
            data_loss=torch.stack(data_losses).mean()
            pde,boundary,guard,_=physics_loss(model,case,float(rng.uniform(.05,case['scene']['acquisition']['duration_us']-.05)),scales)
            weight=args.physics_weight*max(0.,min(1.,(step-args.data_warmup)/400));loss=data_loss+weight*(pde+boundary)+10*guard
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss; checkpoint will not be enabled.')
            learning_rate=args.lr*(.2+.8*.5*(1+math.cos(math.pi*step/args.steps)))
            for group in optimizer.param_groups:group['lr']=learning_rate
            optimizer.zero_grad(set_to_none=True);loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            if not torch.isfinite(gradient):raise ValueError('Nonfinite gradient; training stopped.')
            optimizer.step()
            if step==1 or step%50==0 or step==args.steps:
                val=sample_score(model,validation,designs,scales);score=val['normalized_mse']+args.physics_weight*(val['pde']+val['traction'])
                improved=step>=max(100,args.data_warmup+400) and score<best
                if improved:best=score
                row={'step':step,'elapsed_s':time.monotonic()-started,'case_id':case['case_id'],'total':float(loss.detach()),'data':float(data_loss.detach()),'pde':float(pde.detach()),'traction':float(boundary.detach()),'guard':float(guard.detach()),'physics_weight':weight,'validation_mse':val['normalized_mse'],'validation_pde':val['pde'],'validation_traction':val['traction'],'selection_score':score}
                with (out/'history.jsonl').open('a') as file:file.write(json.dumps(row,allow_nan=False)+'\n')
                checkpoint={'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'architecture':args.architecture,'width':args.width,'scales':scale.tolist(),'step':step,'dataset_sha256':dataset_sha,'best_score':best,'numpy_rng_state':rng.bit_generator.state,'torch_rng_state':torch.get_rng_state(),'elapsed_s':time.monotonic()-started}
                torch.save(checkpoint,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
                if improved:shutil.copyfile(out/'last.pt',out/'best.pt')
                publish(step=step,losses=row,validation=val);print(json.dumps(row),flush=True)
        saved=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['model_state']);frozen_sha=digest((out/'best.pt').read_bytes());atomic(out/'frozen-checkpoint-before-test.json',{'step':saved['step'],'sha256':frozen_sha,'frozen_at_utc':now(),'test_labels_used_for_selection':False});publish(status='evaluating',step=args.steps)
        if args.development_only:
            evaluation=validation
        else:
            evaluation,check_scale,evaluation_description=load_dataset(root,args.campaign)
            assert evaluation_description['campaign_sha256']==description['campaign_sha256'];np.testing.assert_allclose(check_scale,scale,rtol=0,atol=0);atomic(out/'evaluation-dataset.json',evaluation_description)
        metrics=final_evaluation(model,evaluation,scales,out,args.seed+2);assert digest((out/'best.pt').read_bytes())==frozen_sha
        metrics.update(selected_step=saved['step'],checkpoint_sha256=frozen_sha,completed_at_utc=now(),elapsed_s=time.monotonic()-started,dataset_sha256=dataset_sha);atomic(out/'metrics.json',metrics);publish(status='completed_experimental',metrics=metrics)
        print(json.dumps({'completed':True,'run_id':out.name,'selected_step':saved['step'],'acceptance':metrics['acceptance']}),flush=True)
    except BaseException as error:publish(status='failed',error=str(error));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/workspace/workai-remote');p.add_argument('--campaign',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--steps',type=int,default=2000);p.add_argument('--width',type=int,default=64);p.add_argument('--lr',type=float,default=.0025);p.add_argument('--seed',type=int,default=20261023);p.add_argument('--physics-weight',type=float,default=.005);p.add_argument('--data-warmup',type=int,default=0);p.add_argument('--architecture',choices=['baseline','propagation','propagation_separate'],default='baseline');p.add_argument('--batch-cases',type=int,default=1);p.add_argument('--development-only',action='store_true');p.add_argument('--resume',action='store_true');args=p.parse_args()
    if not 400<=args.steps<=10000 or not 16<=args.width<=128 or not 0<=args.data_warmup<=args.steps-400 or not 0<args.physics_weight<=.1 or not 1<=args.batch_cases<=4:p.error('Use 400–10000 steps, bounded physics warmup, width 16–128 and 1–4 cases per batch for this CPU pilot.')
    run(args)
