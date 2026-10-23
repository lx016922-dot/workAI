"""Real shared-network optimization with SHA-bound, admitted spectral labels."""
import argparse,json,math,shutil,time
from pathlib import Path
import numpy as np
import torch
from .data import load_dataset,digest
from .physics import physics_loss,choose_targets
from operator3d.propagation_model import PropagationSeparatedField
from operator3d.train import atomic,now,design,predict_case
from operator3d.physics import U0
from spectral3d.compare import compare,complex_coefficient
from spectral3d.field_compare import weighted


def tensor(v):return torch.as_tensor(v,dtype=torch.float64)


def validation(model,cases,designs,scales,targets):
    errors=[];pdes=[];boundaries=[]
    with torch.no_grad():
        for c in cases:
            ids,frames=designs[c['case_id']];pred=model(c,torch.as_tensor(ids),tensor(c['times'][frames]));truth=tensor(c['parts'][frames,ids]);errors.append(float(((pred-truth)/scales[None,:,None]).square().mean()))
            a,b,_,_=physics_loss(model,c,c['scene']['acquisition']['duration_us']*.53,scales,targets[c['case_id']]);pdes.append(float(a));boundaries.append(float(b))
    return {'normalized_mse':float(np.mean(errors)),'interior':float(np.mean(pdes)),'boundary':float(np.mean(boundaries))}


def evaluate(model,cases,scales,out,seed):
    rows=[];probes={};rng=np.random.default_rng(seed)
    for c in cases:
        pred=predict_case(model,c)*U0;truth=c['parts']*U0;mass=c['mass'].numpy();checks={}
        for name,a,b in [('full',pred.sum(2),truth.sum(2)),('material',pred[:,:,2],truth[:,:,2]),('geometric',pred[:,:,1],truth[:,:,1])]:
            checks[name+'_vector']=weighted(a,b,mass)
            for axis,label in enumerate('xyz'):checks[name+'_'+label]=weighted(a[:,:,axis:axis+1],b[:,:,axis:axis+1],mass)
        predicted=[];actual=[]
        for i,(ids,weights) in enumerate(c['receivers']):
            p=np.einsum('takc,a->tkc',pred[:,ids],weights);t=np.einsum('takc,a->tkc',truth[:,ids],weights);predicted.append(p);actual.append(t)
            probes[c['case_id']+f'_P{i+1}_prediction']=p;probes[c['case_id']+f'_P{i+1}_fem']=t
        probes[c['case_id']+'_times_us']=c['times'];common={'scene':c['scene'],'times':c['times'],'diagnostics':[]}
        trace=compare({c['case_id']:{**common,'parts':np.stack(predicted,1)}},{c['case_id']:{**common,'parts':np.stack(actual,1)}})['cases'][0]
        for i,(p,receiver) in enumerate(zip(trace['probes'],c['scene']['receivers'])):
            direction=np.asarray(receiver['direction']);direction/=np.linalg.norm(direction);pc=complex_coefficient(c['times'],predicted[i][:,2]@direction,2*c['scene']['source']['frequency_mhz']);tc=complex_coefficient(c['times'],actual[i][:,2]@direction,2*c['scene']['source']['frequency_mhz'])
            p.update(direction=receiver['direction'],full_waveform_relative_l2=p['checks']['full_requested']['relative_l2'],material_waveform_relative_l2=p['checks']['material_requested']['relative_l2'],material_2f_predicted_nm=float(abs(pc)*1e9),material_2f_fem_nm=float(abs(tc)*1e9),material_2f_complex_relative_error=float(abs(pc-tc)/abs(tc)) if abs(tc)>1e-13 else None)
        physics=[]
        with torch.no_grad():
            for _ in range(5):
                t=float(rng.uniform(.05,c['times'][-1]-.05));targets=choose_targets(c,rng,count=5);a,b,_,guard=physics_loss(model,c,t,scales,targets)
                physics.append({'time_us':t,'targets':targets.tolist(),'interior_rms':float(a.sqrt()),'boundary_rms':float(b.sqrt()),'max_strain':float(guard['max_strain']),'min_J':float(guard['min_J'])})
        prediction_sha=None
        if c['split']=='test':
            prediction_file=out/(c['case_id']+'-prediction.npz');np.savez_compressed(prediction_file,parts_m=pred,times_us=c['times']);prediction_sha=digest(prediction_file)
        row={'case_id':c['case_id'],'split':c['split'],'geometry':c['scene']['geometry'],'source':c['scene']['source'],'boundary':c['scene']['boundary'],'field_checks':checks,'probes':trace['probes'],'probe_checks_pass':trace['all_probe_checks_pass'],'sampled_physics':physics,'prediction_archive_sha256':prediction_sha};rows.append(row)
        print(json.dumps({'evaluated':c['case_id'],'full':checks['full_vector']['relative_l2'],'material':checks['material_vector']['relative_l2']}),flush=True)
    np.savez_compressed(out/'probe-predictions.npz',**probes);test=[r for r in rows if r['split']=='test']
    gates={'independent_test_present':len(test)==4,'held_out_fields':bool(test) and all(v['pass'] for r in test for v in r['field_checks'].values()),'held_out_receivers_and_2f':bool(test) and all(r['probe_checks_pass'] for r in test),'sampled_weak_physics':bool(test) and all(max(p['interior_rms'],p['boundary_rms'])<=.05 for r in test for p in r['sampled_physics']),'strain_and_positive_J':bool(test) and all(p['max_strain']<=.001 and p['min_J']>0 for r in test for p in r['sampled_physics']),'reference_h_p_time_and_fields':True}
    return {'schema':'workai.spectral_operator_metrics.v1','cases':rows,'probe_archive_sha256':digest(out/'probe-predictions.npz'),'acceptance':{'threshold':.05,**gates,'all_gates_passed':all(gates.values())},'test_used_for_checkpoint_selection':False,'accepted_for_solver_routing':False,'scope':'Four new test geometries in a bounded parameter range; sampled weak balance is not a continuous physics proof. Full-record 2f includes reflections.'}


def run(args):
    torch.set_num_threads(2);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed);out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.monotonic();source=Path(__file__).resolve().parents[1];hashes={}
    for module in ['solid','spectral3d','operator3d','spectralop']:
        dest=out/'source_snapshot'/module;dest.mkdir(parents=True)
        for p in (source/module).glob('*.py'):shutil.copyfile(p,dest/p.name);hashes[module+'/'+p.name]=digest(p)
    header=json.loads(args.dataset.read_text());latest=out.parents[1]/'latest.json';status={'schema':'workai.spectral_operator_training.v1','run_id':out.name,'status':'preparing','started_at_utc':now(),'step':0,'steps_total':args.steps,'physical_case_count':len(header['cases']),'split_counts':header['split_counts'],'accepted_for_solver_routing':False}
    def publish(**values):
        status.update(values,updated_at_utc=now(),elapsed_s=time.monotonic()-started);atomic(out/'status.json',status);atomic(latest,status)
    publish()
    try:
        cases,scale,description=load_dataset(args.dataset,admission_path=args.admission);scales=tensor(scale);atomic(out/'dataset.json',description);dataset_sha=digest(out/'dataset.json')
        train=[c for c in cases if c['split']=='train'];valid=[c for c in cases if c['split']=='validation'];vrng=np.random.default_rng(args.seed+1);designs={c['case_id']:design(c,vrng,nodes=160,frames=23) for c in valid};targets={c['case_id']:choose_targets(c,vrng,count=5) for c in valid}
        atomic(out/'validation-design.json',{c['case_id']:{'node_indices':designs[c['case_id']][0].tolist(),'frame_indices':designs[c['case_id']][1].tolist(),'physics_targets':targets[c['case_id']].tolist(),'physics_time_us':c['scene']['acquisition']['duration_us']*.53} for c in valid})
        model=PropagationSeparatedField(scale,args.width);optimizer=torch.optim.Adam(model.parameters(),lr=args.lr);best=float('inf');config={'steps':args.steps,'width':args.width,'seed':args.seed,'learning_rate':args.lr,'physics_weight':args.physics_weight,'physics_time_difference_us':.025,'physics_sampling':'Complete incident-element stars at 3 interior, 1 loaded-boundary and 1 traction-free node per step; target nodes selected independently of labels.','dtype':'float64','threads':2,'dataset_sha256':dataset_sha,'source_sha256':hashes,'checkpoint_selection':'Validation normalized MSE + fixed weight times validation interior and boundary weak residual. Test labels loaded only after checkpoint freeze.','constraints':['Objective Murnaghan Piola stress','High-order GLL weak momentum and integrated natural traction','Exact zero displacement and velocity before source delay','Exact fixed-boundary displacement','Strain <= .001 and positive J sampled guard']};atomic(out/'config.json',config)
        baseline=validation(model,valid,designs,scales,targets);atomic(out/'initial-validation.json',baseline);publish(status='training',validation=baseline,split_counts=description['split_counts'],physical_case_count=len(description['cases'])+4,trainable_parameters=sum(p.numel() for p in model.parameters()),constraints=config['constraints'])
        for step in range(1,args.steps+1):
            losses=[]
            for index in rng.choice(len(train),size=2,replace=False):
                c=train[int(index)];mass=c['mass'].numpy();ids=rng.choice(len(mass),size=384,p=mass/mass.sum());frames=rng.integers(1,len(c['times']),size=len(ids));prediction=model(c,torch.as_tensor(ids),tensor(c['times'][frames]));truth=tensor(c['parts'][frames,ids]);losses.append(((prediction-truth)/scales[None,:,None]).square().mean())
            data=torch.stack(losses).mean();a,b,guard,_=physics_loss(model,c,float(rng.uniform(.05,c['times'][-1]-.05)),scales,choose_targets(c,rng));weight=args.physics_weight*min(1.,max(0.,(step-400)/600));loss=data+weight*(a+b)+10*guard
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss; routing stays disabled.')
            for group in optimizer.param_groups:group['lr']=args.lr*(.15+.85*(1+math.cos(math.pi*step/args.steps))/2)
            optimizer.zero_grad(set_to_none=True);loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            if not torch.isfinite(gradient):raise ValueError('Nonfinite gradient.')
            optimizer.step()
            if step==1 or step%100==0 or step==args.steps:
                val=validation(model,valid,designs,scales,targets);score=val['normalized_mse']+args.physics_weight*(val['interior']+val['boundary']);improved=step>=1000 and score<best
                if improved:best=score
                checkpoint={'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'step':step,'best_score':best,'dataset_sha256':dataset_sha,'scales':scales,'width':args.width,'numpy_rng_state':rng.bit_generator.state,'torch_rng_state':torch.get_rng_state()};torch.save(checkpoint,out/'last.pt')
                if improved:torch.save(checkpoint,out/'best.pt')
                row={'step':step,'elapsed_s':time.monotonic()-started,'data':float(data.detach()),'interior':float(a.detach()),'boundary':float(b.detach()),'guard':float(guard.detach()),'weight':weight,'validation':val,'selected':improved}
                with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                publish(step=step,validation=val);print(json.dumps(row),flush=True)
        frozen=digest(out/'best.pt');saved=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['model_state']);atomic(out/'checkpoint-freeze.json',{'checkpoint_sha256':frozen,'selected_step':saved['step'],'frozen_at_utc':now(),'test_fields_loaded':False});publish(status='evaluating',checkpoint_sha256=frozen,selected_step=saved['step'])
        # Reload independently and compare exact predictions before opening test labels.
        reload=PropagationSeparatedField(scale,args.width);reload.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=True)['model_state']);c=valid[0];ids=torch.arange(min(127,len(c['points'])));times=torch.full((len(ids),),3.2,dtype=torch.float64)
        with torch.no_grad():torch.testing.assert_close(model(c,ids,times),reload(c,ids,times),rtol=0,atol=0)
        allcases,checkscale,record=load_dataset(args.dataset,splits=('train','validation','test'),admission_path=args.admission);np.testing.assert_array_equal(scale,checkscale);atomic(out/'evaluation-dataset.json',record)
        metrics=evaluate(model,[c for c in allcases if c['split']!='train'],scales,out,args.seed+2);assert digest(out/'best.pt')==frozen
        metrics.update(checkpoint_sha256=frozen,selected_step=saved['step'],new_training_steps=args.steps,completed_at_utc=now(),checkpoint_reload_exact=True);atomic(out/'metrics.json',metrics);publish(status='completed_experimental',metrics=metrics)
    except BaseException as error:publish(status='failed',error=str(error));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--admission',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=8000);p.add_argument('--width',type=int,default=96);p.add_argument('--lr',type=float,default=.0015);p.add_argument('--physics-weight',type=float,default=.002);p.add_argument('--seed',type=int,default=2026102317);a=p.parse_args()
    if not 1000<=a.steps<=12000 or not 32<=a.width<=128 or not 0<a.physics_weight<=.1:p.error('Use bounded experimental settings.')
    run(a)
