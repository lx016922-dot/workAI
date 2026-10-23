"""Fit the bounded energy layer to grouped, mass-weighted FEM increments."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from operator3d.train import atomic,now
from .energy import EnergyCoupling

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(2026102308);start=time.monotonic();manifest=json.loads(args.dataset.read_text());assert manifest['complete'] and manifest['split_counts']=={'train':48,'validation':8,'test':0};groups={}
    for c in manifest['cases']:
        if c['geometry_group_sha256'] in groups:raise ValueError('Duplicate geometry in energy fit.')
        groups[c['geometry_group_sha256']]=c['split']
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for package in ['modal3d','operator3d','solid']:
        (snapshot/package).mkdir()
        for p in (Path(__file__).resolve().parents[1]/package).glob('*.py'):shutil.copyfile(p,snapshot/package/p.name)
    data={}
    for split in ['train','validation']:
        grams=[];rhs=[]
        for row in manifest['cases']:
            if row['split']!=split:continue
            p=args.dataset.parent/row['file'];assert digest(p)==row['sha256']
            with np.load(p,allow_pickle=False) as values:grams.append(values['gram']);rhs.append(values['rhs'])
        data[split]=(torch.tensor(np.mean(grams,axis=0),dtype=torch.float64),torch.tensor(np.mean(rhs,axis=0),dtype=torch.float64))
    model=EnergyCoupling();optimizer=torch.optim.Adam(model.parameters(),lr=.02);best=float('inf');history=[]
    loss_for=lambda alpha,split:alpha@data[split][0]@alpha-2*data[split][1]@alpha+1
    config={'architecture':'objective_cubic_energy_layer_plus_physical_modal_propagator','trainable_parameters':3,'classical_geometry_preparation':True,'pure_neural_transient_operator':False,'modes':manifest['modes'],'step_us':manifest['step_us'],'static_correction':True,'steps':1000,'learning_rate':.02,'multiplier_bounds':[.9,1.1],'reference_prior_weight':.01,'dataset_sha256':digest(args.dataset),'seed':2026102308,'checkpoint_selection':'minimum original 8-geometry validation material-field relative MSE','scope':'Numerical energy-coupling calibration, not independently identified new material constants. Reference FEM and sourced material remain unchanged.','source_sha256':{p.name:digest(p) for p in (snapshot/'modal3d').glob('*.py')}};atomic(out/'config.json',config)
    status={'schema':'workai.modal_energy_training.v1','run_id':out.name,'kind':'modal_energy','status':'training','step':0,'steps_total':1000,'physical_case_count':56,'split_counts':{'train':48,'validation':8,'test':0},'accepted_for_solver_routing':False,'started_at_utc':now(),'constraints':['objective cubic energy','bounded material couplings','exact fixed nodes and zero initial state','physical modal propagation and residual flexibility']}
    def publish(**values):status.update(values,updated_at_utc=now(),elapsed_s=time.monotonic()-start);atomic(out/'status.json',status)
    publish()
    with torch.no_grad():baseline={s:float(loss_for(model(),s)) for s in data}
    atomic(out/'physical-baseline.json',baseline)
    for step in range(1,1001):
        alpha=model();data_loss=loss_for(alpha,'train');loss=data_loss+.01*((alpha-1)**2).mean();optimizer.zero_grad();loss.backward();optimizer.step()
        if step==1 or step%10==0:
            with torch.no_grad():a=model();train=float(loss_for(a,'train'));val=float(loss_for(a,'validation'))
            row={'step':step,'train_relative_mse':train,'validation_relative_mse':val,'multipliers':a.tolist()};history.append(row)
            checkpoint={'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'step':step,'multipliers':a.tolist(),'dataset_sha256':config['dataset_sha256']};torch.save(checkpoint,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
            if val<best:best=val;shutil.copyfile(out/'last.pt',out/'best.pt')
            publish(step=step,losses=row)
    chosen=torch.load(out/'best.pt',map_location='cpu',weights_only=True);atomic(out/'history.json',history);receipt={'schema':'workai.modal_energy_weights.v1','run_id':out.name,'selected_step':chosen['step'],'multipliers':chosen['multipliers'],'checkpoint_sha256':digest(out/'best.pt'),'dataset_sha256':config['dataset_sha256'],'frozen_at_utc':now(),'new_test_labels_loaded':False,'pure_neural_transient_operator':False,'classical_geometry_preparation':True,'accepted_for_solver_routing':False};atomic(out/'energy-layer.json',receipt);publish(status='trained_pending_new_test',step=1000,selected_step=chosen['step'],multipliers=chosen['multipliers']);print(json.dumps({'trained':True,'selected_step':chosen['step'],'multipliers':chosen['multipliers'],'baseline':baseline,'best_validation_relative_mse':best,'checkpoint_sha256':receipt['checkpoint_sha256']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
