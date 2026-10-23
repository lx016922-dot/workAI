"""Diagnose existing development evidence without loading a new test set."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from operator3d.data import load_dataset
from operator3d.physics import weak_residual

def main(args):
    torch.set_num_threads(2);out=args.output;out.mkdir(parents=True,exist_ok=False)
    cases,scale,description=load_dataset(args.root,args.campaign,include_splits=['train','validation'],require_test=False)
    old=json.loads((args.run/'metrics.json').read_text());saved=np.load(args.run/'probe-predictions.npz',allow_pickle=False);rows=[]
    for c in cases:
        dt=c['times'][1]-c['times'][0];scales=torch.tensor(scale,dtype=torch.float64)[:,None]*(2*torch.pi*c['scene']['source']['frequency_mhz'])**2;residuals=[]
        for k in np.linspace(1,len(c['times'])-2,9,dtype=int):
            before,current,after=[torch.tensor(c['parts'][j],dtype=torch.float64) for j in [k-1,k,k+1]]
            residual,_,_=weak_residual(before,current,after,torch.tensor(c['times'][k],dtype=torch.float64),dt,c);free=~c['fixed'];squared=(residual/scales)**2;mass=c['mass'][free]
            rms=torch.sqrt((squared[free]*mass[:,None,None]).sum((0,2))/(mass.sum()*3));residuals.append(rms.numpy())
        prefix=c['case_id']+'_P1';pred=saved[prefix+'_prediction'].sum(1)[:,0];truth=saved[prefix+'_fem'].sum(1)[:,0];keep=slice(40,len(truth)-40);loss=lambda p,t:float(np.linalg.norm(p-t)/max(np.linalg.norm(t),1e-30));lags=range(-35,36)
        candidate=[loss(pred[np.arange(40,len(truth)-40)+lag],truth[keep]) for lag in lags];best=int(np.argmin(candidate));record=next(x for x in old['cases'] if x['case_id']==c['case_id'])
        rows.append({'case_id':c['case_id'],'current_development_split':c['split'],'old_v2_split':record['split'],'fem_weak_residual_dt_us':float(dt),'fem_normalized_residual_rms_by_branch_mean':np.mean(residuals,axis=0).tolist(),'fem_normalized_residual_rms_by_branch_max':np.max(residuals,axis=0).tolist(),'p1_x_cropped_original_error':loss(pred[keep],truth[keep]),'p1_x_best_shift_error':candidate[best],'best_shift_us':list(lags)[best]*dt})
    result={'scope':'Existing V2 cases only; new final tests not read. Best-shift errors are development diagnostics, never acceptance metrics. FEM residual uses stored 0.05 us samples, not the neural 0.025 us difference.','normalization_training_cases':description['split_counts']['train'],'rows':rows};(out/'diagnosis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'cases':len(rows),'fem_residual_branch_median':np.median([r['fem_normalized_residual_rms_by_branch_mean'] for r in rows],axis=0).tolist(),'note':result['scope']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/workspace/workai-remote'));p.add_argument('--campaign',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
