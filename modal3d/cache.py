"""Freeze label-bound reduced responses for training an energy coupling layer."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
from solid.mesh import prepare_mesh
from operator3d.data import load_dataset
from operator3d.train import relative,atomic
from .basis import build_basis
from .response import response

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);snapshot=out/'source_snapshot/modal3d';snapshot.mkdir(parents=True)
    for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,snapshot/p.name)
    cases,_,labels=load_dataset(args.root,args.campaign,include_splits=['train','validation'],require_test=False);rows=[];started=time.monotonic()
    atomic(out/'status.json',{'phase':'preparing','cases_total':len(cases),'cases_completed':0})
    for c in cases:
        mesh=prepare_mesh(c['points'].numpy()*.01,c['tetra'].numpy());basis=build_basis(mesh,c['scene'],args.modes);pred=response(mesh,c['scene'],basis,.025,True);phi=basis['phi'].reshape(-1,3,args.modes);features=[]
        for k in range(3):features.append((np.einsum('tm,ncm->tnc',pred['modal_basis_responses'][:,k+2],phi)+pred['static_basis_responses'][k+1].reshape(len(c['times']),-1,3))*pred['material_coefficients'][k])
        x=np.stack(features,2);y=c['parts'][:,:,2]*1e-7;mass=c['mass'].numpy();norm=np.einsum('tni,n,tni->',y,mass,y);gram=np.einsum('tnic,n,tnjc->ij',x,mass,x,optimize=True)/norm;rhs=np.einsum('tnic,n,tnc->i',x,mass,y,optimize=True)/norm
        if not norm>0 or not np.isfinite(gram).all():raise ValueError('Nonfinite or zero-energy material labels.')
        path=out/(c['case_id']+'.npz');np.savez_compressed(path,material_features_m=x,base_parts_m=pred['parts_m'][:,:,:2],reference_parts_m=c['parts']*1e-7,mass_kg=mass,times_us=c['times'],tetra=c['tetra'].numpy(),points_mm=c['points'].numpy()*10,gram=gram,rhs=rhs)
        weight=mass[None,:,None];truth=c['parts']*1e-7;row={'case_id':c['case_id'],'split':c['split'],'geometry_group_sha256':c['group'],'scene':c['scene'],'file':path.name,'sha256':digest(path),'receivers':[{'cell':int(cell),'weights':weights.tolist()} for cell,weights in c['receivers']],'preparation_seconds':basis['preparation_seconds'],'response_seconds':pred['response_seconds'],'modes':args.modes,'max_frequency_mhz':float(basis['frequency_mhz'].max()),'max_eigen_residual':float(basis['eigen_residual'].max()),'baseline_full_relative_l2':relative(pred['parts_m'].sum(2),truth.sum(2),weight),'baseline_material_relative_l2':relative(pred['parts_m'][:,:,2],truth[:,:,2],weight)};rows.append(row);atomic(out/'status.json',{'phase':'preparing','cases_total':len(cases),'cases_completed':len(rows),'elapsed_s':time.monotonic()-started});print(json.dumps(row|{'scene':None,'receivers':None}),flush=True)
    manifest={'schema':'workai.modal_energy_dataset.v1','complete':True,'modes':args.modes,'step_us':.025,'static_correction':True,'case_count':len(rows),'split_counts':{s:sum(c['split']==s for c in rows) for s in ['train','validation','test']},'campaign_sha256':digest(args.campaign),'labels':labels,'source_sha256':{p.name:digest(p) for p in snapshot.glob('*.py')},'scope':'Three trainable energy coupling multipliers with physical modal propagation; not a pure neural transient operator.','cases':rows};atomic(out/'manifest.json',manifest);atomic(out/'status.json',{'phase':'completed','cases_completed':len(rows),'elapsed_s':time.monotonic()-started});print(json.dumps({'complete':True,'cases':len(rows),'seconds':time.monotonic()-started}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/workspace/workai-remote'));p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--modes',type=int,default=160);main(p.parse_args())
