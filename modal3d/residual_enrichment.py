"""Development-only correction directions from an input-only momentum residual."""
import time
import numpy as np
from scipy.linalg import eigh
from solid.mesh import nodal_load
from .response import pulse
from .nonlinear import stress,linear_stress

def residual_enrich(mesh,scene,basis,prediction,scales,count=48):
    started=time.monotonic();mass=basis['mass'];root=np.sqrt(mass);v=basis['phi']*root[:,None];load,_=nodal_load(mesh,scene['source']);columns=[];indices=np.unique(np.linspace(1,len(prediction['times_us'])-2,64,dtype=int))
    for i in indices:
        parts=prediction['parts_m'][i];g=np.einsum('eaki,eaj->ekij',parts[mesh['tetra']],mesh['grad']);cumulative=g.cumsum(1);piola=[linear_stress(cumulative[:,0],scene['material']),stress(cumulative[:,1],scene['material']),stress(cumulative[:,2],scene['material'],[1.,1.,1.])];forces=np.stack([basis['divergence']@p.reshape(-1) for p in piola],1);forces[:,2]-=forces[:,1];forces[:,1]-=forces[:,0];forces[:,0]-=load.ravel()*pulse(np.array(prediction['times_us'][i]),scene['source']);acc=prediction['acceleration_m_us2'][i].transpose(0,2,1).reshape(-1,3);residual=forces+mass[:,None]*acc*1e12;correction=basis['residual_flexibility'](residual);columns.append(correction/np.asarray(scales)[None,:])
    snapshots=np.concatenate(columns,axis=1)*root[:,None];snapshots-=v@(v.T@snapshots);force_snapshots=(basis['stiffness']@(snapshots/root[:,None]))/root[:,None];_,s,right=np.linalg.svd(force_snapshots,full_matrices=False);rank=min(count,int(np.sum(s>s[0]*1e-10)));extra=snapshots@right[:rank].T;extra-=v@(v.T@extra);free=~basis['fixed'];q,_=np.linalg.qr(extra[free]);extra[:]=0;extra[free]=q;combined=np.column_stack([v,extra])/root[:,None];reduced=combined.T@(basis['stiffness']@combined)*1e-12;values,rotation=eigh((reduced+reduced.T)*.5);values[np.abs(values)<1e-7]=0
    if values.min()<-1e-7:raise ValueError('Residual-enriched stiffness lost positivity.')
    phi=combined@rotation;result=dict(basis);result.update(phi=phi,gradient=np.einsum('eaim,eaj->eijm',phi.reshape(-1,3,len(values))[mesh['tetra']],mesh['grad']),lambda_us2=values,frequency_mhz=np.sqrt(values)/(2*np.pi),preparation_seconds=basis['preparation_seconds']+time.monotonic()-started,residual_enrichment={'vectors':rank,'uses_transient_labels':False,'retained_force_snapshot_energy':float(np.sum(s[:rank]**2)/np.sum(s**2)),'seconds':time.monotonic()-started});return result
