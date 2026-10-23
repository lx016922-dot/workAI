"""Differentiable GLL weak momentum on complete incident-element stars.

All elements incident to each sampled node are assembled, including natural
surface traction. No tetrahedral or point-collocation residual is substituted.
"""
import numpy as np
import torch
from operator3d.physics import piola,pulse,U0,T0


def geometry(case):
    if 'sem_geometry' in case:return case['sem_geometry']
    mesh=case['mesh'];n=int(mesh['degree'])+1;eye=np.eye(n);d=mesh['derivative']
    ref=np.stack([np.einsum('ia,jb,kc->ijkabc',d,eye,eye),np.einsum('ia,jb,kc->ijkabc',eye,d,eye),np.einsum('ia,jb,kc->ijkabc',eye,eye,d)],-1).reshape(n**3,n**3,3)
    if 'inverse_jacobian' in mesh:inverse=mesh['inverse_jacobian'].reshape(-1,n**3,3,3)
    else:inverse=np.broadcast_to(np.eye(3)[None,None]* (2/mesh['size'])[:,None,None,:],(len(mesh['elements']),n**3,3,3))
    grad=np.einsum('qaA,eqAj->eqaj',ref,inverse)
    cells=mesh['elements'].reshape(-1,n**3).astype(np.int64);incident=[[] for _ in mesh['points']]
    for e,ids in enumerate(cells):
        for i in ids:incident[i].append(e)
    case['sem_geometry']={'cells':cells,'grad':torch.from_numpy(grad),'weights':torch.as_tensor(mesh['volume_weight'].reshape(-1,n**3)),'incident':incident}
    return case['sem_geometry']


def star(case,targets):
    g=geometry(case);targets=np.unique(np.asarray(targets,dtype=np.int64));els=np.unique([e for i in targets for e in g['incident'][i]])
    ids,inverse=np.unique(g['cells'][els],return_inverse=True);cells=torch.as_tensor(inverse.reshape(len(els),-1));at=torch.as_tensor(np.searchsorted(ids,targets))
    return {'ids':torch.as_tensor(ids),'cells':cells,'grad':g['grad'][els],'weights':g['weights'][els],'at':at,'targets':torch.as_tensor(targets)}


def internal(parts,case,patch):
    total=parts.cumsum(1)*U0;cells=patch['cells'];forces=[];strains=[];jac=[]
    for k,law in enumerate(['linear','stvk','murnaghan']):
        grad=torch.einsum('eai,eqaj->eqij',total[cells,k],patch['grad']);p,e,j=piola(grad,case['scene']['material'],law)
        local=torch.einsum('eqij,eqaj,eq->eai',p,patch['grad'],patch['weights'])
        force=torch.zeros((len(parts),3),dtype=parts.dtype).index_add(0,cells.flatten(),local.reshape(-1,3))[patch['at']]
        forces.append(force);strains.append(e);jac.append(j)
    return torch.stack([forces[0],forces[1]-forces[0],forces[2]-forces[1]],1),torch.stack(strains,1),torch.stack(jac,1)


def choose_targets(case,rng,count=3):
    free=~case['fixed'].numpy();boundary=case['boundary_nodes'].numpy();loaded=np.any(case['load'].numpy()!=0,axis=1);selected=[]
    # Every sampled balance includes a forced node as well as a traction-free
    # node; a zero field must not appear to satisfy only unloaded boundaries.
    for mask,n in [(free&~boundary,count),(free&boundary&loaded,1),(free&boundary&~loaded,1)]:
        candidates=np.flatnonzero(mask)
        if len(candidates):selected.extend(rng.choice(candidates,size=min(n,len(candidates)),replace=False))
    return np.array(selected,dtype=np.int64)


def physics_loss(model,case,time,scales,targets,delta_us=.025):
    patch=star(case,targets);n=len(patch['ids']);times=torch.cat([torch.full((n,),time+delta,dtype=torch.float64) for delta in [-delta_us,0,delta_us]])
    before,current,after=model(case,patch['ids'].repeat(3),times).split(n);force,strain,jac=internal(current,case,patch)
    accel=(after-2*current+before)[patch['at']]/delta_us**2
    ids=patch['targets'];external=case['load'][ids]*pulse(torch.tensor(time,dtype=torch.float64),case['scene']['source']);ext=torch.stack([external,external*0,external*0],1)
    residual=accel+(force-ext)/(case['mass'][ids,None,None]*(U0/T0**2));scale=scales[:,None]*(2*torch.pi*case['scene']['source']['frequency_mhz'])**2
    squared=(residual/scale).square().mean((1,2));boundary=case['boundary_nodes'][ids];free=~case['fixed'][ids]
    def mean(mask):return squared[mask].mean() if mask.any() else squared.sum()*0
    norm=torch.linalg.matrix_norm(strain,dim=(-2,-1));guard=torch.relu(norm/.001-1).square().mean()+torch.relu(1e-6-jac).square().mean()
    return mean(free&~boundary),mean(free&boundary),guard,{'max_strain':norm.max(),'min_J':jac.min(),'assembled_nodes':n,'target_nodes':len(ids)}
