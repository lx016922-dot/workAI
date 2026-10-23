"""Input-only elastic modes, with the same P1/lumped mass as the reference.

Computing these geometry-dependent modes is classical numerical preparation,
not neural inference and not a solved transient response.
"""
import time
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import eigsh,splu
from solid.mesh import face_selection

def assemble(mesh,material):
    g=mesh['grad'];v=mesh['volume'];lam=material['lambda_pa'];mu=material['mu_pa'];n=len(mesh['points']);eye=np.eye(3)
    blocks=(lam*np.einsum('eai,ebj->eaibj',g,g)+mu*np.einsum('eaj,ebi->eaibj',g,g)+mu*np.einsum('ij,eab->eaibj',eye,np.einsum('eak,ebk->eab',g,g)))*v[:,None,None,None,None]
    dofs=(mesh['tetra'][:,:,None]*3+np.arange(3)).reshape(-1,12);rows=np.repeat(dofs,12,axis=1).reshape(-1);cols=np.tile(dofs,(1,12)).reshape(-1)
    stiffness=coo_matrix((blocks.reshape(-1),(rows,cols)),shape=(3*n,3*n)).tocsr();mass=np.zeros(n)
    for a in range(4):np.add.at(mass,mesh['tetra'][:,a],material['rho_kg_m3']*v/4)
    return stiffness,np.repeat(mass,3)

def build_basis(mesh,scene,count=96):
    started=time.monotonic();k,m=assemble(mesh,scene['material']);fixed=np.zeros(len(m),dtype=bool)
    if scene['boundary']['fixed_face']!='none':
        nodes=np.unique(mesh['surface'][face_selection(mesh,scene['boundary']['fixed_face'])]);fixed[(nodes[:,None]*3+np.arange(3)).ravel()]=True
    free=np.flatnonzero(~fixed);scale=1/np.sqrt(m[free]);a=diags(scale)@k[free][:,free]@diags(scale)*1e-12
    if not 8<=count<len(free)-1:raise ValueError('Mode count must leave a real reduction.')
    values,vectors=eigsh(a,k=count,sigma=-.001,which='LM',v0=np.sin(np.arange(len(free))*.73+.31),tol=2e-11,ncv=min(len(free),max(2*count+8,30)))
    order=np.argsort(values);values=values[order];vectors=vectors[:,order]
    if values.min() < -1e-7:raise ValueError('Elastic stiffness has an unexpected negative eigenvalue.')
    values[np.abs(values)<1e-7]=0.
    phi=np.zeros((len(m),count));phi[free]=vectors*scale[:,None]
    residual=np.linalg.norm(a@vectors-vectors*values[None,:],axis=0)/np.maximum(np.linalg.norm(a@vectors,axis=0),1.)
    if residual.max()>1e-7:raise ValueError('Elastic mode residual exceeds tolerance.')
    ortho=phi.T@(m[:,None]*phi)
    if np.linalg.norm(ortho-np.eye(count),ord=np.inf)>1e-7:raise ValueError('Mass orthogonality failed.')
    shape=phi.reshape(-1,3,count);gradient=np.einsum('eaim,eaj->eijm',shape[mesh['tetra']],mesh['grad'])
    # Divergence transpose: internal nodal force from per-element Piola tensors.
    rows=[];cols=[];data=[];elements=np.arange(len(mesh['tetra']))
    for local in range(4):
        for i in range(3):
            for j in range(3):rows.append(mesh['tetra'][:,local]*3+i);cols.append(elements*9+i*3+j);data.append(mesh['volume']*mesh['grad'][:,local,j])
    divergence=coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(len(m),len(elements)*9)).tocsr()
    factor=splu((a+diags(np.full(len(free),.001))).tocsc())
    def residual_flexibility(force):
        normalized=force[free]*scale[:,None]*1e-12;remainder=normalized-vectors@(vectors.T@normalized);value=factor.solve(remainder);value-=vectors@(vectors.T@value);result=np.zeros_like(force);result[free]=value*scale[:,None];return result
    return {'phi':phi,'gradient':gradient,'lambda_us2':values,'mass':m,'fixed':fixed,'stiffness':k,'divergence':divergence,'residual_flexibility':residual_flexibility,'frequency_mhz':np.sqrt(values)/(2*np.pi),'eigen_residual':residual,'preparation_seconds':time.monotonic()-started,'input_only':True}
