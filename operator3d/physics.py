"""Differentiable 3-D total-Lagrangian weak residual on actual tetrahedra.

Displacements are in U0, time in microseconds. Spatial gradients and forces
are assembled in SI; natural tractions are integrated on actual surface faces.
"""
import torch
L0,T0,U0=.01,1e-6,1e-7

def piola(g,m,law):
    eye=torch.eye(3,dtype=g.dtype,device=g.device);f=eye+g
    e=(g+g.transpose(-1,-2))*.5 if law=='linear' else (g+g.transpose(-1,-2)+g.transpose(-1,-2)@g)*.5
    tr=e.diagonal(dim1=-2,dim2=-1).sum(-1);e2=e@e
    a=m['n_pa'] if law=='murnaghan' else 0.;b=m['m_pa']-m['n_pa']/2 if law=='murnaghan' else 0.;c=m['l_pa']-m['m_pa']+m['n_pa']/2 if law=='murnaghan' else 0.
    s=(m['lambda_pa']*tr+b*e2.diagonal(dim1=-2,dim2=-1).sum(-1)+c*tr**2)[...,None,None]*eye+(2*m['mu_pa']+2*b*tr)[...,None,None]*e+a*e2
    return (s if law=='linear' else f@s),e,torch.linalg.det(f)

def internal(parts,case):
    """Return increments of internal nodal force, strain and determinant."""
    total=parts.cumsum(1)*U0;tet=case['tetra'];grad=case['grad'];vol=case['volume'];forces=[];strains=[];jac=[]
    for k,law in enumerate(['linear','stvk','murnaghan']):
        g=torch.einsum('eai,eaj->eij',total[tet,k],grad);p,e,j=piola(g,case['scene']['material'],law)
        local=torch.einsum('eij,eaj,e->eai',p,grad,vol)
        force=torch.zeros((len(parts),3),dtype=parts.dtype,device=parts.device).index_add(0,tet.reshape(-1),local.reshape(-1,3))
        forces.append(force);strains.append(e);jac.append(j)
    return torch.stack([forces[0],forces[1]-forces[0],forces[2]-forces[1]],1),torch.stack(strains,1),torch.stack(jac,1)

def pulse(time,source):
    tau=time-source['delay_us'];duration=source['cycles']/source['frequency_mhz'];value=.5*(1-torch.cos(2*torch.pi*tau/duration))*torch.sin(2*torch.pi*source['frequency_mhz']*tau+source['phase_deg']*torch.pi/180)
    return torch.where((tau>=0)&(tau<=duration),value,0.)

def weak_residual(before,current,after,time,delta_us,case):
    force,strain,jac=internal(current,case)
    acceleration=(after-2*current+before)/(delta_us**2)
    external=case['load']*pulse(time,case['scene']['source']);target=torch.stack([external,external*0,external*0],1)
    residual=acceleration+(force-target)/(case['mass'][:,None,None]*(U0/T0**2))
    return residual,strain,jac

def physics_loss(model,case,time,scales,delta_us=.025):
    t=torch.as_tensor(time,dtype=torch.float64);n=len(case['points']);indices=torch.arange(n);times=torch.cat([torch.full((n,),time-delta_us,dtype=torch.float64),torch.full((n,),time,dtype=torch.float64),torch.full((n,),time+delta_us,dtype=torch.float64)])
    values=model(case,indices.repeat(3),times);before,current,after=values.split(n)
    residual,strain,jac=weak_residual(before,current,after,t,delta_us,case)
    scale=scales[:,None]*(2*torch.pi*case['scene']['source']['frequency_mhz'])**2
    squared=((residual/scale)**2).mean((1,2));mass=case['mass'];free=~case['fixed'];boundary=case['boundary_nodes']&free;interior=(~case['boundary_nodes'])&free
    def weighted(mask):
        return (squared[mask]*mass[mask]).sum()/mass[mask].sum() if mask.any() else squared.sum()*0
    guard=torch.relu(torch.linalg.matrix_norm(strain,dim=(-2,-1))/.001-1).square().mean()+torch.relu(1e-6-jac).square().mean()
    return weighted(interior),weighted(boundary),guard,{'max_strain':torch.linalg.matrix_norm(strain,dim=(-2,-1)).max(),'min_J':jac.min()}
