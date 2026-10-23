"""Matrix-free GLL energy derivative and explicit full hyperelastic dynamics.

This solver is separate from the historical P1 teacher. No modal reduction,
neural correction, mass scaling, damping, or artificial hourglass terms.
"""
import math
import time
import numpy as np
from numba import njit
from solid.fem import pulse
from .mesh import boundary_data,interpolation


@njit(cache=True)
def constitutive(g,constants,law):
    lam,mu,l,m,n=constants;f=g+np.eye(3)
    e=np.empty((3,3))
    for i in range(3):
        for j in range(3):
            e[i,j]=.5*(g[i,j]+g[j,i])
            if law:
                for k in range(3): e[i,j]+=.5*g[k,i]*g[k,j]
    e2=e@e;tr=e[0,0]+e[1,1]+e[2,2];tr2=e2[0,0]+e2[1,1]+e2[2,2]
    a=n if law==2 else 0.;b=m-n/2 if law==2 else 0.;c=l-m+n/2 if law==2 else 0.
    s=(2*mu+2*b*tr)*e+a*e2
    for i in range(3): s[i,i]+=lam*tr+b*tr2+c*tr*tr
    p=s if law==0 else f@s
    tr3=0.
    for i in range(3):
        for j in range(3): tr3+=e2[i,j]*e[j,i]
    energy=.5*lam*tr*tr+mu*tr2+a/3*tr3+b*tr*tr2+c/3*tr**3
    jac=(f[0,0]*(f[1,1]*f[2,2]-f[1,2]*f[2,1])-f[0,1]*(f[1,0]*f[2,2]-f[1,2]*f[2,0])+f[0,2]*(f[1,0]*f[2,1]-f[1,1]*f[2,0]))
    return p,energy,math.sqrt(tr2),jac


@njit(cache=True)
def internal_force(u,elements,derivative,sizes,volume_weight,constants,law):
    force=np.zeros_like(u);energy=0.;max_strain=0.;min_j=1e100;n=elements.shape[1]
    # Scratch is reused per element; assembly is serial and deterministic.
    stress=np.empty((n,n,n,3,3));local=np.empty((n,n,n,3));g=np.empty((3,3))
    for e in range(len(elements)):
        ids=elements[e]
        for i in range(n):
            for j in range(n):
                for k in range(n): local[i,j,k]=u[ids[i,j,k]]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    g[:]=0.
                    for a in range(n):
                        for c in range(3):
                            g[c,0]+=derivative[i,a]*local[a,j,k,c]*2/sizes[e,0]
                            g[c,1]+=derivative[j,a]*local[i,a,k,c]*2/sizes[e,1]
                            g[c,2]+=derivative[k,a]*local[i,j,a,c]*2/sizes[e,2]
                    p,w,strain,jac=constitutive(g,constants,law)
                    energy+=w*volume_weight[e,i,j,k];max_strain=max(max_strain,strain);min_j=min(min_j,jac)
                    stress[i,j,k]=p*volume_weight[e,i,j,k]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    node=ids[i,j,k]
                    for a in range(n):
                        for c in range(3):
                            force[node,c]+=derivative[a,i]*stress[a,j,k,c,0]*2/sizes[e,0]
                            force[node,c]+=derivative[a,j]*stress[i,a,k,c,1]*2/sizes[e,1]
                            force[node,c]+=derivative[a,k]*stress[i,j,a,c,2]*2/sizes[e,2]
    return force,energy,max_strain,min_j


def solve(scene,mesh,cfl=.3,keep_fields=False,callback=None):
    if not .05<=cfl<=.35: raise ValueError('Use CFL 0.05–0.35.')
    start=time.monotonic();material=scene['material'];mass=mesh['mass'];load,fixed,area=boundary_data(mesh,scene)
    constants=np.array([material[k] for k in ['lambda_pa','mu_pa','l_pa','m_pa','n_pa']])
    law={'linear':0,'stvk':1,'murnaghan':2}[scene['constitutive']]
    curved=str(mesh.get('mapping_kind',''))=='isoparametric'
    force_function=internal_force;geometry=mesh.get('size')
    if curved:
        from .curved_solver import internal_force as force_function
        geometry=mesh['inverse_jacobian']
    speed=math.sqrt((material['lambda_pa']+2*material['mu_pa'])/material['rho_kg_m3'])
    sample=scene['acquisition']['sample_step_us']*1e-6;subdivisions=math.ceil(sample/(cfl*mesh['min_node_spacing']/(speed*1.05)));dt=sample/subdivisions
    frames=round(scene['acquisition']['duration_us']/scene['acquisition']['sample_step_us'])+1;steps=(frames-1)*subdivisions
    if steps>100000 or steps*mesh['elements'].size>2_000_000_000: raise ValueError('Spectral reference compute budget exceeded.')
    if keep_fields and frames*len(mass)*3*8>512*1024**2: raise ValueError('Field exceeds 512 MiB.')
    fields=np.zeros((frames,len(mass),3)) if keep_fields else None
    probes=np.zeros((frames,len(scene['receivers']),3));receivers=[interpolation(mesh,r['point_mm']) for r in scene['receivers']]
    u=np.zeros_like(load);v=u.copy();acc=u.copy();previous_ext=u.copy();work=0.;energies=[];max_strain=0.;min_j=1.
    for step in range(1,steps+1):
        next_u=u+dt*v+.5*dt*dt*acc;next_u[fixed]=0
        fint,energy,strain,jac=force_function(next_u,mesh['elements'],mesh['derivative'],geometry,mesh['volume_weight'],constants,law)
        if not np.isfinite(energy) or strain>.001 or jac<=0: raise ValueError(f'Elastic guard failed: strain={strain}, J={jac}')
        max_strain=max(max_strain,strain);min_j=min(min_j,jac)
        ext=load*pulse(step*dt*1e6,scene['source']);next_acc=(ext-fint)/mass[:,None];next_acc[fixed]=0
        next_v=v+.5*dt*(acc+next_acc);next_v[fixed]=0
        work+=float(np.sum(.5*(ext+previous_ext)*(next_u-u)));u,v,acc=next_u,next_v,next_acc;previous_ext=ext
        if step%subdivisions==0:
            frame=step//subdivisions
            if fields is not None: fields[frame]=u
            for i,(ids,w) in enumerate(receivers): probes[frame,i]=w@u[ids]
            energies.append([step*dt*1e6,float(np.sum(.5*mass[:,None]*v*v)),energy,work])
            if callback and (frame%40==0 or frame==frames-1): callback({'frame':frame,'frames':frames,'elapsed_s':time.monotonic()-start})
    energies=np.array(energies);scale=max(np.max(abs(energies[:,3])),np.max(abs(energies[:,1]+energies[:,2])),1e-30)
    balance=float(np.max(abs(energies[:,1]+energies[:,2]-energies[:,3]))/scale)
    diagnostics={'method':'curved_gll_spectral_total_lagrangian_velocity_verlet_v2' if curved else 'cartesian_gll_spectral_total_lagrangian_velocity_verlet_v1','degree':mesh['degree'],'h_mm':mesh['h_mm'],
        'nodes':len(mass),'elements':len(mesh['elements']),'quadrature_points':int(mesh['elements'].size),'internal_steps':steps,'internal_step_us':dt*1e6,
        'min_node_spacing_mm':mesh['min_node_spacing']*1000,'loaded_area_mm2':area*1e6,'volume_mm3':float(mesh['volume_weight'].sum()*1e9),
        'max_strain':max_strain,'min_J':min_j,'external_work_balance_relative':balance,'energy_balance_within_1_percent':balance<=.01,
        'elapsed_s':time.monotonic()-start,'calculation_dtype':'float64','output_dtype':'float64','accepted_for_solver_routing':False,
        'spatial_convergence_verified':False,'temporal_convergence_verified':False,'energy_samples':energies.tolist()}
    return {'probes_m':probes,'fields_m':fields,'times_us':np.arange(frames)*scene['acquisition']['sample_step_us'],'diagnostics':diagnostics}
