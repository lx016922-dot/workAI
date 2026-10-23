"""Float64 lumped-mass tetrahedra, objective total-Lagrangian hyperelasticity."""
import math
import numpy as np
from numba import njit

@njit(cache=True)
def stress_energy(g,lam,mu,l,m,n,law):
    eye=np.eye(3);f=eye+g
    if law==0:e=(g+g.T)*.5
    else:e=(g+g.T+g.T@g)*.5
    e2=e@e;tr=np.trace(e);tr2=np.trace(e2)
    a=n if law==2 else 0.;b=m-n/2 if law==2 else 0.;c=l-m+n/2 if law==2 else 0.
    s=(lam*tr+b*tr2+c*tr*tr)*eye+(2*mu+2*b*tr)*e+a*e2
    p=s if law==0 else f@s
    w=.5*lam*tr*tr+mu*tr2+a/3*np.trace(e2@e)+b*tr*tr2+c/3*tr**3
    return p,w,math.sqrt(tr2),np.linalg.det(f)

@njit(cache=True)
def internal_force(u,tetra,grad,vol,constants,law):
    force=np.zeros_like(u);energy=0.;max_strain=0.;min_j=1e100
    for k in range(len(tetra)):
        g=np.zeros((3,3))
        for a in range(4):
            for i in range(3):
                for j in range(3):g[i,j]+=u[tetra[k,a],i]*grad[k,a,j]
        p,w,strain,jac=stress_energy(g,constants[0],constants[1],constants[2],constants[3],constants[4],law)
        energy+=w*vol[k];max_strain=max(max_strain,strain);min_j=min(min_j,jac)
        for a in range(4):
            for i in range(3):
                for j in range(3):force[tetra[k,a],i]+=vol[k]*p[i,j]*grad[k,a,j]
    return force,energy,max_strain,min_j

def lumped_mass(mesh,rho):
    mass=np.zeros(len(mesh['points']))
    for j in range(4):np.add.at(mass,mesh['tetra'][:,j],rho*mesh['volume']/4)
    if not np.all(mass>0):raise ValueError('存在无质量节点。')
    return mass

def pulse(time_us,source):
    tau=time_us-source['delay_us'];duration=source['cycles']/source['frequency_mhz']
    if tau<0 or tau>duration:return 0.
    return .5*(1-math.cos(2*math.pi*tau/duration))*math.sin(2*math.pi*source['frequency_mhz']*tau+math.radians(source['phase_deg']))

def solve(scene,mesh,load,fixed_nodes,callback=None,output_precision='float32'):
    material=scene['material'];rho=material['rho_kg_m3'];lam=material['lambda_pa'];mu=material['mu_pa']
    constants=np.array([material[k] for k in ['lambda_pa','mu_pa','l_pa','m_pa','n_pa']],dtype=np.float64)
    law={'linear':0,'stvk':1,'murnaghan':2}[scene['constitutive']]
    mass=lumped_mass(mesh,rho);cl=math.sqrt((lam+2*mu)/rho)
    acq=scene['acquisition'];sample_step=acq['sample_step_us']*1e-6
    stable=scene['numerics']['cfl']*mesh['min_altitude']/(cl*1.05)
    subdivisions=max(1,math.ceil(sample_step/stable));dt=sample_step/subdivisions
    frames=round(acq['duration_us']/acq['sample_step_us'])+1;steps=(frames-1)*subdivisions
    if steps>50000 or steps*len(mesh['tetra'])>350_000_000:raise ValueError(f'实际网格需要 {steps} 个时间步，超出当前 CPU 预算；请缩小计算域或时长。')
    if output_precision not in ['float32','float64']:raise ValueError('场输出精度无效。')
    dtype=np.dtype(output_precision)
    if frames*len(mass)*3*dtype.itemsize>192*1024**2:raise ValueError('完整位移场超过 192 MiB；请减少输出采样数量或节点数。')
    fields=np.zeros((frames,len(mass),3),dtype=dtype)
    u=np.zeros((len(mass),3));v=np.zeros_like(u);a=np.zeros_like(u)
    previous_ext=np.zeros_like(u);work=0.;energies=[];max_strain=0.;min_j=1.;max_quantization=0.
    if callback:callback({'phase':'running','step':0,'steps_total':steps,'internal_step_us':dt*1e6,'nodes':len(mass),'elements':len(mesh['tetra'])})
    for step in range(1,steps+1):
        next_u=u+dt*v+.5*dt*dt*a;next_u[fixed_nodes]=0
        fint,strain_energy,strain,jac=internal_force(next_u,mesh['tetra'],mesh['grad'],mesh['volume'],constants,law)
        if not np.isfinite(strain_energy) or strain>.001 or jac<=0:raise ValueError(f'超出当前铝弹性范围：应变={strain:.5g}, J={jac:.5g}；未发布有效结果。')
        max_strain=max(max_strain,strain);min_j=min(min_j,jac)
        ext=load*pulse(step*dt*1e6,scene['source'])
        next_a=(ext-fint)/mass[:,None];next_a[fixed_nodes]=0
        next_v=v+.5*dt*(a+next_a);next_v[fixed_nodes]=0
        work+=float(np.sum(.5*(ext+previous_ext)*(next_u-u)))
        previous_ext=ext;u,v,a=next_u,next_v,next_a
        if step%subdivisions==0:
            frame=step//subdivisions;fields[frame]=u
            max_quantization=max(max_quantization,float(np.max(abs(u-fields[frame]))))
            kinetic=float(np.sum(.5*mass[:,None]*v*v))
            energies.append([step*dt*1e6,kinetic,strain_energy,work])
            if callback and (frame%10==0 or frame==frames-1):callback({'phase':'running','step':step,'steps_total':steps,'output_frame':frame,'frames_total':frames,'max_strain':max_strain,'min_J':min_j})
    energies=np.array(energies);energy_scale=max(float(np.max(abs(energies[:,3]))),float(np.max(abs(energies[:,1]+energies[:,2]))),1e-30)
    balance=float(np.max(abs(energies[:,1]+energies[:,2]-energies[:,3]))/energy_scale)
    return fields,{'method':'total_lagrangian_p1_tetra_lumped_mass_velocity_verlet','calculation_dtype':'float64','output_dtype':output_precision,
                   'internal_step_us':dt*1e6,'internal_steps':steps,'frames':frames,'mass_kg':float(mass.sum()),'min_altitude_mm':mesh['min_altitude']*1000,
                   'max_strain':max_strain,'min_J':min_j,'external_work_balance_relative':balance,'max_export_roundoff_m':max_quantization,
                   'energy_samples':energies.tolist(),'energy_balance_within_1_percent':balance<=.01,
                   'spatial_convergence_verified':False,'temporal_convergence_verified':False,'comsol_agreement_verified':False,
                   'surrogate_used':False,'accuracy_status':'requires_convergence_and_reference_checks'}
