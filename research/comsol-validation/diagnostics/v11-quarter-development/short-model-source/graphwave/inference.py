"""Compiled forward pass of the frozen polynomial recurrent graph network.

Only geometry preprocessing and learned tensors are used. No teacher law,
teacher displacement, FEM solver, Newton iteration or residual correction is
available in this module. The recurrence has a fixed geometry-based number
of layers, and is explicit about being a discretized graph dynamics model.
The compiled operations are checked against the PyTorch network at export.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def energy_activation(gradient, quadratic, cubic, law):
    strain=.5*(gradient+gradient.T)
    if law:strain=strain+.5*gradient.T@gradient
    z=np.array([strain[0,0],strain[1,1],strain[2,2],math.sqrt(2)*strain[1,2],math.sqrt(2)*strain[0,2],math.sqrt(2)*strain[0,1]])/1e-4
    linear=2*(quadratic@z);nonlinear=np.zeros(6)
    if law==2:
        for a in range(6):
            for b in range(6):
                for c in range(6):nonlinear[a]+=3*cubic[a,b,c]*z[b]*z[c]
    s=1e7*(linear+1e-4*nonlinear)
    second=np.array([[s[0],s[5]/math.sqrt(2),s[4]/math.sqrt(2)],
                     [s[5]/math.sqrt(2),s[1],s[3]/math.sqrt(2)],
                     [s[4]/math.sqrt(2),s[3]/math.sqrt(2),s[2]]])
    f=np.eye(3)+gradient;first=second if law==0 else f@second
    energy=1e3*(.5*np.dot(z,linear)+1e-4/3*np.dot(z,nonlinear))
    j=(f[0,0]*(f[1,1]*f[2,2]-f[1,2]*f[2,1])-f[0,1]*(f[1,0]*f[2,2]-f[1,2]*f[2,0])+f[0,2]*(f[1,0]*f[2,1]-f[1,1]*f[2,0]))
    return first,energy,math.sqrt(np.sum(strain*strain)),j


@njit(cache=True)
def graph_message(current,cells,derivative,inverse,volume,quadratic,cubic,law):
    message=np.zeros_like(current);potential=0.;strain_max=0.;jac_min=1.;n=cells.shape[1]
    local=np.empty((n,n,n,3));flux=np.empty((n,n,n,3,3));ref=np.empty((3,3))
    for element in range(len(cells)):
        ids=cells[element]
        for i in range(n):
            for j in range(n):
                for k in range(n):local[i,j,k]=current[ids[i,j,k]]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    ref[:]=0.
                    for neighbor in range(n):
                        for axis in range(3):
                            ref[axis,0]+=derivative[i,neighbor]*local[neighbor,j,k,axis]
                            ref[axis,1]+=derivative[j,neighbor]*local[i,neighbor,k,axis]
                            ref[axis,2]+=derivative[k,neighbor]*local[i,j,neighbor,axis]
                    transform=inverse[element,i,j,k];g=ref@transform
                    stress,energy,strain,jac=energy_activation(g,quadratic,cubic,law)
                    flux[i,j,k]=(stress@transform.T)*volume[element,i,j,k]
                    potential+=energy*volume[element,i,j,k];strain_max=max(strain_max,strain);jac_min=min(jac_min,jac)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    node=ids[i,j,k]
                    for neighbor in range(n):
                        for axis in range(3):
                            message[node,axis]+=(derivative[neighbor,i]*flux[neighbor,j,k,axis,0]+
                                                derivative[neighbor,j]*flux[i,neighbor,k,axis,1]+
                                                derivative[neighbor,k]*flux[i,j,neighbor,axis,2])
    return message,potential,strain_max,jac_min


@njit(cache=True)
def recurrent_forward(cells,derivative,inverse,volume,mass,load,fixed,quadratic,cubic,state_weights,
                      law,dt,subdivisions,frames,pulse_values):
    current=np.zeros_like(load);previous=np.zeros_like(load);last_external=np.zeros_like(load)
    fields=np.zeros((frames,len(mass),3));energy_log=np.zeros((frames,4));work=0.;max_strain=0.;min_j=1.
    total_steps=(frames-1)*subdivisions
    for step in range(total_steps+1):
        message,potential,strain,jac=graph_message(current,cells,derivative,inverse,volume,quadratic,cubic,law)
        if not np.isfinite(potential) or strain>.001 or jac<=0:raise ValueError('Graph recurrence violates strain/J/finite-energy guard.')
        max_strain=max(max_strain,strain);min_j=min(min_j,jac)
        external=load*pulse_values[step];scaled=(external-message)/mass[:,None]*dt*dt
        scaled[fixed]=0.
        work+=np.sum(.5*(external+last_external)*(current-previous))
        if step%subdivisions==0:
            frame=step//subdivisions;fields[frame]=current
            velocity=(current-previous+.5*scaled)/dt;velocity[fixed]=0.
            kinetic=np.sum(.5*mass[:,None]*velocity*velocity)
            energy_log[frame]=np.array([step*dt*1e6,kinetic,potential,work])
        if step<total_steps:
            next_state=current+state_weights[0]*(current-previous)+state_weights[1]*scaled
            next_state[fixed]=0.;previous=current;current=next_state;last_external=external
    return fields,energy_log,max_strain,min_j


def inverse_mapping(mesh):
    if 'inverse_jacobian' in mesh:return np.ascontiguousarray(mesh['inverse_jacobian'])
    n=int(mesh['degree'])+1
    values=np.eye(3)[None,None,None,None]*(2/mesh['size'])[:,None,None,None,None,:]
    return np.ascontiguousarray(np.broadcast_to(values,(len(mesh['elements']),n,n,n,3,3)))


def source_values(source,times_us):
    t=times_us-source['delay_us'];duration=source['cycles']/source['frequency_mhz']
    value=.5*(1-np.cos(2*np.pi*t/duration))*np.sin(2*np.pi*source['frequency_mhz']*t+source['phase_deg']*np.pi/180)
    return np.where((t>=0)&(t<=duration),value,0.)
