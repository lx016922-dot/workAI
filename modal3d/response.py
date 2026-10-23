"""Weakly nonlinear modal representation, before learning material coupling.

This is a physical reduced model, not a neural prediction. It is used first to
test whether the representation can resolve the weak material increment.
"""
import time
import numpy as np
from scipy.signal import fftconvolve
from solid.mesh import nodal_load

def pulse(times,source):
    tau=times-source['delay_us'];duration=source['cycles']/source['frequency_mhz'];inside=(tau>=0)&(tau<=duration)
    return np.where(inside,.5*(1-np.cos(2*np.pi*tau/duration))*np.sin(2*np.pi*source['frequency_mhz']*tau+np.deg2rad(source['phase_deg'])),0.)

def oscillators(forces,times,lambdas):
    """Trapezoidal Duhamel convolution, zero initial displacement/velocity."""
    dt=times[1]-times[0];omega=np.sqrt(lambdas);impulse=np.empty((len(times),len(omega)));positive=omega>0
    impulse[:,positive]=np.sin(times[:,None]*omega[None,positive])/omega[None,positive];impulse[:,~positive]=times[:,None]
    output=fftconvolve(forces,impulse,mode='full',axes=0)[:len(times)]*dt-.5*dt*impulse*forces[0]
    output[0]=0.;return output

def quadratic_stresses(gradient,material):
    g=gradient;e=(g+np.swapaxes(g,-1,-2))*.5;eye=np.eye(3);e2=e@e;tr=np.trace(e,axis1=-2,axis2=-1);tr2=np.trace(e2,axis1=-2,axis2=-1);gtg=np.swapaxes(g,-1,-2)@g
    lam=material['lambda_pa'];mu=material['mu_pa'];linear=lam*tr[...,None,None]*eye+2*mu*e
    geometric=lam*.5*np.trace(gtg,axis1=-2,axis2=-1)[...,None,None]*eye+mu*gtg+g@linear
    # Material energy basis: a/3 tr(E^3), b tr(E)tr(E^2), c/3 tr(E)^3.
    return geometric,e2*1e11,(tr2[...,None,None]*eye+2*tr[...,None,None]*e)*1e11,(tr**2)[...,None,None]*eye*1e11

def response(mesh,scene,basis,step_us=.01,static_correction=False,integrator=oscillators):
    started=time.monotonic();duration=scene['acquisition']['duration_us'];sample=scene['acquisition']['sample_step_us'];sub=round(sample/step_us)
    if sub<1 or abs(sub*step_us-sample)>1e-10:raise ValueError('Reduced time step must divide output spacing.')
    times=np.arange(round(duration/step_us)+1)*step_us;phi=basis['phi'];grad=basis['gradient'];lam=basis['lambda_us2'];load,_=nodal_load(mesh,scene['source']);forcing=(phi.T@load.ravel())*1e-12
    envelope=pulse(times,scene['source']);q0=integrator(envelope[:,None]*forcing[None,:],times,lam);projected=np.empty((4,len(times),len(lam)));weighted=(grad*mesh['volume'][:,None,None,None]).reshape(-1,len(lam))
    rest0=basis['residual_flexibility'](load.reshape(-1,1))[:,0] if static_correction else np.zeros(phi.shape[0]);rest_gradient=np.einsum('eai,eaj->eij',rest0.reshape(-1,3)[mesh['tetra']],mesh['grad'])
    corrections=np.zeros((4,len(times[::sub]),phi.shape[0]))
    flat_gradient=grad.reshape(-1,len(lam))
    for start in range(0,len(times),48):
        stop=min(start+48,len(times));g=(q0[start:stop]@flat_gradient.T).reshape(-1,len(mesh['tetra']),3,3)+envelope[start:stop,None,None,None]*rest_gradient[None]
        for k,stress in enumerate(quadratic_stresses(g,scene['material'])):
            projected[k,start:stop]=-stress.reshape(stop-start,-1)@weighted*1e-12
            if static_correction:
                selected=np.arange(start,stop);selected=selected[selected%sub==0];force=-basis['divergence']@stress[selected-start].reshape(len(selected),-1).T
                corrections[k,selected//sub]=basis['residual_flexibility'](force).T
    branches=[q0[::sub]]+[integrator(force,times,lam)[::sub] for force in projected]
    m=scene['material'];coeff=np.array([m['n_pa'],m['m_pa']-m['n_pa']/2,m['l_pa']-m['m_pa']+m['n_pa']/2])/1e11
    q=np.stack([branches[0],branches[1],sum(c*b for c,b in zip(coeff,branches[2:]))],2)
    # time,node,part,xyz; never mark these as reference FEM results.
    parts=np.einsum('tmk,ncm->tnkc',q,phi.reshape(-1,3,len(lam)));parts[:,:,0]+=envelope[::sub,None,None]*rest0.reshape(1,-1,3);parts[:,:,1]+=corrections[0].reshape(len(parts),-1,3);parts[:,:,2]+=np.einsum('k,ktd->td',coeff,corrections[1:]).reshape(len(parts),-1,3)
    acceleration=np.stack([envelope[::sub,None]*forcing[None,:]-branches[0]*lam]+[projected[k,::sub]-branches[k+1]*lam for k in range(4)],1)
    return {'parts_m':parts,'modal_basis_responses':np.stack(branches,1),'modal_basis_acceleration_m_us2':acceleration,'static_basis_responses':corrections,'material_coefficients':coeff,'times_us':times[::sub],'response_seconds':time.monotonic()-started,'static_correction':static_correction,'method':'second_order_modal_physical_baseline_not_neural','inference_uses_transient_labels':False,'accepted_for_solver_routing':False}
