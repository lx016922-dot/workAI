"""Development-only full hyperelastic modal waveform relaxation.

This adds nonlinear feedback omitted by the second-order representation.
Convergence and full-mesh residual checks are required for every prediction.
"""
import time
import numpy as np
from solid.mesh import nodal_load
from .response import pulse
from .propagator import linear_force_oscillators

def stress(g,material,multipliers=None):
    eye=np.eye(3);f=eye+g;e=(g+np.swapaxes(g,-1,-2)+np.swapaxes(g,-1,-2)@g)*.5;e2=e@e;tr=np.trace(e,axis1=-2,axis2=-1)
    s=material['lambda_pa']*tr[...,None,None]*eye+2*material['mu_pa']*e
    if multipliers is not None:
        a,b,c=np.asarray(multipliers)*[material['n_pa'],material['m_pa']-material['n_pa']/2,material['l_pa']-material['m_pa']+material['n_pa']/2]
        s+=(b*np.trace(e2,axis1=-2,axis2=-1)+c*tr**2)[...,None,None]*eye+2*b*tr[...,None,None]*e+a*e2
    return f@s

def linear_stress(g,material):
    return material['lambda_pa']*np.trace(g,axis1=-2,axis2=-1)[...,None,None]*np.eye(3)+material['mu_pa']*(g+np.swapaxes(g,-1,-2))

def nonlinear_response(mesh,scene,basis,multipliers=(1.,1.,1.),step_us=.025,iterations=3):
    started=time.monotonic();sub=round(scene['acquisition']['sample_step_us']/step_us)
    if sub<1 or abs(sub*step_us-scene['acquisition']['sample_step_us'])>1e-10:raise ValueError('Time step must divide output spacing.')
    times=np.arange(round(scene['acquisition']['duration_us']/step_us)+1)*step_us;phi=basis['phi'];gradient=basis['gradient'];lambdas=basis['lambda_us2'];modes=len(lambdas);load,_=nodal_load(mesh,scene['source']);external=pulse(times,scene['source'])[:,None]*(phi.T@load.ravel())[None,:]*1e-12;q0=linear_force_oscillators(external,times,lambdas);increments=np.zeros((2,len(times),modes));weighted=(gradient*mesh['volume'][:,None,None,None]).reshape(-1,modes);flat=gradient.reshape(-1,modes);history=[]
    for iteration in range(iterations):
        forces=np.empty_like(increments)
        for start in range(0,len(times),32):
            stop=min(start+32,len(times));g=(q0[start:stop]@flat.T).reshape(-1,len(mesh['tetra']),3,3);gg=(increments[0,start:stop]@flat.T).reshape(g.shape);gm=(increments[1,start:stop]@flat.T).reshape(g.shape);geo=stress(g+gg,scene['material']);full=stress(g+gg+gm,scene['material'],multipliers)
            forces[0,start:stop]=-(geo-linear_stress(g+gg,scene['material'])).reshape(stop-start,-1)@weighted*1e-12
            forces[1,start:stop]=-(full-geo-linear_stress(gm,scene['material'])).reshape(stop-start,-1)@weighted*1e-12
        next_increments=np.stack([linear_force_oscillators(f,times,lambdas) for f in forces]);history.append({'iteration':iteration+1,'relative_change':[float(np.linalg.norm(next_increments[k]-increments[k])/max(np.linalg.norm(next_increments[k]),1e-100)) for k in range(2)]});increments=next_increments
    q=np.stack([q0,increments[0],increments[1]],1)[::sub];acc=np.stack([external-q0*lambdas,forces[0]-increments[0]*lambdas,forces[1]-increments[1]*lambdas],1)[::sub];modeshape=phi.reshape(-1,3,modes)
    return {'parts_m':np.einsum('tkm,ncm->tnkc',q,modeshape),'acceleration_m_us2':np.einsum('tkm,ncm->tnkc',acc,modeshape),'times_us':times[::sub],'iteration_history':history,'response_seconds':time.monotonic()-started,'accepted_for_solver_routing':False,'method':'full_hyperelastic_modal_waveform_relaxation_development_only','inference_uses_transient_labels':False}
