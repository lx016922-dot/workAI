"""Accurate driven linear states from candidate frequencies and the declared pulse.

This module has no reference-data inputs. The oscillators use candidate graph
frequencies; the forcing interval is integrated to a fixed high tolerance and
the unforced tail uses the exact candidate oscillator solution.
"""
import numpy as np
from scipy.integrate import solve_ivp

from .signals import pulse_values,specification


def driven_states(scene,frequency,times_us,*,rtol=1e-12):
    frequency=np.asarray(frequency,dtype=float);times=np.asarray(times_us,dtype=float)
    if frequency.ndim!=1 or np.any(frequency<=0) or not np.isfinite(frequency).all() or not np.isfinite(times).all():
        raise ValueError('Finite positive frequencies and finite query times required')
    omega=2*np.pi*frequency;count=len(omega)
    duration=specification(scene)['duration_us']*1e-6;delay=scene['source']['delay_us']*1e-6
    def rhs(t,state):
        drive=float(pulse_values(scene,(t+delay)*1e6))
        return np.r_[omega*state[count:],omega*(drive-state[:count])]
    solution=solve_ivp(rhs,(0.,duration),np.zeros(2*count),method='DOP853',
        rtol=rtol,atol=rtol*1e-2,max_step=1/(frequency.max()*20),dense_output=True)
    if not solution.success:raise ValueError('Candidate forcing integration failed: '+solution.message)
    local=times.ravel()*1e-6-delay
    position=np.zeros((len(local),count));velocity=np.zeros_like(position)
    forced=np.flatnonzero((local>=0)&(local<=duration));free=np.flatnonzero(local>duration)
    # Bound temporary arrays for complete 5 ms records and many eigenmodes.
    for first in range(0,len(forced),2048):
        indices=forced[first:first+2048];state=solution.sol(local[indices])
        position[indices]=state[:count].T/(omega[None,:]**2)
        velocity[indices]=state[count:].T/omega[None,:]
    terminal=solution.y[:,-1]
    for first in range(0,len(free),2048):
        indices=free[first:first+2048];angle=(local[indices,None]-duration)*omega[None,:]
        cosine,sine=np.cos(angle),np.sin(angle)
        position[indices]=(cosine*terminal[None,:count]+sine*terminal[None,count:])/(omega[None,:]**2)
        velocity[indices]=(-sine*terminal[None,:count]+cosine*terminal[None,count:])/omega[None,:]
    return position.reshape(times.shape+(count,)),velocity.reshape(times.shape+(count,)),{
        'method':'DOP853 candidate forcing; exact candidate free oscillator evolution',
        'relative_tolerance':rtol,'scaled_absolute_tolerance':rtol*1e-2,
        'forcing_rhs_evaluations':solution.nfev,'comsol_arrays_used':False}
