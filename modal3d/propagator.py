"""Exact oscillator response for piecewise-linear input forces.

Trapezoidal convolution biases the static response of retained high modes.
This recurrence integrates each linearly interpolated force interval exactly.
"""
import numpy as np

def linear_force_oscillators(forces,times,lambdas):
    dt=times[1]-times[0];theta=np.sqrt(lambdas)*dt;c=np.cos(theta);sinc=np.sinc(theta/np.pi);velocity=np.zeros_like(lambdas);position=np.zeros_like(lambdas);out=np.empty_like(forces);out[0]=0.
    i0=.5*dt*dt*np.sinc(theta/(2*np.pi))**2;a=np.empty_like(theta);small=abs(theta)<1e-3
    a[small]=dt*dt*(1/6-theta[small]**2/120+theta[small]**4/5040)
    a[~small]=dt*dt*(1-sinc[~small])/theta[~small]**2
    for i in range(1,len(times)):
        next_position=c*position+dt*sinc*velocity+(i0-a)*forces[i-1]+a*forces[i]
        velocity=-lambdas*dt*sinc*position+c*velocity+(dt*sinc-i0/dt)*forces[i-1]+i0/dt*forces[i]
        position=next_position;out[i]=position
    return out
