"""Full-mesh momentum check using exact modal time derivatives.

The force uses independent full Murnaghan Piola evaluation, including terms
omitted by the quadratic propagator. No projection of this residual is taken.
"""
import numpy as np
import torch
from operator3d.data import tensors
from operator3d.physics import internal,pulse,U0,T0

def exact_checks(predicted,acceleration,times,mesh,scene,scales):
    case=tensors(scene,mesh);rows=[];mass=case['mass'];free=~case['fixed'];boundary=case['boundary_nodes']&free;interior=(~case['boundary_nodes'])&free;normal=torch.tensor(scales)[:,None]*(2*np.pi*scene['source']['frequency_mhz'])**2
    with torch.no_grad():
        for i in np.unique(np.linspace(1,len(times)-2,9,dtype=int)):
            force,strain,j=internal(torch.tensor(predicted[i]/U0),case);external=case['load']*pulse(torch.tensor(times[i]),scene['source']);target=torch.stack([external,external*0,external*0],1);residual=torch.tensor(acceleration[i]/U0)+(force-target)/(mass[:,None,None]*(U0/T0**2));squared=((residual/normal)**2).mean((1,2))
            def rms(mask):return float(torch.sqrt((squared[mask]*mass[mask]).sum()/mass[mask].sum())) if mask.any() else 0.
            rows.append({'time_us':float(times[i]),'interior_rms':rms(interior),'boundary_rms':rms(boundary),'max_strain':float(torch.linalg.matrix_norm(strain).max()),'min_J':float(j.min())})
    return rows

def reconstruct_acceleration(prediction,basis,multipliers):
    if prediction['static_correction']:raise ValueError('Static correction does not supply its acceleration.')
    q=prediction['modal_basis_acceleration_m_us2'];coeff=prediction['material_coefficients']*np.asarray(multipliers);parts=np.stack([q[:,0],q[:,1],np.einsum('tkm,k->tm',q[:,2:],coeff)],1)
    return np.einsum('tkm,ncm->tnkc',parts,basis['phi'].reshape(-1,3,len(basis['lambda_us2'])))
