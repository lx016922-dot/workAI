"""Recovered independent momentum-balance checks; never used during inference."""
import numpy as np
import torch
from spectralop.physics import star, internal
from operator3d.physics import U0, T0, pulse


def physics_checks(case,parts,scales):
    targets=np.flatnonzero(~case['fixed'].numpy());patch=star(case,targets);ids=patch['ids'].numpy();dt=case['times'][1]-case['times'][0];rows=[];zero_peak=0.
    with torch.no_grad():
        for frame in np.unique(np.linspace(10,len(parts)-4,9,dtype=int)):
            values=torch.as_tensor(parts[frame-2:frame+3,ids]/U0,dtype=torch.float64);force,strain,jac=internal(values[2],case,patch)
            acceleration=(-values[4]+16*values[3]-30*values[2]+16*values[1]-values[0])[patch['at']]/(12*dt**2)
            target=patch['targets'];ext=case['load'][target]*pulse(torch.tensor(case['times'][frame]),case['scene']['source']);ext=torch.stack([ext,ext*0,ext*0],1)
            normal=scales[:,None]*(2*torch.pi*case['scene']['source']['frequency_mhz'])**2;mass=case['mass'][target];residual=acceleration+(force-ext)/(mass[:,None,None]*(U0/T0**2));squared=(residual/normal).square().mean(2);boundary=case['boundary_nodes'][target]
            def rms(mask):return torch.sqrt((squared[mask]*mass[mask,None]).sum(0)/mass[mask].sum()).tolist() if mask.any() else [0.,0.,0.]
            zero=ext/(mass[:,None,None]*(U0/T0**2))/normal
            if boundary.any():zero_peak=max(zero_peak,float(torch.sqrt((zero[boundary].square().mean(2)*mass[boundary,None]).sum(0)/mass[boundary].sum()).max()))
            rows.append({'time_us':float(case['times'][frame]),'interior_branch_rms':rms(~boundary),'boundary_branch_rms':rms(boundary),'max_strain':float(torch.linalg.matrix_norm(strain,dim=(-2,-1)).max()),'min_J':float(jac.min())})
    return {'checks':rows,'all_free_nodes_checked':len(targets),'time_frames_checked':len(rows),'stencil':'fourth-order five-point central acceleration at output interval','normalization_from_previous_training_only':True,'zero_field_boundary_rms_peak':zero_peak,'zero_field_rejected':zero_peak>.05,'passed':all(max(r['interior_branch_rms']+r['boundary_branch_rms'])<=.05 for r in rows) and zero_peak>.05}

