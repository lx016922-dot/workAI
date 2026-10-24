"""Linear transient postprocessing of verified COMSOL eigenpairs.

This is a reference/research tool, never an inference fallback. It is explicitly
a truncated linear modal response, not a native nonlinear COMSOL transient.
Forced motion is integrated at high accuracy, followed by exact unforced modal
evolution. No recorded waveform is repeated, extended with zeroes, or aligned.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from graphwave.modal_calibration import read_modes, digest
from graphwave.signals import specification, pulse_values


def receiver_modes(reference, scene):
    """Use actual COMSOL probe values, or exact matching exported coordinates."""
    points=np.array([r['point_mm'] for r in scene['receivers']])*1e-3
    exported=np.array([r['point_mm'] for r in reference['scene']['receivers']])*1e-3
    direct=reference['receiver_fields']
    if direct.shape[1] and np.array_equal(points,exported):
        return direct,'direct_COMSOL_receiver_interpolation'
    indices=[]
    for point in points:
        distance=np.max(abs(reference['coordinates']-point),axis=1)
        hits=np.flatnonzero(distance<1e-14)
        if len(hits)!=1:
            raise ValueError('Receiver lacks a direct COMSOL modal export; no unverified spatial interpolation is allowed')
        indices.append(int(hits[0]))
    return reference['fields'][:,indices],'exact_matching_COMSOL_export_coordinate'


def oscillator_response(scene, frequency, times_us, *, rtol=1e-11):
    """For each mode, solve x''+omega²*x=omega²*unit_load(t)."""
    omega=2*np.pi*np.asarray(frequency)
    duration=specification(scene)['duration_us']*1e-6
    delay=scene['source']['delay_us']*1e-6
    count=len(omega)
    def rhs(t,y):
        drive=float(pulse_values(scene,(t+delay)*1e6))
        return np.r_[omega*y[count:],omega*(drive-y[:count])]
    # Scaling x and x'/omega keeps tolerances meaningful for every eigenmode.
    solution=solve_ivp(rhs,(0,duration),np.zeros(2*count),method='DOP853',
                       rtol=rtol,atol=rtol*1e-2,max_step=1/(frequency.max()*20),dense_output=True)
    if not solution.success:raise ValueError('Modal forcing integration failed: '+solution.message)
    local=np.asarray(times_us)*1e-6-delay
    values=np.zeros((len(local),count))
    forced=(local>=0)&(local<=duration)
    values[forced]=solution.sol(local[forced])[:count].T
    free=local>duration
    angle=(local[free,None]-duration)*omega[None]
    terminal=solution.y[:,-1]
    values[free]=np.cos(angle)*terminal[None,:count]+np.sin(angle)*terminal[None,count:]
    return values/(omega[None]**2),{'integrator':'DOP853 during forcing; exact modal free evolution afterward',
                                  'relative_tolerance':rtol,'absolute_tolerance_scaled':rtol*1e-2,
                                  'forced_steps':len(solution.t)-1,'rhs_evaluations':solution.nfev}


def response(reference,scene,*,count=None,rtol=1e-11):
    for key in ['geometry','material','boundary','constitutive']:
        if scene[key]!=reference['scene'][key]:raise ValueError('Modal reference changes physical problem: '+key)
    if scene['source']['face']!=reference['scene']['source']['face'] or scene['source']['radius_mm']!=0:
        raise ValueError('Modal force geometry mismatch')
    fields,method=receiver_modes(reference,scene)
    count=len(fields) if count is None else count
    if not 1<=count<=len(fields):raise ValueError('Modal truncation out of range')
    acq=scene['acquisition'];times=np.arange(round(acq['duration_us']/acq['sample_step_us'])+1)*acq['sample_step_us']
    coordinates,details=oscillator_response(scene,reference['frequency_hz'][:count],times,rtol=rtol)
    integrals=reference['integrals'][:count]
    forces=np.stack([integrals['force_'+axis+'_integral'] for axis in 'xyz'],-1)
    direction=np.array(scene['source']['direction'],dtype=float);direction/=np.linalg.norm(direction)
    coupling=(forces@direction)*scene['source']['amplitude_mpa']*1e6/integrals['mass_integral']
    traces=np.einsum('tm,m,mrc->trc',coordinates,coupling,fields[:count],optimize=True)
    if not np.isfinite(traces).all():raise ValueError('Nonfinite modal response')
    return times,traces,dict(details,modes=count,receiver_values=method,
                             highest_frequency_hz=float(reference['frequency_hz'][count-1]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--modes',type=int);p.add_argument('--rtol',type=float,default=1e-11)
    args=p.parse_args();raw=json.loads(args.input.read_text());scene=raw.get('scene',raw)
    reference=read_modes(args.reference)
    times,traces,details=response(reference,scene,count=args.modes,rtol=args.rtol)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,times_us=times,linear=traces)
    record={'schema':'workai.comsol_linear_modal_response.v1','reference_identity':reference['identity'],
            'input_sha256':digest(args.input),'output_sha256':digest(args.output),'duration_us':float(times[-1]),
            'samples':len(times),'method':details,'native_comsol_time_dependent_solve':False,
            'nonlinear_reference':False,'truncation_convergence_verified':False,
            'spatial_convergence_verified':False,'long_record_accuracy_certified':False}
    args.output.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
