"""Frozen conservative modal time integration with equivalent nodal force assembly.

The integration equations match modal_reduction.propagate at commit 1c200022.
Only the nonlinear-force contraction order differs; no material, quadrature,
time coefficient, reference response or online correction is substituted.
"""
import numpy as np
from .modal_reduction import oscillator_flow
from .modal_nodal_force import nonlinear_force


def propagate_nodal(model, scene, *, step_us, composition=None, keep_coordinates=False, progress=None):
    """Conservative nonlinear kicks around accurately driven candidate flow.

    All three constitutive controls start at zero. The same declared fourth
    order symmetric composition is applied to each. Output covers the complete
    requested interval; no phase alignment, amplitude fit or repeated tail.
    """
    from .signals import specification
    from .modal_forcing import driven_states
    if composition is None:
        a = 1/(2-2**(1/3)); composition = np.array([a, 1-2*a, a])
    composition = np.asarray(composition, dtype=float)
    if composition.shape != (3,) or not np.isfinite(composition).all() or abs(composition.sum()-1) > 1e-12 or abs(composition[0]-composition[-1]) > 1e-12:
        raise ValueError('A consistent symmetric three-stage composition is required')
    acq = scene['acquisition']; sample_us = acq['sample_step_us']; duration_us = acq['duration_us']
    if not np.isfinite(step_us) or not 0 < step_us <= sample_us:
        raise ValueError('Finite positive step no longer than the output interval required')
    subdivisions = round(sample_us/step_us)
    if abs(subdivisions*step_us-sample_us) > 1e-10 or abs(round(duration_us/sample_us)*sample_us-duration_us) > 1e-8:
        raise ValueError('Complete integer time grid required')
    steps = round(duration_us/step_us); frames = round(duration_us/sample_us)+1
    omega, load = model['omega'], model['load']
    q = np.zeros((3, len(omega))); v = np.zeros_like(q); work = np.zeros(3)
    nonlinear = np.zeros_like(q); nonlinear_energy = np.zeros(3)
    max_strain, min_j = np.zeros(3), np.ones(3)
    traces = np.zeros((3, frames, len(scene['receivers']), 3))
    energy_log = np.zeros((3, frames, 3))
    retained = np.zeros((3, frames, len(omega))) if keep_coordinates else None
    # Evolve only the nonlinear increment around the independently evaluated
    # candidate linear solution. External-force kicks are inaccurate for a
    # rectangular burst when the retained spectrum contains high frequencies.
    # No reference arrays or fitted time/amplitude corrections enter this flow.
    stage_offsets = np.r_[0., np.cumsum(composition)]
    pulse_times = (np.arange(steps)[:, None]+stage_offsets[None, :])*step_us
    linear_q,linear_v,forcing_details=driven_states(scene,model['frequency_hz'],pulse_times)
    linear_q*=load;linear_v*=load
    delay=scene['source']['delay_us'];end=delay+specification(scene)['duration_us']
    laws = np.array([1, 2], dtype=np.int64)
    for index in range(steps):
        for stage, weight in enumerate(composition):
            h = weight*step_us*1e-6
            new_v = v-.5*h*nonlinear
            forced=(min(pulse_times[index,stage:stage+2])<end and
                    max(pulse_times[index,stage:stage+2])>delay)
            if forced:
                before=.5*np.sum((new_v+linear_v[index,stage])**2+
                                ((q+linear_q[index,stage])*omega)**2,axis=1)
            q, v = oscillator_flow(q, new_v, omega, h)
            full_q=q+linear_q[index,stage+1]
            if forced:
                after=.5*np.sum((v+linear_v[index,stage+1])**2+(full_q*omega)**2,axis=1)
                work+=after-before
            nforce, nenergy, strain, jac = nonlinear_force(model, full_q[1:], laws)
            nonlinear[1:], nonlinear_energy[1:] = nforce, nenergy
            max_strain[1:] = np.maximum(max_strain[1:], strain); min_j[1:] = np.minimum(min_j[1:], jac)
            v-=.5*h*nonlinear
        if (index+1) % subdivisions == 0:
            at = (index+1)//subdivisions
            full_v=v+linear_v[index,-1]
            traces[:, at] = np.einsum('lm,mrc->lrc', full_q, model['receiver_fields'], optimize=True)
            kinetic = np.sum(.5*full_v*full_v, axis=1)
            potential = np.sum(.5*(full_q*omega)**2, axis=1)+nonlinear_energy
            energy_log[:, at] = np.stack([kinetic, potential, work], axis=-1)
            if retained is not None:retained[:, at] = full_q
            if progress is not None and (at % 100 == 0 or at == frames-1):
                progress({'completed_us':at*sample_us,'duration_us':duration_us})
    # Check the original energy budget per branch, with its own fixed history.
    scale = np.maximum(np.max(abs(energy_log[:, :, 2]), axis=1),
                       np.max(abs(energy_log[:, :, :2].sum(axis=-1)), axis=1))
    error = np.max(abs(energy_log[:, :, :2].sum(axis=-1)-energy_log[:, :, 2]), axis=1)/np.maximum(scale, 1e-30)
    if not np.isfinite(traces).all() or np.any(error > .01):
        raise ValueError('Modal propagation exceeds the 1% energy-balance guard')
    return {'times_us':np.arange(frames)*sample_us, 'traces':traces, 'energy':energy_log,
            'coordinates':retained, 'energy_relative_error':error,
            'maximum_nonlinear_strain':max_strain[1:], 'minimum_nonlinear_j':min_j[1:],
            'modes':len(omega), 'step_us':step_us, 'modal_truncation_convergence_verified':False,
            'forcing_integration':forcing_details,
            'time_convergence_verified':False, 'scientific_accuracy_certified':False}
