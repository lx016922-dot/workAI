"""Validation-only analytic constitutive reference for long Cartesian records.

Uses an analytic force routine verified against the original numerical solver and a fixed fourth-order symmetric
composition. Neither network weights nor network message functions are read.
The reference runs on its own GLL mesh and time step.
"""
import math
import numpy as np
from numba import njit, prange
from .long_reference_force import internal_force


def signal_samples(scene, times):
    tx = scene['source']; kind = scene.get('signal', {}).get('kind', 'hann')
    f0 = tx['frequency_mhz']
    f1 = scene.get('signal', {}).get('end_frequency_mhz', f0) if kind == 'chirp' else f0
    length = 2 * tx['cycles'] / (f0 + f1)
    x = np.asarray(times) - tx['delay_us']
    angle = 2 * np.pi * f0 * x + np.pi * (f1 - f0) / length * x**2 + np.deg2rad(tx['phase_deg'])
    if kind in ('hann', 'chirp'):
        window = np.sin(np.pi * x / length)**2
    elif kind == 'gaussian':
        window = np.exp(-18 * (x / length - .5)**2)
    elif kind == 'rectangular':
        window = np.ones_like(x)
    else:
        raise ValueError('Unknown reference excitation.')
    return np.where((x >= 0) & (x <= length), window * np.sin(angle), 0.)


COMPOSITION = np.array([1 / (2 - 2**(1/3)), -2**(1/3) / (2 - 2**(1/3)), 1 / (2 - 2**(1/3))])


@njit(cache=True)
def segment(u, v, acc, old_force, work, elements, derivative, size, volume, mass,
            constants, load, fixed, law, dt, subdivisions, start, end, initial, pulse):
    fields = np.zeros(((end-start)//subdivisions + int(initial), len(mass), 3))
    energies = np.zeros((len(fields), 4))
    at = int(initial)
    strain_max, j_min = 0., 1.
    for step in range(start+1, end+1):
        for stage in range(3):
            h = COMPOSITION[stage] * dt
            next_u = u + h*v + .5*h*h*acc
            next_u[fixed] = 0.
            force, potential, strain, jac = internal_force(next_u, elements, derivative, size, volume, constants, law)
            strain_max = max(strain_max, strain); j_min = min(j_min, jac)
            if not np.isfinite(potential) or strain > .001 or jac <= 0:
                raise ValueError('Reference strain/J/energy guard failed.')
            ext = load * pulse[step-start-1, stage]
            next_acc = (ext-force)/mass[:,None]; next_acc[fixed] = 0.
            next_v = v + .5*h*(acc+next_acc); next_v[fixed] = 0.
            work += np.sum(.5*(ext+old_force)*(next_u-u))
            u,v,acc,old_force = next_u,next_v,next_acc,ext
        if step % subdivisions == 0:
            fields[at] = u
            energies[at] = np.array([step*dt*1e6, np.sum(.5*mass[:,None]*v*v), potential, work])
            at += 1
    return fields, energies, u,v,acc,old_force,work,strain_max,j_min


@njit(cache=True, parallel=True)
def three_segments(positions, velocities, accelerations, externals, works,
                   elements, derivative, size, volume, mass, constants, load, fixed,
                   dt, subdivisions, start, end, initial, pulse):
    fields = np.empty((3,(end-start)//subdivisions + int(initial),len(mass),3))
    energy = np.empty((3,len(fields[0]),4)); guards = np.empty((3,2))
    for law in prange(3):
        row = segment(positions[law], velocities[law], accelerations[law], externals[law], works[law],
                      elements, derivative, size, volume, mass, constants, load, fixed,
                      law, dt, subdivisions, start, end, initial, pulse)
        fields[law],energy[law] = row[0],row[1]
        positions[law],velocities[law],accelerations[law],externals[law],works[law] = row[2],row[3],row[4],row[5],row[6]
        guards[law] = np.array([row[7],row[8]])
    return fields,energy,guards
