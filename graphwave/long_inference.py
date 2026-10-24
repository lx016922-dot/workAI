"""Chunked learned graph propagation with a trained symmetric time layer."""
import numpy as np
from numba import njit, prange
from .buffered_inference import graph_message


@njit(cache=True)
def _segment(position, velocity, acceleration, previous_external, work,
             cells, derivative, inverse, volume, mass, load, fixed, quadratic, cubic, composition,
             law, dt, subdivisions, start_step, end_step, include_initial, pulse_values,
             prior_potential, prior_max_strain, prior_min_j):
    count = (end_step - start_step) // subdivisions + (1 if include_initial else 0)
    fields = np.empty((count, len(mass), 3))
    energies = np.empty((count, 4))
    potential, max_strain, min_j = prior_potential, prior_max_strain, prior_min_j
    at = 0
    if include_initial:
        fields[at] = position
        energies[at] = np.array([start_step * dt * 1e6, np.sum(.5 * mass[:, None] * velocity * velocity), potential, work])
        at += 1
    for step in range(start_step + 1, end_step + 1):
        for stage in range(3):
            h = composition[stage] * dt
            next_position = position + h * velocity + .5 * h * h * acceleration
            next_position[fixed] = 0.
            message, potential, strain, jac = graph_message(next_position, cells, derivative, inverse, volume, quadratic, cubic, law)
            if not np.isfinite(potential) or strain > .001 or jac <= 0:
                raise ValueError('Long recurrence violates strain/J/finite-energy guard.')
            max_strain, min_j = max(max_strain, strain), min(min_j, jac)
            external = load * pulse_values[step - start_step - 1, stage]
            next_acceleration = (external - message) / mass[:, None]
            next_acceleration[fixed] = 0.
            next_velocity = velocity + .5 * h * (acceleration + next_acceleration)
            next_velocity[fixed] = 0.
            work += np.sum(.5 * (external + previous_external) * (next_position - position))
            position, velocity, acceleration, previous_external = next_position, next_velocity, next_acceleration, external
        if step % subdivisions == 0:
            fields[at] = position
            energies[at] = np.array([step * dt * 1e6, np.sum(.5 * mass[:, None] * velocity * velocity), potential, work])
            at += 1
    return fields, energies, position, velocity, acceleration, previous_external, work, potential, max_strain, min_j


@njit(cache=True, parallel=True)
def three_segments(positions, velocities, accelerations, externals, works, potentials, strains, jacobians,
                   cells, derivative, inverse, volume, mass, load, fixed, quadratic, cubic, composition,
                   dt, subdivisions, start_step, end_step, include_initial, pulse_values):
    count = (end_step - start_step) // subdivisions + (1 if include_initial else 0)
    fields = np.empty((3, count, len(mass), 3))
    energies = np.empty((3, count, 4))
    for law in prange(3):
        row = _segment(positions[law], velocities[law], accelerations[law], externals[law], works[law],
                       cells, derivative, inverse, volume, mass, load, fixed, quadratic, cubic, composition,
                       law, dt, subdivisions, start_step, end_step, include_initial, pulse_values,
                       potentials[law], strains[law], jacobians[law])
        fields[law], energies[law] = row[0], row[1]
        positions[law], velocities[law], accelerations[law], externals[law] = row[2], row[3], row[4], row[5]
        works[law], potentials[law], strains[law], jacobians[law] = row[6], row[7], row[8], row[9]
    return fields, energies
