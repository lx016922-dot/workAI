"""Frozen graph recurrence with a multi-source, component-masked, damped external channel.

Element messages come from the frozen compiled kernels (`graphwave.inference`
for the short record, `graphwave.buffered_inference` for the long record), so
the learned filters and energy network are evaluated exactly as in the frozen
predictor. Only the external channel and the state update's boundary terms
change:

    external = Σ_k load_k · s_k(t)
    scaled   = (external − message) / m · Δt²          (zeroed on constrained components)
    u⁺       = u + c₁ w₀ (u − u⁻) + c₂ w₁ scaled       c₁ = (1 − αΔt/2)/(1 + αΔt/2), c₂ = 1/(1 + αΔt/2)

With α = 0 the update is the frozen one. The damped form is the central-difference
equivalent of mass-proportional damping m ü + m α u̇ = f with the centred velocity.
"""
import numpy as np
from numba import njit, prange
from graphwave.inference import graph_message
from graphwave.buffered_inference import graph_message as buffered_message


@njit(cache=True)
def forward(cells, derivative, inverse, volume, mass, loads, signals, free, alpha, damped,
            quadratic, cubic, state_weights, law, dt, subdivisions, frames):
    nodes = len(mass); sources = loads.shape[0]
    current = np.zeros((nodes, 3)); previous = np.zeros((nodes, 3)); last_external = np.zeros((nodes, 3)); last_velocity = np.zeros((nodes, 3))
    fields = np.zeros((frames, nodes, 3)); energy_log = np.zeros((frames, 5))
    work = 0.; dissipated = 0.; max_strain = 0.; min_j = 1.
    c1 = (1 - .5 * alpha * dt) / (1 + .5 * alpha * dt); c2 = 1 / (1 + .5 * alpha * dt)
    total_steps = (frames - 1) * subdivisions
    for step in range(total_steps + 1):
        message, potential, strain, jac = graph_message(current, cells, derivative, inverse, volume, quadratic, cubic, law)
        if not np.isfinite(potential) or strain > .001 or jac <= 0:
            raise ValueError('Graph recurrence violates strain/J/finite-energy guard.')
        max_strain = max(max_strain, strain); min_j = min(min_j, jac)
        external = np.zeros((nodes, 3))
        for k in range(sources): external += loads[k] * signals[k, step]
        scaled = (external - message) / mass[:, None] * dt * dt
        scaled *= free
        work += np.sum(.5 * (external + last_external) * (current - previous))
        if damped:
            velocity = ((current - previous) + .5 * scaled) / dt * c2[:, None]
            velocity *= free
            if step > 0: dissipated += np.sum(mass[:, None] * alpha[:, None] * .5 * (last_velocity + velocity) * (current - previous))
            last_velocity = velocity
        if step % subdivisions == 0:
            frame = step // subdivisions; fields[frame] = current
            if not damped:
                velocity = (current - previous + .5 * scaled) / dt
                velocity *= free
            energy_log[frame, 0] = step * dt * 1e6; energy_log[frame, 1] = np.sum(.5 * mass[:, None] * velocity * velocity)
            energy_log[frame, 2] = potential; energy_log[frame, 3] = work; energy_log[frame, 4] = dissipated
        if step < total_steps:
            if damped: next_state = current + state_weights[0] * c1[:, None] * (current - previous) + state_weights[1] * c2[:, None] * scaled
            else: next_state = current + state_weights[0] * (current - previous) + state_weights[1] * scaled
            next_state *= free
            previous = current; current = next_state; last_external = external
    return fields, energy_log, max_strain, min_j


@njit(cache=True)
def _segment(position, velocity, acceleration, previous_external, work, dissipated,
             cells, derivative, inverse, volume, mass, loads, free, alpha, damped, quadratic, cubic, composition,
             law, dt, subdivisions, start_step, end_step, include_initial, signals,
             prior_potential, prior_max_strain, prior_min_j):
    """Long record: the frozen symmetric three-stage composition, each stage a damped velocity-Verlet layer."""
    count = (end_step - start_step) // subdivisions + (1 if include_initial else 0)
    nodes = len(mass); sources = loads.shape[0]
    fields = np.empty((count, nodes, 3)); energies = np.empty((count, 5))
    potential, max_strain, min_j = prior_potential, prior_max_strain, prior_min_j
    at = 0
    if include_initial:
        fields[at] = position
        energies[at, 0] = start_step * dt * 1e6; energies[at, 1] = np.sum(.5 * mass[:, None] * velocity * velocity)
        energies[at, 2] = potential; energies[at, 3] = work; energies[at, 4] = dissipated
        at += 1
    for step in range(start_step + 1, end_step + 1):
        for stage in range(3):
            h = composition[stage] * dt
            next_position = position + h * velocity + .5 * h * h * acceleration
            next_position *= free
            message, potential, strain, jac = buffered_message(next_position, cells, derivative, inverse, volume, quadratic, cubic, law)
            if not np.isfinite(potential) or strain > .001 or jac <= 0:
                raise ValueError('Long recurrence violates strain/J/finite-energy guard.')
            max_strain, min_j = max(max_strain, strain), min(min_j, jac)
            external = np.zeros((nodes, 3))
            for k in range(sources): external += loads[k] * signals[k, step - start_step - 1, stage]
            undamped = (external - message) / mass[:, None]
            undamped *= free
            if damped:
                next_velocity = (velocity + .5 * h * (acceleration + undamped)) / (1 + .5 * h * alpha)[:, None]
                next_velocity *= free
                next_acceleration = undamped - alpha[:, None] * next_velocity
                next_acceleration *= free
                dissipated += np.sum(mass[:, None] * alpha[:, None] * .5 * (velocity + next_velocity) * (next_position - position))
            else:
                next_acceleration = undamped
                next_velocity = velocity + .5 * h * (acceleration + next_acceleration)
                next_velocity *= free
            work += np.sum(.5 * (external + previous_external) * (next_position - position))
            position, velocity, acceleration, previous_external = next_position, next_velocity, next_acceleration, external
        if step % subdivisions == 0:
            fields[at] = position
            energies[at, 0] = step * dt * 1e6; energies[at, 1] = np.sum(.5 * mass[:, None] * velocity * velocity)
            energies[at, 2] = potential; energies[at, 3] = work; energies[at, 4] = dissipated
            at += 1
    return fields, energies, position, velocity, acceleration, previous_external, work, dissipated, potential, max_strain, min_j


@njit(cache=True, parallel=True)
def three_segments(positions, velocities, accelerations, externals, works, dissipations, potentials, strains, jacobians,
                   cells, derivative, inverse, volume, mass, loads, free, alpha, damped, quadratic, cubic, composition,
                   dt, subdivisions, start_step, end_step, include_initial, signals):
    count = (end_step - start_step) // subdivisions + (1 if include_initial else 0)
    fields = np.empty((3, count, len(mass), 3)); energies = np.empty((3, count, 5))
    for law in prange(3):
        row = _segment(positions[law], velocities[law], accelerations[law], externals[law], works[law], dissipations[law],
                       cells, derivative, inverse, volume, mass, loads, free, alpha, damped, quadratic, cubic, composition,
                       law, dt, subdivisions, start_step, end_step, include_initial, signals, potentials[law], strains[law], jacobians[law])
        fields[law], energies[law] = row[0], row[1]
        positions[law], velocities[law], accelerations[law], externals[law] = row[2], row[3], row[4], row[5]
        works[law], dissipations[law], potentials[law], strains[law], jacobians[law] = row[6], row[7], row[8], row[9], row[10]
    return fields, energies
