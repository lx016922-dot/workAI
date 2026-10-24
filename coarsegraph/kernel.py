"""Element message and recurrence with one derivative filter per local axis.

`graph_message` is the frozen kernel's operation order (gather → per-axis filter →
frozen energy activation → tied transpose scatter) with the filter allowed to
differ between the three local axes. With the same filter on all axes it is
bit-identical to `graphwave.inference.graph_message` (tested). The recurrence is
`graphnet.recurrence.forward` with this message.
"""
import numpy as np
from numba import njit
from graphwave.inference import energy_activation


@njit(cache=True)
def graph_message(current, cells, dx, dy, dz, inverse, volume, quadratic, cubic, law):
    message = np.zeros_like(current); potential = 0.; strain_max = 0.; jac_min = 1.; n = cells.shape[1]
    local = np.empty((n, n, n, 3)); flux = np.empty((n, n, n, 3, 3)); ref = np.empty((3, 3))
    for element in range(len(cells)):
        ids = cells[element]
        for i in range(n):
            for j in range(n):
                for k in range(n): local[i, j, k] = current[ids[i, j, k]]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    ref[:] = 0.
                    for neighbor in range(n):
                        for axis in range(3):
                            ref[axis, 0] += dx[i, neighbor] * local[neighbor, j, k, axis]
                            ref[axis, 1] += dy[j, neighbor] * local[i, neighbor, k, axis]
                            ref[axis, 2] += dz[k, neighbor] * local[i, j, neighbor, axis]
                    transform = inverse[element, i, j, k]; g = ref @ transform
                    stress, energy, strain, jac = energy_activation(g, quadratic, cubic, law)
                    flux[i, j, k] = (stress @ transform.T) * volume[element, i, j, k]
                    potential += energy * volume[element, i, j, k]; strain_max = max(strain_max, strain); jac_min = min(jac_min, jac)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    node = ids[i, j, k]
                    for neighbor in range(n):
                        for axis in range(3):
                            message[node, axis] += (dx[neighbor, i] * flux[neighbor, j, k, axis, 0] +
                                                    dy[neighbor, j] * flux[i, neighbor, k, axis, 1] +
                                                    dz[neighbor, k] * flux[i, j, neighbor, axis, 2])
    return message, potential, strain_max, jac_min


@njit(cache=True)
def forward(cells, dx, dy, dz, inverse, volume, mass, loads, signals, free, alpha, damped,
            quadratic, cubic, state_weights, law, dt, subdivisions, frames):
    nodes = len(mass); sources = loads.shape[0]
    current = np.zeros((nodes, 3)); previous = np.zeros((nodes, 3)); last_external = np.zeros((nodes, 3)); last_velocity = np.zeros((nodes, 3))
    fields = np.zeros((frames, nodes, 3)); energy_log = np.zeros((frames, 5))
    work = 0.; dissipated = 0.; max_strain = 0.; min_j = 1.
    c1 = (1 - .5 * alpha * dt) / (1 + .5 * alpha * dt); c2 = 1 / (1 + .5 * alpha * dt)
    total_steps = (frames - 1) * subdivisions
    for step in range(total_steps + 1):
        message, potential, strain, jac = graph_message(current, cells, dx, dy, dz, inverse, volume, quadratic, cubic, law)
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
