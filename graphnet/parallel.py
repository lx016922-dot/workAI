"""The long-record layer on all cores, bit for bit the same as `graphnet.recurrence`.

`recurrence.three_segments` gives each constitutive control one thread, so the slowest control sets the time and
the other cores idle. Here every control is spread over the cores instead:

* element messages: the elements of a control are split into contiguous chunks, one task each. A task evaluates
  the frozen pointwise operations (`graphwave.buffered_inference._activation`) and stores the flux and the energy
  of every quadrature point;
* node updates: the nodes are split into contiguous blocks. Each node collects its message from the stored fluxes
  of its elements, in element order and with the same additions as the frozen loop, and then takes the layer
  update.

Nothing is summed in a different order than in the serial code: a node's message is accumulated element by
element exactly as the frozen kernel does, the strain energy is summed point by point, and the work and
dissipation sums run over the loaded and the damped nodes in node order (every other node contributes an exact
zero). The fields, energies and guards are therefore identical to `three_segments` whatever the number of threads
(tested), and the choice between the two is one of speed only.
"""
import numpy as np
from numba import njit, prange
from graphwave.buffered_inference import _activation
from .recurrence import contact_terms

# Cost of one element message of each control (linear, StVK, complete) relative to each other: sizes the element chunks
# so that every task is about the same work.
MESSAGE_COST = (1.6, 1.9, 4.1)
TASKS_PER_THREAD = 8
NODE_BLOCK = 2048
# Whole-layer CPU time of each control relative to the complete one, and what the spread layer costs on top of the
# per-control one (stored fluxes, six parallel regions per layer): measured on 13 689 and 56 457 nodes, docs/large-models.md 7.5.
CONTROL_COST = {'linear': .5, 'stvk': .55, 'murnaghan': 1.}
SPREAD_OVERHEAD = 1.2


def layer_time(controls, threads, spread):
    """Wall time of one layer relative to the complete control alone on one thread.

    Per control (`spread` false): one thread each, so the slowest control or the shared threads set the time.
    Spread: all the work divided over the threads, plus the overhead."""
    costs = [CONTROL_COST[name] for name in controls]
    if spread: return SPREAD_OVERHEAD * sum(costs) / threads
    return max(max(costs), sum(costs) / max(1, min(len(costs), threads)))


def choose(controls, threads, nodes):
    """(spread, threads to use): the faster way to run these controls on this many threads. Both give the same bits."""
    spread = threads >= 2 and nodes * len(controls) >= 1000 and layer_time(controls, threads, True) < layer_time(controls, threads, False)
    return spread, threads if spread else max(1, min(len(controls), threads))


def prepare(mesh, loads, alpha, laws, threads):
    """Index arrays of the parallel layer for this mesh, excitation and thread count."""
    cells = np.asarray(mesh['elements']); nodes = len(mesh['mass']); elements = len(cells); flat = cells.reshape(-1).astype(np.int64)
    entries = np.argsort(flat, kind='stable').astype(np.int64)   # a node's (element, local index) entries, in element order
    pointer = np.zeros(nodes + 1, dtype=np.int64); np.cumsum(np.bincount(flat, minlength=nodes), out=pointer[1:])
    loaded = np.flatnonzero(np.any(np.asarray(loads) != 0, axis=(0, 2))).astype(np.int64)
    absorbing = np.flatnonzero(np.asarray(alpha) != 0).astype(np.int64)
    weights = np.array([MESSAGE_COST[law] for law in laws]); wanted = max(len(laws), TASKS_PER_THREAD * threads)
    rows = []
    for control, weight in enumerate(weights):
        chunks = int(min(elements, max(1, round(wanted * weight / weights.sum())))); edges = np.linspace(0, elements, chunks + 1).astype(np.int64)
        rows += [(control, a, b) for a, b in zip(edges[:-1], edges[1:])]
    blocks = max(1, min(nodes, max(-(-nodes // NODE_BLOCK), TASKS_PER_THREAD * threads // len(laws)))); edges = np.linspace(0, nodes, blocks + 1).astype(np.int64)
    n = cells.shape[1]; controls = len(laws)
    return {'pointer': pointer, 'entries': entries, 'loaded': loaded, 'absorbing': absorbing, 'element_tasks': np.array(rows, dtype=np.int64),
            'node_blocks': np.stack([edges[:-1], edges[1:]], axis=1).astype(np.int64),
            # work arrays kept between calls: two buffers per state array, the message, and the flux and energy of every quadrature point
            'P': np.empty((2, controls, nodes, 3)), 'V': np.empty((2, controls, nodes, 3)), 'A': np.empty((2, controls, nodes, 3)), 'X': np.empty((2, controls, nodes, 3)),
            'message': np.empty((controls, nodes, 3)), 'flux': np.empty((controls, elements, n, n, n, 3, 3)), 'energy': np.empty((controls, elements, n, n, n))}


ARRAYS = ('pointer', 'entries', 'loaded', 'absorbing', 'element_tasks', 'node_blocks', 'P', 'V', 'A', 'X', 'message', 'flux', 'energy')


@njit(cache=True)
def _elements(current, cells, derivative, inverse, volume, quadratic, cubic, law, first, last, flux, energy):
    """Flux and energy of every quadrature point of elements first..last−1 (the frozen pointwise operations); (largest strain, smallest J)."""
    n = cells.shape[1]; strain_max = 0.; jac_min = 1.
    local = np.empty((n, n, n, 3)); ref, g = np.empty((3, 3)), np.empty((3, 3))
    stress, strain, second = np.empty((3, 3)), np.empty((3, 3)), np.empty((3, 3)); z, linear, nonlinear = np.empty(6), np.empty(6), np.empty(6)
    for element in range(first, last):
        ids = cells[element]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for axis in range(3): local[i, j, k, axis] = current[ids[i, j, k], axis]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    ref[:] = 0.
                    for neighbor in range(n):
                        for axis in range(3):
                            ref[axis, 0] += derivative[i, neighbor] * local[neighbor, j, k, axis]
                            ref[axis, 1] += derivative[j, neighbor] * local[i, neighbor, k, axis]
                            ref[axis, 2] += derivative[k, neighbor] * local[i, j, neighbor, axis]
                    transform = inverse[element, i, j, k]
                    for a in range(3):
                        for b in range(3):
                            g[a, b] = 0.
                            for c in range(3): g[a, b] += ref[a, c] * transform[c, b]
                    point, norm, jac = _activation(g, quadratic, cubic, law, strain, z, linear, nonlinear, second, stress)
                    for a in range(3):
                        for b in range(3):
                            value = 0.
                            for c in range(3): value += stress[a, c] * transform[b, c]
                            flux[element, i, j, k, a, b] = value * volume[element, i, j, k]
                    energy[element, i, j, k] = point * volume[element, i, j, k]
                    strain_max = max(strain_max, norm); jac_min = min(jac_min, jac)
    return strain_max, jac_min


@njit(cache=True, parallel=True)
def parallel_segments(positions, velocities, accelerations, externals, works, dissipations, potentials, strains, jacobians,
                      cells, derivative, inverse, volume, mass, loads, free, alpha, damped, quadratic, cubic, composition,
                      dt, subdivisions, start_step, end_step, include_initial, signals, ca, cb, cn, ck, keep, subset, laws,
                      pointer, entries, loaded, absorbing, element_tasks, node_blocks, P, V, A, X, message, flux, energy):
    """`recurrence.three_segments` with every control spread over the threads; same arguments plus the arrays of `prepare` (in the order of `ARRAYS`)."""
    count = (end_step - start_step) // subdivisions + (1 if include_initial else 0); controls = len(laws); stages = len(composition)
    nodes = len(mass); elements = cells.shape[0]; n = cells.shape[1]; n3 = n * n * n; sources = loads.shape[0]; blocks = node_blocks.shape[0]
    fields = np.empty((controls, count, len(keep) if subset else nodes, 3)); energies = np.empty((controls, count, 5)); failed = np.full(controls, -1, dtype=np.int64)
    # Two buffers per state array; `now[c]` says which one holds control c's current state, so a layer ends with a flip, not a copy.
    P[0] = positions; V[0] = velocities; A[0] = accelerations; X[0] = externals
    now = np.zeros(controls, dtype=np.int64); active = np.ones(controls, dtype=np.bool_); at = np.zeros(controls, dtype=np.int64)
    task_strain = np.empty(len(element_tasks)); task_jac = np.empty(len(element_tasks))
    work = works.copy(); dissipated = dissipations.copy(); potential = potentials.copy(); max_strain = strains.copy(); min_j = jacobians.copy()
    if include_initial:
        for c in prange(controls):
            if subset: fields[c, 0] = P[0, c][keep]
            else: fields[c, 0] = P[0, c]
            kinetic = 0.
            for node in range(nodes):
                for axis in range(3): kinetic += .5 * mass[node] * V[0, c, node, axis] * V[0, c, node, axis]
            energies[c, 0, 0] = start_step * dt * 1e6; energies[c, 0, 1] = kinetic
            energies[c, 0, 2] = potential[c]; energies[c, 0, 3] = work[c]; energies[c, 0, 4] = dissipated[c]
            at[c] = 1
    for step in range(start_step + 1, end_step + 1):
        for stage in range(stages):
            h = composition[stage] * dt
            # 1. positions of the layer
            for task in prange(controls * blocks):
                c = task // blocks
                if active[c]:
                    a0 = now[c]; a1 = 1 - a0
                    for node in range(node_blocks[task % blocks, 0], node_blocks[task % blocks, 1]):
                        for axis in range(3):
                            P[a1, c, node, axis] = (P[a0, c, node, axis] + h * V[a0, c, node, axis] + .5 * h * h * A[a0, c, node, axis]) * free[node, axis]
            # 2. element messages: flux and energy of every quadrature point
            for task in prange(len(element_tasks)):
                c = element_tasks[task, 0]
                if active[c]:
                    s, j = _elements(P[1 - now[c], c], cells, derivative, inverse, volume, quadratic, cubic, laws[c], element_tasks[task, 1], element_tasks[task, 2], flux[c], energy[c])
                    task_strain[task] = s; task_jac[task] = j
            # 3. each node collects its message, element by element as the frozen loop does
            for task in prange(controls * blocks):
                c = task // blocks
                if active[c]:
                    for node in range(node_blocks[task % blocks, 0], node_blocks[task % blocks, 1]):
                        m0 = 0.; m1 = 0.; m2 = 0.
                        for q in range(pointer[node], pointer[node + 1]):
                            e = entries[q] // n3; rest = entries[q] % n3; i = rest // (n * n); j = (rest // n) % n; k = rest % n
                            for neighbor in range(n):
                                m0 += (derivative[neighbor, i] * flux[c, e, neighbor, j, k, 0, 0] + derivative[neighbor, j] * flux[c, e, i, neighbor, k, 0, 1]
                                       + derivative[neighbor, k] * flux[c, e, i, j, neighbor, 0, 2])
                                m1 += (derivative[neighbor, i] * flux[c, e, neighbor, j, k, 1, 0] + derivative[neighbor, j] * flux[c, e, i, neighbor, k, 1, 1]
                                       + derivative[neighbor, k] * flux[c, e, i, j, neighbor, 1, 2])
                                m2 += (derivative[neighbor, i] * flux[c, e, neighbor, j, k, 2, 0] + derivative[neighbor, j] * flux[c, e, i, neighbor, k, 2, 1]
                                       + derivative[neighbor, k] * flux[c, e, i, j, neighbor, 2, 2])
                        message[c, node, 0] = m0; message[c, node, 1] = m1; message[c, node, 2] = m2
            # 4. energy, guard and contact of each control (serial sums in the frozen order)
            for c in prange(controls):
                if active[c]:
                    total = 0.
                    for e in range(elements):
                        for i in range(n):
                            for j in range(n):
                                for k in range(n): total += energy[c, e, i, j, k]
                    strain = 0.; jac = 1.
                    for task in range(len(element_tasks)):
                        if element_tasks[task, 0] == c: strain = max(strain, task_strain[task]); jac = min(jac, task_jac[task])
                    if not np.isfinite(total) or strain > .001 or jac <= 0:
                        failed[c] = step; active[c] = False; potential[c] = total; max_strain[c] = max(max_strain[c], strain); min_j[c] = min(min_j[c], jac)
                    else:
                        potential[c] = total + contact_terms(P[1 - now[c], c], message[c], ca, cb, cn, ck)
                        max_strain[c] = max(max_strain[c], strain); min_j[c] = min(min_j[c], jac)
            # 5. velocities and accelerations of the layer
            for task in prange(controls * blocks):
                c = task // blocks
                if active[c]:
                    a0 = now[c]; a1 = 1 - a0
                    for node in range(node_blocks[task % blocks, 0], node_blocks[task % blocks, 1]):
                        for axis in range(3):
                            external = 0.
                            for source in range(sources): external += loads[source, node, axis] * signals[source, step - start_step - 1, stage]
                            undamped = (external - message[c, node, axis]) / mass[node] * free[node, axis]
                            if damped:
                                velocity = (V[a0, c, node, axis] + .5 * h * (A[a0, c, node, axis] + undamped)) / (1 + .5 * h * alpha[node]) * free[node, axis]
                                acceleration = (undamped - alpha[node] * velocity) * free[node, axis]
                            else:
                                acceleration = undamped
                                velocity = (V[a0, c, node, axis] + .5 * h * (A[a0, c, node, axis] + acceleration)) * free[node, axis]
                            X[a1, c, node, axis] = external; V[a1, c, node, axis] = velocity; A[a1, c, node, axis] = acceleration
            # 6. work and dissipation (only loaded and damped nodes contribute; the others add exact zeros), then the layer becomes the state
            for c in prange(controls):
                if active[c]:
                    a0 = now[c]; a1 = 1 - a0
                    if damped:
                        total = 0.
                        for q in range(len(absorbing)):
                            node = absorbing[q]
                            for axis in range(3):
                                total += mass[node] * alpha[node] * .5 * (V[a0, c, node, axis] + V[a1, c, node, axis]) * (P[a1, c, node, axis] - P[a0, c, node, axis])
                        dissipated[c] += total
                    total = 0.
                    for q in range(len(loaded)):
                        node = loaded[q]
                        for axis in range(3): total += .5 * (X[a1, c, node, axis] + X[a0, c, node, axis]) * (P[a1, c, node, axis] - P[a0, c, node, axis])
                    work[c] += total; now[c] = a1
        if step % subdivisions == 0:
            for c in prange(controls):
                if active[c]:
                    a0 = now[c]
                    if subset: fields[c, at[c]] = P[a0, c][keep]
                    else: fields[c, at[c]] = P[a0, c]
                    kinetic = 0.
                    for node in range(nodes):
                        for axis in range(3): kinetic += .5 * mass[node] * V[a0, c, node, axis] * V[a0, c, node, axis]
                    energies[c, at[c], 0] = step * dt * 1e6; energies[c, at[c], 1] = kinetic
                    energies[c, at[c], 2] = potential[c]; energies[c, at[c], 3] = work[c]; energies[c, at[c], 4] = dissipated[c]
                    at[c] += 1
    for c in range(controls):
        positions[c] = P[now[c], c]; velocities[c] = V[now[c], c]; accelerations[c] = A[now[c], c]; externals[c] = X[now[c], c]
        works[c] = work[c]; dissipations[c] = dissipated[c]; potentials[c] = potential[c]; strains[c] = max_strain[c]; jacobians[c] = min_j[c]
    return fields, energies, failed
