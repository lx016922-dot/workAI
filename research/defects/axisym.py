"""Independent axisymmetric reference for a circular delamination: no code shared with the GLL network.

A plate seen from the axis of a normal load on its top face: coordinates (r, z), displacements (u_r, u_z), nothing
depends on the angle. A circular delamination centred on the axis is then a line z = const, 0 ≤ r ≤ a, and this
two-dimensional model is the exact reduction of the three-dimensional problem.

Discretization, as research/defects/fem2d.py: bilinear (Q1) quadrilaterals on a uniform grid, 2 × 2 Gauss
stiffness with the hoop strain u_r / r, row-sum lumped mass (both per radian, weight r), explicit central
differences, small-strain linear elasticity.

* Delamination: the grid nodes on z = at with r < a are duplicated (the elements above take the copies), the axis
  node included; the front node at r = a stays shared.
* Contact (closed delamination): exact frictionless unilateral constraint by the forward-increment
  Lagrange-multiplier method (Carpenter, Taylor & Katona 1991), as in fem2d.py: no penalty stiffness.
* Excitation: normal traction on the top face with a Gaussian profile, A · exp(−r² / 2σ²), consistent nodal forces.
* The axis: u_r = 0. The outer edge r = R is free; R is chosen so that nothing returns from it within the record.

Units: SI internally; mm, μs, MHz, MPa at the interface.
"""
import math
import numpy as np
from numba import njit

GAUSS = (-1 / math.sqrt(3), 1 / math.sqrt(3))
CORNERS = [(-1, -1), (1, -1), (1, 1), (-1, 1)]


def stiffness(r0, h, lam, mu):
    """8 × 8 stiffness per radian of the h × h element whose inner edge is at r0 (dofs u0r, u0z, u1r, …; nodes counter-clockwise from (r0, z0))."""
    D = np.array([[lam + 2 * mu, lam, lam, 0.], [lam, lam + 2 * mu, lam, 0.], [lam, lam, lam + 2 * mu, 0.], [0., 0., 0., mu]])   # r, z, θ, rz
    K = np.zeros((8, 8))
    for xi in GAUSS:
        for eta in GAUSS:
            r = r0 + (1 + xi) * h / 2; B = np.zeros((4, 8))
            for a, (sa, ta) in enumerate(CORNERS):
                N = (1 + sa * xi) * (1 + ta * eta) / 4; dr = sa * (1 + ta * eta) / 4 * 2 / h; dz = ta * (1 + sa * xi) / 4 * 2 / h
                B[0, 2 * a] = dr; B[1, 2 * a + 1] = dz; B[2, 2 * a] = N / r; B[3, 2 * a] = dz; B[3, 2 * a + 1] = dr
            K += B.T @ D @ B * r * h * h / 4
    return K


def build(R, T, h, splits):
    """splits = [{'at': z mm, 'radius': a mm, 'contact': bool}]. Masses per radian and unit density."""
    nr, nz = round(R / h), round(T / h)
    assert abs(nr * h - R) < 1e-9 and abs(nz * h - T) < 1e-9, 'grid must fit the plate'
    node = lambda i, j: i * (nz + 1) + j
    rs, zs = np.meshgrid(np.arange(nr + 1) * h, np.arange(nz + 1) * h, indexing='ij')
    points = list(np.stack([rs.ravel(), zs.ravel()], axis=1))
    elements = np.array([[node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)] for i in range(nr) for j in range(nz)], dtype=np.int64)
    column = np.repeat(np.arange(nr), nz)
    pairs_a, pairs_b = [], []
    for s in splits:
        j, ia = round(s['at'] / h), round(s['radius'] / h)
        assert abs(j * h - s['at']) < 1e-9 and abs(ia * h - s['radius']) < 1e-9 and 0 < j < nz, 'the delamination must lie on grid lines'
        copies = {}
        for i in range(ia):
            old = node(i, j); copies[old] = len(points); points.append(points[old].copy())
        for i in range(ia):
            e = elements[i * nz + j]
            for old, new in copies.items(): e[e == old] = new
        if s.get('contact'):
            for old, new in copies.items(): pairs_a.append(old); pairs_b.append(new)
    points = np.asarray(points) * 1e-3; hm = h * 1e-3
    mass = np.zeros(len(points))
    for a, (sa, ta) in enumerate(CORNERS):   # ∫ N_a r dr dz over the element, exact
        np.add.at(mass, elements[:, a], hm * hm / 4 * (column * hm + hm / 2 * (1 + sa / 3)))
    return {'points': points, 'elements': elements, 'column': column, 'mass_per_rho': mass, 'nr': nr, 'nz': nz, 'h': h,
            'contact_a': np.asarray(pairs_a, dtype=np.int64), 'contact_b': np.asarray(pairs_b, dtype=np.int64)}


@njit(cache=True)
def run(elements, column, K, mass, fixed_r, load, signal, dt, steps, every, probes, ca, cb):
    """Central differences; returns the probe displacements (frames × probes × 2) and the number of closed pairs at every stored frame."""
    n = len(mass); u = np.zeros((n, 2)); prev = np.zeros((n, 2)); f = np.zeros((n, 2))
    frames = steps // every + 1; out = np.zeros((frames, len(probes), 2)); closed = np.zeros(frames, dtype=np.int64); touching = 0
    for step in range(steps + 1):
        f[:] = 0.
        for e in range(len(elements)):
            Ke = K[column[e]]
            for a in range(4):
                na = elements[e, a]
                for b in range(4):
                    nb = elements[e, b]
                    for p in range(2):
                        for q in range(2):
                            f[na, p] += Ke[2 * a + p, 2 * b + q] * u[nb, q]
        if step % every == 0:
            k = step // every; closed[k] = touching
            for r in range(len(probes)):
                out[k, r, 0] = u[probes[r], 0]; out[k, r, 1] = u[probes[r], 1]
        if step == steps: break
        new = np.empty((n, 2))
        for i in range(n):
            new[i, 0] = 0. if fixed_r[i] else 2 * u[i, 0] - prev[i, 0] - dt * dt * f[i, 0] / mass[i]
            new[i, 1] = 2 * u[i, 1] - prev[i, 1] + dt * dt * (load[i] * signal[step] - f[i, 1]) / mass[i]
        touching = 0
        for k in range(len(ca)):   # forward-increment Lagrange multiplier on the normal (z) gap: exact, frictionless, momentum conserving
            a = ca[k]; b = cb[k]; gap = new[b, 1] - new[a, 1]
            if gap < 0.:
                total = mass[a] + mass[b]; new[a, 1] += gap * mass[b] / total; new[b, 1] -= gap * mass[a] / total; touching += 1
        prev = u; u = new
    return out, closed


def simulate(case, h, cfl=.35):
    """case: {'R','T','rho','lam','mu','amplitude_mpa' (positive pushes the top face down),'sigma_mm','f_mhz','cycles','duration_us','sample_us',
    'splits', 'receivers' [(name, r, z)]}."""
    m = build(case['R'], case['T'], h, case['splits'])
    lam, mu, rho = case['lam'], case['mu'], case['rho']; cl = math.sqrt((lam + 2 * mu) / rho); hm = h * 1e-3
    K = np.array([stiffness(i * hm, hm, lam, mu) for i in range(m['nr'])]); mass = m['mass_per_rho'] * rho
    sample = case['sample_us'] * 1e-6; every = math.ceil(sample / (cfl * hm / cl)); dt = sample / every
    steps = round(case['duration_us'] / case['sample_us']) * every
    # top-face traction −A g(r) e_z: consistent nodal forces per radian, ∫ N_i g r dr with 8-point Gauss on every element edge
    xg, wg = np.polynomial.legendre.leggauss(8); sigma = case['sigma_mm'] * 1e-3; load = np.zeros(len(mass)); top = m['nz']
    for i in range(m['nr']):
        r = (i + (1 + xg) / 2) * hm; g = np.exp(-.5 * (r / sigma) ** 2) * r * wg * hm / 2
        load[i * (m['nz'] + 1) + top] -= case['amplitude_mpa'] * 1e6 * np.sum(g * (1 - xg) / 2)
        load[(i + 1) * (m['nz'] + 1) + top] -= case['amplitude_mpa'] * 1e6 * np.sum(g * (1 + xg) / 2)
    t = np.arange(steps + 1) * dt * 1e6; f = case['f_mhz']; span = case['cycles'] / f
    signal = np.where(t <= span, .5 * (1 - np.cos(2 * np.pi * f * t / case['cycles'])) * np.sin(2 * np.pi * f * t), 0.)
    fixed = m['points'][:, 0] < 1e-12
    probes = np.array([round(r / h) * (m['nz'] + 1) + round(z / h) for _, r, z in case['receivers']], dtype=np.int64)
    out, closed = run(m['elements'], m['column'], K, mass, fixed, load, signal, dt, steps, every, probes, m['contact_a'], m['contact_b'])
    return {'times_us': np.arange(out.shape[0]) * case['sample_us'], 'probes_m': out, 'closed_pairs': closed, 'dt_s': dt, 'steps': steps, 'nodes': len(mass),
            'elements': len(m['elements']), 'contact_pairs': len(m['contact_a'])}
