"""Independent references for the free–free bar (does not import the software's solvers or network).

exact_linear      — d'Alembert solution with images: a traction burst σ(t) on x = 0, both ends traction-free
                    after the burst; velocity reflects with the same sign at a free end, so
                    u(x,t) = (1/ρc) Σ_n [S(t − (2nL + x)/c) + S(t − (2(n+1)L − x)/c)],  S = ∫σ dt.
solve             — 1D Lagrangian elastodynamics ρ u_tt = ∂P/∂x with the exact uniaxial-strain kinematics of the
                    three controls (P = (1 + u_x) S(E), E = u_x + u_x²/2; linear P = C11 u_x; StVK S = C11 E;
                    Murnaghan S = C11 E + ½ C111 E²), spectral elements of degree 8 and classical RK4, written here
                    for this check. Its own accuracy is shown by refinement and by the exact linear solution.
"""
import math
import numpy as np
from numba import njit


def hann_burst(t, f, cycles):
    tau = t * f / cycles
    return np.where((tau >= 0) & (tau <= 1), .5 * (1 - np.cos(2 * math.pi * tau)) * np.sin(2 * math.pi * f * t), 0.)


def hann_integral(t, f, cycles):
    """∫₀ᵗ of the Hann burst (closed form; zero again after the burst for whole cycles)."""
    w = 2 * math.pi * f; w1, w2 = w * (1 + 1 / cycles), w * (1 - 1 / cycles); T = cycles / f
    def F(x): return -.5 * np.cos(w * x) / w + .25 * (np.cos(w1 * x) / w1 + np.cos(w2 * x) / w2)
    tc = np.clip(t, 0, T); return np.where(t <= 0, 0., F(tc) - F(0.))


def exact_linear(x, t, L, c, rho, sigma, f, cycles):
    """Displacement (m) at positions x (m, array) and times t (s, array): u[t, x]."""
    u = np.zeros((len(t), len(x))); T = cycles / f; n_max = int(math.ceil((t[-1] * c) / (2 * L))) + 1
    for j, xj in enumerate(x):
        for n in range(n_max + 1):
            for delay in ((2 * n * L + xj) / c, (2 * (n + 1) * L - xj) / c):
                if delay > t[-1]: continue
                u[:, j] += hann_integral(t - delay, f, cycles)
    return sigma / (rho * c) * u


def gll(p):
    """GLL nodes, weights and differentiation matrix on [−1, 1]."""
    from numpy.polynomial import legendre as leg
    c = np.zeros(p + 1); c[-1] = 1; x = np.concatenate([[-1.], np.sort(np.real(leg.legroots(leg.legder(c)))), [1.]])
    Pn = leg.legval(x, c); w = 2 / (p * (p + 1) * Pn ** 2); D = np.zeros((p + 1, p + 1))
    for i in range(p + 1):
        for j in range(p + 1):
            if i != j: D[i, j] = Pn[i] / Pn[j] / (x[i] - x[j])
    D[0, 0] = -p * (p + 1) / 4; D[p, p] = p * (p + 1) / 4
    return x, w, D


@njit(cache=True)
def _force(u, D, w, jac, ne, p, law, C11, C111):
    f = np.zeros(len(u))
    for e in range(ne):
        a = e * p
        for q in range(p + 1):
            g = 0.
            for k in range(p + 1): g += D[q, k] * u[a + k]
            g /= jac
            if law == 0: P = C11 * g
            else:
                E = g + .5 * g * g; S = C11 * E + (.5 * C111 * E * E if law == 2 else 0.); P = (1 + g) * S
            for i in range(p + 1): f[a + i] += w[q] * P * D[q, i]
    return f


@njit(cache=True)
def _run(D, w, jac, ne, p, mass, law, C11, C111, dt, steps_per_sample, samples, probe_index, probe_weight, f0, cycles, sigma):
    n = ne * p + 1; u = np.zeros(n); v = np.zeros(n); out = np.zeros((samples, len(probe_index)))
    T = cycles / f0; t = 0.
    for s in range(1, samples):
        for _ in range(steps_per_sample):
            # RK4 on (u, v); traction σ(t) pushes node 0 in +x
            t1, t2, t3 = t, t + .5 * dt, t + dt
            l1 = sigma * (.5 * (1 - math.cos(2 * math.pi * f0 * t1 / cycles)) * math.sin(2 * math.pi * f0 * t1) if t1 <= T else 0.)
            l2 = sigma * (.5 * (1 - math.cos(2 * math.pi * f0 * t2 / cycles)) * math.sin(2 * math.pi * f0 * t2) if t2 <= T else 0.)
            l3 = sigma * (.5 * (1 - math.cos(2 * math.pi * f0 * t3 / cycles)) * math.sin(2 * math.pi * f0 * t3) if t3 <= T else 0.)
            a1 = -_force(u, D, w, jac, ne, p, law, C11, C111); a1[0] += l1; a1 /= mass
            u2 = u + .5 * dt * v; v2 = v + .5 * dt * a1
            a2 = -_force(u2, D, w, jac, ne, p, law, C11, C111); a2[0] += l2; a2 /= mass
            u3 = u + .5 * dt * v2; v3 = v + .5 * dt * a2
            a3 = -_force(u3, D, w, jac, ne, p, law, C11, C111); a3[0] += l2; a3 /= mass
            u4 = u + dt * v3; v4 = v + dt * a3
            a4 = -_force(u4, D, w, jac, ne, p, law, C11, C111); a4[0] += l3; a4 /= mass
            u = u + dt / 6 * (v + 2 * v2 + 2 * v3 + v4); v = v + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4); t += dt
        for j in range(len(probe_index)):
            k = probe_index[j]; out[s, j] = u[k] * (1 - probe_weight[j]) + u[k + 1] * probe_weight[j]
    return out


def solve(law, L, rho, C11, C111, sigma, f, cycles, duration, sample, probes, element=.5e-3, degree=8, steps_per_sample=100):
    """Displacement at probe positions (must be element boundaries), sampled every `sample` seconds; law 0/1/2."""
    x, w, D = gll(degree); ne = int(round(L / element)); jac = L / ne / 2
    mass = np.zeros(ne * degree + 1)
    for e in range(ne): mass[e * degree:e * degree + degree + 1] += rho * w * jac
    nodes = np.concatenate([e * 2 * jac + (x[:-1] + 1) * jac for e in range(ne)] + [[L]])
    index = np.array([int(np.argmin(np.abs(nodes - pr))) for pr in probes]); assert np.allclose(nodes[index], probes, atol=1e-12), 'probes must be nodes'
    samples = int(round(duration / sample)) + 1; dt = sample / steps_per_sample
    return _run(D, w, jac, ne, degree, mass, law, C11, C111, dt, steps_per_sample, samples, index, np.zeros(len(index)), f, cycles, sigma)
