"""Constrained element operators for a coarse graph: derivative filter, stiffness quadrature, mass weights.

The learned coarse operator keeps the frozen network's structure (gather →
derivative filter → energy network → tied transpose scatter, explicit state
update) and changes only the per-degree element operators shared by every
element and every direction:

* derivative filter D (n × n, n = p + 1);
* stiffness quadrature weights q (n): the volume weight of a GLL point becomes
  |J| q_i q_j q_k instead of |J| w_i w_j w_k;
* mass weights m (n): the lumped mass becomes ρ |J| m_i m_j m_k assembled.

Hard constraints (by construction, for every parameter value; all hold for GLL):

* reflection symmetry: D[i, j] = −D[n−1−i, n−1−j], q and m symmetric;
* D annihilates constants (rigid translation gives zero strain, zero force);
* D is exact on linear functions (uniform strain and rigid rotation are exact);
* summation by parts with the stiffness weights, Σ_i q_i D[i, j] = δ_jp − δ_j0:
  a uniform stress puts no force on interior nodes (patch test), so a uniform
  strain is an equilibrium state and the long-wavelength stiffness stays exact;
* Σ q = Σ m = 2 (element volume and mass exact).

With the tied transpose and positive q, the stiffness stays symmetric positive
semi-definite, so the scheme keeps a conserved discrete energy. θ = 0 gives the
exact GLL operators; θ lives in the null space of the constraints.
"""
import numpy as np
from numpy.polynomial.legendre import Legendre


def gll(degree):
    """GLL nodes, weights and differentiation matrix on [−1, 1]."""
    p = degree; x = np.r_[-1., np.sort(Legendre.basis(p).deriv().roots()), 1.]; L = Legendre.basis(p)(x)
    w = 2 / (p * (p + 1) * L ** 2); n = p + 1; D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j: D[i, j] = L[i] / (L[j] * (x[i] - x[j]))
    D[0, 0] = -p * (p + 1) / 4; D[-1, -1] = p * (p + 1) / 4
    return x, w, D


def _null_space(A, tol=1e-12):
    _, s, vt = np.linalg.svd(A); rank = int(np.sum(s > tol * max(s.max(), 1)))
    return vt[rank:].T


class Parameterization:
    """θ = (θ_q, θ_m, θ_D) ↔ (D, q, m) for one polynomial degree.

    q and m live in the null space of their own (linear) constraints. D's
    constraints include the q-weighted summation by parts, so its admissible set
    moves with q: D = D_p(q) + N(q) θ_D, where D_p(q) is the smallest correction
    of the GLL matrix that satisfies the constraints for this q and N(q) is a
    fixed reference basis projected onto the constraint null space (continuous
    in q, so θ_D keeps its meaning while q changes).
    """

    def __init__(self, degree):
        self.degree = degree; self.x, self.w, self.D0 = gll(degree); n = degree + 1; self.n = n
        rows = [np.ones(n)]
        for i in range(n // 2):
            a = np.zeros(n); a[i] = 1; a[n - 1 - i] = -1; rows.append(a)
        self.NW = _null_space(np.array(rows))
        A, _ = self._constraints(self.w); self.ND = _null_space(A)
        self.sizes = {'quadrature': self.NW.shape[1], 'mass': self.NW.shape[1], 'derivative': self.ND.shape[1]}
        self.size = sum(self.sizes.values())

    def _constraints(self, q):
        n = self.n; rows = []; rhs = []
        for i in range(n):
            for j in range(n):
                k, l = n - 1 - i, n - 1 - j
                if (i, j) < (k, l):
                    a = np.zeros(n * n); a[i * n + j] = 1; a[k * n + l] = 1; rows.append(a); rhs.append(0.)
                elif (i, j) == (k, l):
                    a = np.zeros(n * n); a[i * n + j] = 1; rows.append(a); rhs.append(0.)
            a = np.zeros(n * n); a[i * n:(i + 1) * n] = 1; rows.append(a); rhs.append(0.)
            a = np.zeros(n * n); a[i * n:(i + 1) * n] = self.x; rows.append(a); rhs.append(1.)
        for j in range(n):
            a = np.zeros(n * n); a[j::n] = q; rows.append(a); rhs.append((1. if j == n - 1 else 0.) - (1. if j == 0 else 0.))
        return np.array(rows), np.array(rhs)

    def unpack(self, theta):
        theta = np.asarray(theta, dtype=float); a, b = self.sizes['quadrature'], self.sizes['mass']
        q = self.w + self.NW @ theta[:a]; m = self.w + self.NW @ theta[a:a + b]
        A, rhs = self._constraints(q); pinv = np.linalg.pinv(A, rcond=1e-12); d0 = self.D0.ravel()
        particular = d0 + pinv @ (rhs - A @ d0); basis = self.ND - pinv @ (A @ self.ND)
        D = (particular + basis @ theta[a + b:]).reshape(self.n, self.n)
        return D, q, m

    def check(self, theta):
        """Constraint residuals for a parameter vector (all ≈ 0 by construction)."""
        D, q, m = self.unpack(theta); n = self.n; flip = np.arange(n)[::-1]; ends = np.zeros(n); ends[0] = -1; ends[-1] = 1
        return {'reflection': float(np.max(np.abs(D + D[np.ix_(flip, flip)]))), 'constants': float(np.max(np.abs(D.sum(1)))),
                'linear': float(np.max(np.abs(D @ self.x - 1))), 'summation_by_parts': float(np.max(np.abs(q @ D - ends))),
                'quadrature_sum': float(abs(q.sum() - 2)), 'mass_sum': float(abs(m.sum() - 2)),
                'symmetric_weights': float(max(np.max(np.abs(q - q[flip])), np.max(np.abs(m - m[flip])))), 'positive': bool(q.min() > 0 and m.min() > 0)}


class AxisParameterization:
    """One independent (D, q, m) block per local element axis (x, y, z on Cartesian meshes); every block satisfies all constraints."""

    def __init__(self, degree):
        self.base = Parameterization(degree); self.degree = degree; self.n = self.base.n; self.x = self.base.x; self.w = self.base.w
        self.sizes = {'per_axis': self.base.sizes, 'axes': 3}; self.size = 3 * self.base.size

    def blocks(self, theta):
        theta = np.asarray(theta, dtype=float); s = self.base.size
        return [theta[a * s:(a + 1) * s] for a in range(3)]

    def unpack(self, theta):
        parts = [self.base.unpack(t) for t in self.blocks(theta)]
        return np.stack([p[0] for p in parts]), np.stack([p[1] for p in parts]), np.stack([p[2] for p in parts])

    def check(self, theta):
        rows = [self.base.check(t) for t in self.blocks(theta)]
        return {k: (all(r[k] for r in rows) if k == 'positive' else max(r[k] for r in rows)) for k in rows[0]}

    def from_shared(self, theta):
        return np.tile(np.asarray(theta, dtype=float), 3)


def parameterization(degree, axes='shared'):
    return AxisParameterization(degree) if axes == 'per_axis' else Parameterization(degree)


def apply(mesh, rho, D, q, m):
    """Element arrays of the coarse graph: derivative filter(s), stiffness volume weights, assembled mass.

    D, q, m are one shared operator (n×n, n, n) or one per local axis (3×n×n, 3×n, 3×n)."""
    degree = int(mesh['degree']); w = gll(degree)[1]
    qs = np.asarray(q) if np.ndim(q) == 2 else np.stack([q, q, q]); ms = np.asarray(m) if np.ndim(m) == 2 else np.stack([m, m, m])
    ratio_q = np.einsum('i,j,k->ijk', qs[0] / w, qs[1] / w, qs[2] / w); ratio_m = np.einsum('i,j,k->ijk', ms[0] / w, ms[1] / w, ms[2] / w)
    volume = np.ascontiguousarray(mesh['volume_weight'] * ratio_q[None])
    mass = np.zeros(len(mesh['points'])); np.add.at(mass, mesh['elements'].ravel(), (mesh['volume_weight'] * ratio_m[None] * rho).ravel())
    return np.ascontiguousarray(D), volume, mass
