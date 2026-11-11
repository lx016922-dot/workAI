"""Guided modes of a hollow isotropic cylinder by the semi-analytical finite element (SAFE) method (independent of the software).

Displacements u_r = U_r(r) cos nθ, u_θ = U_θ(r) sin nθ, u_x = i Ũ_x(r) cos nθ, all times e^{i(kx − ωt)}. With the factor i on
the axial component every strain amplitude is real, so for each real k the stiffness K(k) = Σ w r Bᵀ D B is real symmetric
and ω² are the eigenvalues of K(k) U = ω² M U. U(r) is a single high-order Gauss–Lobatto–Legendre polynomial across the wall.
Mode families: n = 0 gives the longitudinal L(0, m) (U_r, Ũ_x) and the torsional T(0, m) (U_θ); n = 1 the flexural F(1, m).
"""
import math
import numpy as np
from scipy.linalg import eigh


def gll(p):
    """Gauss–Lobatto–Legendre nodes, weights and differentiation matrix on [−1, 1]."""
    from numpy.polynomial import legendre as L
    x = np.concatenate([[-1.], np.sort(np.real(L.legroots(L.legder([0] * p + [1])))), [1.]])
    P = L.legval(x, [0] * p + [1]); w = 2. / (p * (p + 1) * P ** 2)
    D = np.zeros((p + 1, p + 1))
    for i in range(p + 1):
        for j in range(p + 1):
            if i != j: D[i, j] = P[i] / P[j] / (x[i] - x[j])
    D[0, 0] = -p * (p + 1) / 4.; D[p, p] = p * (p + 1) / 4.
    return x, w, D


class Pipe:
    def __init__(self, lam, mu, rho, ri, ro, n, p=24):
        self.lam, self.mu, self.rho, self.n = lam, mu, rho, n
        x, w, D = gll(p); self.r = ri + (x + 1) * (ro - ri) / 2; self.w = w * (ro - ri) / 2; self.D = D * 2 / (ro - ri); self.N = p + 1
        C = np.zeros((6, 6)); C[:3, :3] = lam; C[[0, 1, 2], [0, 1, 2]] += 2 * mu; C[3:, 3:] = mu * np.eye(3); self.C = C
        N = self.N; m = rho * self.w * self.r; self.M = np.diag(np.concatenate([m, m, m]))
        # strain = (B0 + k B1) U, U = [U_r; U_θ; Ũ_x] at the nodes; rows ε_rr, ε_θθ, ε_xx, γ_rθ, γ_rx/i, γ_θx/i at each node
        I = np.eye(N); R = np.diag(1 / self.r); Z = np.zeros((N, N))
        self.B0 = np.block([[self.D, Z, Z], [R, n * R, Z], [Z, Z, Z], [-n * R, self.D - R, Z], [Z, Z, self.D], [Z, Z, -n * R]])
        self.B1 = np.block([[Z, Z, Z], [Z, Z, Z], [Z, Z, -I], [Z, Z, Z], [I, Z, Z], [Z, I, Z]])

    def stiffness(self, k):
        B = self.B0 + k * self.B1; N = self.N; K = np.zeros((3 * N, 3 * N)); wr = self.w * self.r
        for q in range(N):
            rows = B[q::N]   # the six strains at node q
            K += wr[q] * rows.T @ self.C @ rows
        return K

    def frequencies(self, k, count=6):
        """Lowest `count` angular frequencies (rad/s) and mode shapes at real wavenumber k (rad/m)."""
        vals, vecs = eigh(self.stiffness(k), self.M, subset_by_index=[0, count - 1])
        return np.sqrt(np.maximum(vals, 0.)), vecs

    def polarization(self, vec):
        """Share of kinetic energy in (r, θ, x)."""
        N = self.N; m = self.rho * self.w * self.r; e = np.array([np.sum(m * vec[i * N:(i + 1) * N] ** 2) for i in range(3)]); return e / e.sum()

    def branch(self, f_hz, guess_k, pick):
        """Wavenumber of one mode at frequency f: solve ω(k) = 2πf on the branch chosen by `pick(freqs, vecs) → index`."""
        target = 2 * math.pi * f_hz
        def omega(k):
            fr, vec = self.frequencies(k); return fr[pick(fr, vec, self)]
        a, b = guess_k * .7, guess_k * 1.4
        for _ in range(60):
            mid = .5 * (a + b)
            if omega(mid) > target: b = mid
            else: a = mid
        k = .5 * (a + b); dk = k * 1e-5
        return {'k': k, 'cp': target / k, 'cg': (omega(k + dk) - omega(k - dk)) / (2 * dk)}


def longitudinal_pick(order):
    """L(0, order) in the n = 0 family: modes sorted by frequency, skipping torsional (θ-polarized) ones."""
    def pick(fr, vecs, pipe):
        idx = [i for i in range(len(fr)) if pipe.polarization(vecs[:, i])[1] < .5]; return idx[order - 1]
    return pick


def flexural_pick(order):
    def pick(fr, vecs, pipe): return order - 1
    return pick


if __name__ == '__main__':
    lam, mu, rho = 58.1e9, 26.1e9, 2700.
    E = mu * (3 * lam + 2 * mu) / (lam + mu); cs = math.sqrt(mu / rho); cbar = math.sqrt(E / rho)
    # 1. torsional T(0,1) is non-dispersive at c_s; 2. L(0,1) at low frequency is the bar velocity sqrt(E/ρ)
    pipe = Pipe(lam, mu, rho, 2e-3, 3e-3, 0)
    fr, vec = pipe.frequencies(100.); tors = [i for i in range(len(fr)) if pipe.polarization(vec[:, i])[1] > .5][0]
    print('T(0,1) phase velocity / c_s at k = 100 /m:', fr[tors] / 100 / cs)
    print('L(0,1) at k = 10 /m, phase velocity / sqrt(E/ρ):', pipe.frequencies(10.)[0][longitudinal_pick(1)(*pipe.frequencies(10.), pipe)] / 10 / cbar)
    # 3. thin-wall limit: a 1 mm wall at 1 m radius is a plate; compare with the Rayleigh–Lamb S0 and A0 roots
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'literature-verification' / 'scripts'))
    from theory import fundamental
    cl = math.sqrt((lam + 2 * mu) / rho); big = Pipe(lam, mu, rho, .9995, 1.0005, 0)
    for f in (.3e6, 1e6):
        s0 = big.branch(f, 2 * math.pi * f / 5300, longitudinal_pick(2)); a0 = big.branch(f, 2 * math.pi * f / 2000, longitudinal_pick(1))
        print(f'{f/1e6:g} MHz, 1 mm wall at 1 m radius: L(0,2) cp {s0["cp"]:.2f} vs S0 {fundamental(f, cl, cs, 1e-3, "S0")["cp"]:.2f};'
              f' L(0,1) cp {a0["cp"]:.2f} vs A0 {fundamental(f, cl, cs, 1e-3, "A0")["cp"]:.2f}')
    # 4. convergence in the polynomial degree
    for p in (12, 24):
        q = Pipe(lam, mu, rho, 2e-3, 3e-3, 1, p); print('p', p, 'F(1,1..3) at k = 1000 /m (kHz):', np.round(q.frequencies(1000.)[0][:3] / 2 / math.pi / 1e3, 6))
