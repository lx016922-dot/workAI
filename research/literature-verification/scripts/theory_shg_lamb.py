"""Cumulative second harmonic of a synchronous Lamb mode pair: an independent semi-analytical reference.

Written for this verification; it does not import the software. Plane strain, isotropic plate with traction-free
faces, Murnaghan material (second-order λ, μ; third-order l, m, n).

Synchronism (Deng 1999; de Lima & Hamilton 2003; Müller et al. 2010; Matlack et al. 2011). A symmetric mode whose
phase velocity equals the longitudinal velocity has p² = ω²/cL² − k² = 0, and the Rayleigh–Lamb equation then
needs sin(qh) = 0 with q = ω √(1/cT² − 1/cL²), h the half thickness:

    f·d = n · cT / √(1 − cT²/cL²),   n = 1, 2, …

n = 1 is S1 and n = 2 is S2 at twice the frequency and twice the wavenumber: the pair S1(f) → S2(2f) is phase
matched, and it is group-velocity matched as well.

Second-order field by perturbation. With u = u¹ + u², u¹ = Re[U¹(z) e^{i(kx−ωt)}] the primary mode and P²(H) the
part of the first Piola–Kirchhoff stress P = (I + H) S(E) that is quadratic in the displacement gradient H,

    ∫ ρ ü²·w + ∫ σ_lin(u²) : ∇w  =  − ∫ P²(∇u¹) : ∇w      for every w

(the traction-free faces are the natural condition of the total stress, so the surface term of de Lima & Hamilton
is contained in this weak form). The 2ω part of P² is Re[½ P²(H̃) e^{2i(kx−ωt)}] with H̃ the complex gradient of
U¹. Discretised through the thickness (semi-analytical finite elements, high-order Lagrange elements on
Gauss–Lobatto nodes) the left side is L(κ) = K(κ) − Ω² M at κ = 2k, Ω = 2ω, which is singular because S2 is a mode
there. A forcing in resonance with a mode makes it grow linearly: with u² = a(x) U²(z) e^{i(2kx−2ωt)},

    da/dx = i · (U²)ᴴ F / ((U²)ᴴ K′(2k) U²),     K′ = dK/dκ,   F = − Σ_q w_q G_q(2k)ᴴ ½P²(H̃_q).

For a plane longitudinal wave the same expression reduces to the textbook |da/dx| = |β| k² A² / 8 with
β = −(3 + C111/C11), C111 = 2l + 4m (checked in `self_test`).

What is compared with the software is the complex ratio of surface displacements

    κ₂ = (da/dx) · U²ₓ(h) / (U¹ₓ(h))²         so that     u²ₓ(h) = Re[κ₂ · x · (U¹ₓ(h) e^{i(kx−ωt)})²],

which does not depend on how the mode shapes are scaled.
"""
import math
import numpy as np

AXES = (0, 2)   # x and z of the 3 × 3 tensors (plane strain: no y motion, no y dependence)


def gll(degree):
    """Gauss–Lobatto–Legendre nodes and weights on [−1, 1] and the derivative matrix D[i, j] = l_j′(x_i)."""
    n = degree; P = np.polynomial.legendre.Legendre.basis(n)
    x = np.concatenate([[-1.], np.sort(P.deriv().roots().real), [1.]]); w = 2 / (n * (n + 1) * P(x) ** 2)
    D = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j: D[i, j] = P(x[i]) / (P(x[j]) * (x[i] - x[j]))
    D[0, 0] = -n * (n + 1) / 4; D[n, n] = n * (n + 1) / 4
    return x, w, D


def synchronism_fd(cl, ct, n=1):
    """Frequency × thickness (Hz·m) of the n-th symmetric mode with phase velocity cL."""
    return n * ct / math.sqrt(1 - (ct / cl) ** 2)


def second_piola(E, lam, mu, l, m, n):
    """S = ∂W/∂E of the Murnaghan energy W = (λ+2μ)/2 I1² − 2μ I2 + (l+2m)/3 I1³ − 2m I1 I2 + n I3 (no conjugation: E may be complex)."""
    I = np.eye(3); i1 = np.trace(E); i2 = .5 * (i1 * i1 - np.trace(E @ E))
    return ((lam + 2 * mu) * i1 * I - 2 * mu * (i1 * I - E) + (l + 2 * m) * i1 * i1 * I - 2 * m * (i2 * I + i1 * (i1 * I - E)) + n * (E @ E - i1 * E + i2 * I))


def quadratic_stress(H, lam, mu, l, m, n):
    """P²(H): the part of P = (I + H) S(E), E = ½(H + Hᵀ + HᵀH), that is quadratic in H."""
    e = .5 * (H + H.T); linear = lambda X: lam * np.trace(X) * np.eye(3) + 2 * mu * X
    third = second_piola(e, lam, mu, l, m, n) - linear(e)          # the quadratic part of S in the linear strain
    return H @ linear(e) + linear(.5 * H.T @ H) + third


class Plate:
    """Semi-analytical finite elements through the thickness of a free plate (z from −h to h)."""

    def __init__(self, thickness, rho, lam, mu, elements=24, degree=8):
        self.rho, self.lam, self.mu = rho, lam, mu; self.h = thickness / 2
        xi, w, D = gll(degree); edges = np.linspace(-self.h, self.h, elements + 1); half = (edges[1] - edges[0]) / 2
        self.nodes = elements * degree + 1; self.z = np.zeros(self.nodes)
        self.points = []   # (weight, node ids, shape values, shape z-derivatives) of every quadrature point (the nodes themselves)
        for e in range(elements):
            ids = e * degree + np.arange(degree + 1); self.z[ids] = edges[e] + (xi + 1) * half
            for q in range(degree + 1):
                N = np.zeros(degree + 1); N[q] = 1.; self.points.append((w[q] * half, ids, N, D[q] / half))
        self.M = np.zeros((2 * self.nodes, 2 * self.nodes))
        for weight, ids, N, _ in self.points:
            for a in (0, 1): self.M[np.ix_(2 * ids + a, 2 * ids + a)] += rho * weight * np.outer(N, N)
        # Hooke's law on the gradient vector (Hxx, Hxz, Hzx, Hzz): P¹ = λ tr(H) I + μ (H + Hᵀ)
        self.C = np.array([[lam + 2 * mu, 0, 0, lam], [0, mu, mu, 0], [0, mu, mu, 0], [lam, 0, 0, lam + 2 * mu]], dtype=float)

    def gradient(self, point, k):
        """G(k): nodal values (uₓ, u_z interleaved) of an element → (Hxx, Hxz, Hzx, Hzz) at a quadrature point, and dG/dk."""
        _, ids, N, dN = point; n = len(ids); G = np.zeros((4, 2 * n), dtype=complex); dG = np.zeros((4, 2 * n), dtype=complex)
        G[0, 0::2] = 1j * k * N; G[1, 0::2] = dN; G[2, 1::2] = 1j * k * N; G[3, 1::2] = dN
        dG[0, 0::2] = 1j * N; dG[2, 1::2] = 1j * N
        return G, dG

    def dofs(self, ids):
        return np.stack([2 * ids, 2 * ids + 1], 1).reshape(-1)

    def stiffness(self, k):
        """K(k) (Hermitian) and dK/dk."""
        n = 2 * self.nodes; K = np.zeros((n, n), dtype=complex); dK = np.zeros((n, n), dtype=complex)
        for point in self.points:
            G, dG = self.gradient(point, k); d = self.dofs(point[1]); weight = point[0]
            K[np.ix_(d, d)] += weight * G.conj().T @ self.C @ G
            dK[np.ix_(d, d)] += weight * (dG.conj().T @ self.C @ G + G.conj().T @ self.C @ dG)
        return K, dK

    def modes(self, k):
        """Angular frequencies (ascending) and mode shapes (columns, uₓ and u_z interleaved) at wavenumber k."""
        K, _ = self.stiffness(k); root = 1 / np.sqrt(np.diag(self.M))          # the mass matrix is diagonal (nodal quadrature)
        values, vectors = np.linalg.eigh(root[:, None] * K * root[None, :])
        return np.sqrt(np.abs(values)), vectors * root[:, None]

    def mode(self, k, omega):
        """The mode at wavenumber k whose frequency is nearest ω: (ω of the mode, shape, group velocity)."""
        w, U = self.modes(k); i = int(np.argmin(np.abs(w - omega))); u = U[:, i]; _, dK = self.stiffness(k)
        group = float(np.real(u.conj() @ dK @ u) / (2 * w[i] * np.real(u.conj() @ self.M @ u)))
        return float(w[i]), u, group

    def forcing(self, U1, k, l, m, n):
        """F at wavenumber 2k for the primary shape U1 at wavenumber k: − Σ w G(2k)ᴴ ½P²(H̃)."""
        F = np.zeros(2 * self.nodes, dtype=complex)
        for point in self.points:
            G, _ = self.gradient(point, k); G2, _ = self.gradient(point, 2 * k); d = self.dofs(point[1]); g = G @ U1[d]
            H = np.zeros((3, 3), dtype=complex); H[0, 0], H[0, 2], H[2, 0], H[2, 2] = g
            P = .5 * quadratic_stress(H, self.lam, self.mu, l, m, n)
            F[d] -= point[0] * G2.conj().T @ np.array([P[0, 0], P[0, 2], P[2, 0], P[2, 2]])
        return F

    def surface(self, U):
        """(uₓ, u_z) at the top face z = h."""
        return U[2 * (self.nodes - 1)], U[2 * (self.nodes - 1) + 1]


def pair(thickness, rho, lam, mu, l, m, n, order=1, elements=24, degree=8):
    """The synchronous pair S_order(f) → S_2·order(2f) of a plate: frequency, wavenumber, velocities and the growth of the second harmonic.

    `growth[part]` is κ₂ (1/m per m of surface uₓ squared) for part = 'geometric' (l = m = n = 0, the Saint Venant–Kirchhoff
    control), 'total' (Murnaghan) and 'material' (their difference)."""
    cl = math.sqrt((lam + 2 * mu) / rho); ct = math.sqrt(mu / rho); f = synchronism_fd(cl, ct, order) / thickness
    omega = 2 * math.pi * f; k = omega / cl; plate = Plate(thickness, rho, lam, mu, elements, degree)
    w1, U1, g1 = plate.mode(k, omega); w2, U2, g2 = plate.mode(2 * k, 2 * omega); _, dK = plate.stiffness(2 * k)
    out = {'f_hz': f, 'k_rad_m': k, 'cl': cl, 'ct': ct, 'omega_error': w1 / omega - 1, 'omega2_error': w2 / (2 * omega) - 1, 'group_1': g1, 'group_2': g2, 'growth': {},
           'surface_uz_over_ux_1': abs(plate.surface(U1)[1] / plate.surface(U1)[0]), 'surface_uz_over_ux_2': abs(plate.surface(U2)[1] / plate.surface(U2)[0]),
           'z': plate.z, 'U1': U1, 'U2': U2}
    for part, constants in (('geometric', (0., 0., 0.)), ('total', (l, m, n))):
        rate = 1j * (U2.conj() @ plate.forcing(U1, k, *constants)) / (U2.conj() @ dK @ U2)
        out['growth'][part] = complex(rate * plate.surface(U2)[0] / plate.surface(U1)[0] ** 2)
    out['growth']['material'] = out['growth']['total'] - out['growth']['geometric']
    return out


def self_test(rho=2700., lam=58.1e9, mu=26.1e9, l=-143e9, m=-297e9, n=-345e9):
    """Checks of this file against closed forms. Returns the largest relative deviations."""
    out = {}
    # 1. The quadratic stress against a finite difference of the full stress P(εH).
    rng = np.random.default_rng(1); H = rng.standard_normal((3, 3)); eps = 1e-4
    full = lambda H: (np.eye(3) + H) @ second_piola(.5 * (H + H.T + H.T @ H), lam, mu, l, m, n)
    numeric = (full(eps * H) + full(-eps * H)) / (2 * eps * eps)        # the even part: the quadratic term, up to O(ε²)
    out['quadratic_stress'] = float(np.abs(numeric - quadratic_stress(H, lam, mu, l, m, n)).max() / np.abs(numeric).max())
    # 2. Plane longitudinal wave: one degree of freedom uₓ, no z dependence. da/dx = i F / K′ must be −β k² A² / 8.
    c11 = lam + 2 * mu; beta = -(3 + (2 * l + 4 * m) / c11); k = 1234.; A = 1e-9
    H = np.zeros((3, 3), dtype=complex); H[0, 0] = 1j * k * A
    F = -np.conj(2j * k) * .5 * quadratic_stress(H, lam, mu, l, m, n)[0, 0]; rate = 1j * F / (2 * (2 * k) * c11)
    out['plane_wave'] = float(abs(rate / (-beta * k * k * A * A / 8) - 1))
    # 3. The plate modes against the Rayleigh–Lamb equation, and the pair's group velocities against each other.
    from theory import rayleigh_lamb
    r = pair(2e-3, rho, lam, mu, l, m, n); scale = (2 * math.pi * r['f_hz'] / r['ct']) ** 6 * 1e-3
    out['rayleigh_lamb_residual'] = max(abs(rayleigh_lamb(j * r['k_rad_m'], j * 2 * math.pi * r['f_hz'], r['cl'], r['ct'], 1e-3, 'S')) / (scale * j ** 6) for j in (1, 2))
    out['mode_frequency'] = max(abs(r['omega_error']), abs(r['omega2_error'])); out['group_mismatch'] = abs(r['group_2'] / r['group_1'] - 1)
    coarse = pair(2e-3, rho, lam, mu, l, m, n, elements=12, degree=6)
    out['discretisation'] = max(abs(coarse['growth'][p] / r['growth'][p] - 1) for p in r['growth'])
    return out, r


if __name__ == '__main__':
    checks, r = self_test()
    print('self test (relative deviations):', {k: f'{v:.2e}' for k, v in checks.items()})
    print(f"2 mm aluminium: f = {r['f_hz'] / 1e6:.5f} MHz (fd {r['f_hz'] * 2e-3 / 1e3:.4f} MHz·mm), k = {r['k_rad_m']:.2f} rad/m, cp = cL = {r['cl']:.1f} m/s, "
          f"group velocity S1 {r['group_1']:.1f}, S2 {r['group_2']:.1f} m/s; surface |u_z/u_x|: S1 {r['surface_uz_over_ux_1']:.1e}, S2 {r['surface_uz_over_ux_2']:.1e}")
    for part, value in r['growth'].items(): print(f'  κ₂ {part:9s} = {abs(value):.4e} 1/m²  (phase {math.degrees(np.angle(value)):+.2f}°)')
