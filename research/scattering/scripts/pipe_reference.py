"""Transient waves in a pipe for one circumferential order n: a 2.5D spectral-element solver (independent of the software).

Displacements u_r = U_r(x, r) cos nθ, u_θ = U_θ(x, r) sin nθ, u_x = U_x(x, r) cos nθ reduce linear elastodynamics in a
pipe whose loads share that angular dependence to the (x, r) half-section. The weak form is the section integral of
σ : ε with weight r; strains are ε_rr = ∂_r U_r, ε_θθ = (U_r + n U_θ)/r, ε_xx = ∂_x U_x, γ_rθ = ∂_r U_θ − (U_θ + n U_r)/r,
γ_rx = ∂_x U_r + ∂_r U_x, γ_θx = ∂_x U_θ − n U_x / r. Gauss–Lobatto–Legendre elements of degree p on a structured
x–r grid, lumped mass, assembled sparse stiffness, classical fourth-order Runge–Kutta in time.
A uniform traction on the xmin end face: axial (n = 0, T_x = 1) or along y (n = 1, T_r = 1, T_θ = −1, since
e_y = cos θ e_r − sin θ e_θ with θ measured from +y). All other faces are free.
Usage: pipe_reference.py CASE P HX [DT_FACTOR] [NR]   (writes reference/pipe-CASE-pP-hHX-nrNR-dtD.npy)
       pipe_reference.py check   (section eigenfrequencies with sliding ends vs the SAFE curves)
"""
import math, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from pipe_safe import gll
from pipe import CASES, RO, RI, LENGTH, DURATION, STEP, receivers, line_points

REF = Path(__file__).resolve().parents[1] / 'reference'
LAM, MU, RHO = 58.1e9, 26.1e9, 2700.


def tone_burst(t_us, f_mhz, cycles):
    """Hann-windowed sine, not rescaled (the workbench's tone burst definition: only Gaussian pulses, chirps and multi-tone
    signals are normalised to unit peak; a 3-cycle Hann tone burst peaks at 0.9365)."""
    t = np.asarray(t_us, dtype=float); x = t * f_mhz / cycles
    return np.where((x >= 0) & (x <= 1), .5 * (1 - np.cos(2 * np.pi * x)) * np.sin(2 * np.pi * f_mhz * t), 0.)


class Section:
    def __init__(self, n, p, hx, nr, ri=RI * 1e-3, ro=RO * 1e-3, length=LENGTH * 1e-3):
        self.n, self.p = n, p; x, w, D = gll(p); nx = int(round(length / hx)); hx = length / nx; hr = (ro - ri) / nr
        self.xg = np.concatenate([[0.], *[(e + (x[1:] + 1) / 2) * hx for e in range(nx)]])
        self.rg = np.concatenate([[ri], *[ri + (e + (x[1:] + 1) / 2) * hr for e in range(nr)]])
        Nx, Nr = len(self.xg), len(self.rg); self.Nx, self.Nr = Nx, Nr; N = Nx * Nr; self.N = N
        q = p + 1; loc = np.arange(q); C = np.zeros((6, 6)); C[:3, :3] = LAM; C[[0, 1, 2], [0, 1, 2]] += 2 * MU; C[3:, 3:] = MU * np.eye(3)
        Dx = np.kron(D * 2 / hx, np.eye(q)); Dr = np.kron(np.eye(q), D * 2 / hr); Iq = np.eye(q * q); Z = np.zeros((q * q, q * q))
        mass = np.zeros(N); rows, cols, vals = [], [], []
        for ex in range(nx):
            for er in range(nr):
                gx = ex * p + loc; gr = er * p + loc; ids = (gx[:, None] * Nr + gr[None, :]).ravel()   # local node (a, b) → x index a, r index b
                r = np.tile(self.rg[gr], q); wq = np.outer(w, w).ravel() * hx * hr / 4 * r
                R = np.diag(1 / r)
                # B (6 q² × 3 q²): strain rows at every node, columns U_r, U_θ, U_x
                B = np.block([[Dr, Z, Z], [R, n * R, Z], [Z, Z, Dx], [-n * R, Dr - R, Z], [Dx, Z, Dr], [Z, Dx, -n * R]])
                Bq = B.reshape(6, q * q, 3 * q * q)
                K = np.einsum('q,iqa,ij,jqb->ab', wq, Bq, C, Bq)
                dofs = np.concatenate([ids, ids + N, ids + 2 * N])
                rows.append(np.repeat(dofs, len(dofs))); cols.append(np.tile(dofs, len(dofs))); vals.append(K.ravel())
                np.add.at(mass, ids, RHO * wq)
        self.K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * N, 3 * N))
        self.inv_mass = 1 / np.tile(mass, 3)
        # end-face load: GLL weights along r on x = 0 times r
        face = np.zeros(Nr)
        for er in range(nr): np.add.at(face, er * p + loc, w * hr / 2 * self.rg[er * p + loc])
        self.face = face

    def load_vector(self, traction):
        f = np.zeros(3 * self.N); ids = np.arange(self.Nr)   # x index 0
        for c, t in enumerate(traction): f[c * self.N + ids] = t * self.face
        return f

    def omega_max(self):
        v = np.random.default_rng(0).standard_normal(3 * self.N); s = np.sqrt(self.inv_mass)
        for _ in range(200): v = s * (self.K @ (s * v)); lam = np.linalg.norm(v); v /= lam
        return math.sqrt(lam)

    def probe(self, x, r):
        """Interpolation weights (node ids, weights) at (x, r) in metres: tensor Lagrange basis of the containing element."""
        p = self.p; ex = min(int(np.searchsorted(self.xg[::p], x, side='right')) - 1, (self.Nx - 1) // p - 1); er = min(int(np.searchsorted(self.rg[::p], r, side='right')) - 1, (self.Nr - 1) // p - 1)
        xs = self.xg[ex * p: ex * p + p + 1]; rs = self.rg[er * p: er * p + p + 1]
        def lag(nodes, v): return np.array([np.prod([(v - nodes[m]) / (nodes[j] - nodes[m]) for m in range(len(nodes)) if m != j]) for j in range(len(nodes))])
        lx, lr = lag(xs, x), lag(rs, r); ids = ((ex * p + np.arange(p + 1))[:, None] * self.Nr + (er * p + np.arange(p + 1))[None, :]).ravel()
        return ids, np.outer(lx, lr).ravel()


def simulate(case, p, hx_mm, nr, dt_factor=1.):
    direction, n, f, cycles = CASES[case]
    traction = (0., 0., 1e6) if n == 0 else (1e6, -1e6, 0.)   # Pa: amplitude 1 MPa as in the network scene
    sec = Section(n, p, hx_mm * 1e-3, nr); wmax = sec.omega_max()
    sub = int(math.ceil(STEP * 1e-6 / (dt_factor * 2.5 / wmax))); dt = STEP * 1e-6 / sub   # RK4 stability limit 2.83 / ω_max
    frames = int(round(DURATION / STEP)) + 1; F = sec.load_vector(traction); N3 = 3 * sec.N
    probes = [sec.probe(q[0] * 1e-3, np.hypot(q[1], q[2]) * 1e-3) for _, q in receivers() + line_points()]   # receivers, then the line
    u = np.zeros(N3); v = np.zeros(N3); out = np.zeros((frames, len(probes), 3)); t0 = time.time()
    def acc(u, t_s): return sec.inv_mass * (F * tone_burst(t_s * 1e6, f, cycles) - sec.K @ u)
    for frame in range(1, frames):
        for s in range(sub):
            t = ((frame - 1) * sub + s) * dt
            k1u, k1v = v, acc(u, t); k2u, k2v = v + .5 * dt * k1v, acc(u + .5 * dt * k1u, t + .5 * dt)
            k3u, k3v = v + .5 * dt * k2v, acc(u + .5 * dt * k2u, t + .5 * dt); k4u, k4v = v + dt * k3v, acc(u + dt * k3u, t + dt)
            u = u + dt / 6 * (k1u + 2 * k2u + 2 * k3u + k4u); v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        for j, (ids, w) in enumerate(probes): out[frame, j] = [w @ u[c * sec.N + ids] for c in range(3)]   # U_r, U_θ, U_x
    print(case, f'p {p} hx {hx_mm} nr {nr}: dofs {N3}, ω_max {wmax:.3e}, dt {dt*1e9:.3f} ns, {time.time() - t0:.0f} s', flush=True)
    return out


def cartesian(U, n, theta_deg):
    """(u_x, u_y, u_z) at angle θ (from +y towards +z) from the order-n amplitudes U[..., (U_r, U_θ, U_x)]."""
    th = math.radians(theta_deg); c, s = math.cos(n * th), math.sin(n * th)
    ur, ut, ux = U[..., 0] * c, U[..., 1] * s, U[..., 2] * c
    return np.stack([ux, ur * math.cos(th) - ut * math.sin(th), ur * math.sin(th) + ut * math.cos(th)], -1)


def sliding_end_check(length=20e-3, fmax=1e6):
    """With u_x = 0 on both end faces the standing waves of the section are exactly the guided modes at k = mπ/L, so its
    eigenfrequencies must equal the SAFE dispersion curves (pipe_safe.py). Returns the worst relative mismatch per order."""
    from scipy.sparse.linalg import eigsh
    from pipe_safe import Pipe
    out = {}
    for n in (0, 1):
        sec = Section(n, 6, 1e-3, 1, length=length); N = sec.N; xi = np.arange(N) // sec.Nr; ends = np.flatnonzero((xi == 0) | (xi == sec.Nx - 1))
        fixed = set((2 * N + ends).tolist()) | (set((N + np.arange(N)).tolist()) if n == 0 else set())   # n = 0: drop the unloaded torsional field
        keep = np.array([d for d in range(3 * N) if d not in fixed]); Mi = sp.diags(np.sqrt(sec.inv_mass[keep])); A = (Mi @ sec.K[keep][:, keep] @ Mi).tocsc()
        safe = Pipe(LAM, MU, RHO, sec.rg[0], sec.rg[-1], n, 16); targets = []
        for m in range(1, 40):
            fr, vec = safe.frequencies(m * math.pi / length, 6)
            targets += [fr[b] for b in range(6) if fr[b] / 2 / math.pi < fmax and not (n == 0 and safe.polarization(vec[:, b])[1] > .5)]
        w = np.sqrt(np.sort(np.abs(eigsh(A, k=len(targets) + 4, sigma=0., which='LM', return_eigenvectors=False))))
        out[n] = {'frequencies': len(targets), 'worst_relative_mismatch': float(max(abs(w[np.argmin(abs(w - t))] - t) / t for t in targets))}
    return out


if __name__ == '__main__':
    if sys.argv[1] == 'check':
        # the load history must be the workbench's: compare with physics.signals.evaluate for each case's source
        from physics.signals import evaluate
        from pipe import scene
        t = np.linspace(0, DURATION, 40001)
        for case, (_, _, f, cycles) in CASES.items():
            print(case, 'tone burst vs the workbench signal, max |difference|:', float(np.abs(tone_burst(t, f, cycles) - evaluate(scene(case, 1.25)['source'], t)).max()))
        print('sliding-end eigenfrequencies vs SAFE (k = mπ/L, below 1 MHz):', sliding_end_check()); sys.exit()
    case, p, hx = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]); dtf = float(sys.argv[4]) if len(sys.argv) > 4 else 1.; nr = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    REF.mkdir(exist_ok=True); path = REF / f'pipe-{case}-p{p}-h{hx:g}-nr{nr}-dt{dtf:g}.npy'
    if not path.exists(): np.save(path, simulate(case, p, hx, nr, dtf))
    print('saved', path)
