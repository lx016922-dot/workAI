"""Independent analytical references (written for this verification; does not import the software).

Rayleigh-Lamb dispersion of a traction-free isotropic plate (Lamb 1917; Graff 1975 sec. 8.1; Rose 2014 ch. 6),
in the form that stays real for real or imaginary p, q:
  symmetric      F_S = (q²−k²)² cos(ph) S(q) + 4k²p² S(p) cos(qh) = 0
  antisymmetric  F_A = (q²−k²)² S(p) cos(qh) + 4k²q² cos(ph) S(q) = 0
with p² = ω²/cL² − k², q² = ω²/cT² − k², S(x) = sin(xh)/x, h = half thickness.
"""
import math
import numpy as np


def _cs(x2, h):
    """cos(xh) and sin(xh)/x for x² of either sign."""
    if x2 >= 0:
        x = math.sqrt(x2); return math.cos(x * h), (math.sin(x * h) / x if x > 1e-12 else h)
    x = math.sqrt(-x2); return math.cosh(x * h), (math.sinh(x * h) / x if x > 1e-12 else h)


def rayleigh_lamb(k, w, cl, ct, h, family):
    p2 = (w / cl) ** 2 - k * k; q2 = (w / ct) ** 2 - k * k
    cp_, sp = _cs(p2, h); cq, sq = _cs(q2, h)
    if family == 'S': return (q2 - k * k) ** 2 * cp_ * sq + 4 * k * k * p2 * sp * cq
    return (q2 - k * k) ** 2 * sp * cq + 4 * k * k * q2 * cp_ * sq


def roots(w, cl, ct, h, family, kmax=None, n=20000):
    """All real wavenumbers 0 < k < kmax of one family at angular frequency w, ascending."""
    kmax = kmax or 1.2 * w / (0.5 * ct)
    ks = np.linspace(kmax * 1e-6, kmax, n); f = np.array([rayleigh_lamb(k, w, cl, ct, h, family) for k in ks])
    out = []
    for i in np.flatnonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0):
        a, b = ks[i], ks[i + 1]; fa = f[i]
        for _ in range(200):
            m = .5 * (a + b); fm = rayleigh_lamb(m, w, cl, ct, h, family)
            if np.sign(fm) == np.sign(fa): a, fa = m, fm
            else: b = m
        out.append(.5 * (a + b))
    return out


def fundamental(f_hz, cl, ct, thickness, mode):
    """k (rad/m), phase and group velocity of S0 or A0: the largest real wavenumber of its family."""
    h = thickness / 2; fam = mode[0]
    def k_of(f):
        w = 2 * math.pi * f; return max(roots(w, cl, ct, h, fam))
    k = k_of(f_hz); d = f_hz * 1e-4
    cg = 2 * math.pi * 2 * d / (k_of(f_hz + d) - k_of(f_hz - d))
    return {'k': k, 'cp': 2 * math.pi * f_hz / k, 'cg': cg}


def longitudinal_shg(beta, c, f_hz, a1, x):
    """Classical plane-wave second harmonic (Breazeale & Thompson 1963; Matlack et al. 2015 eq. for β): A2 = β k² x A1² / 8."""
    k = 2 * math.pi * f_hz / c; return beta * k * k * x * a1 * a1 / 8


if __name__ == '__main__':
    # Check against the published table: 2 mm aluminium, cL 6350, cT 3100 m/s (KTU paper; see report).
    for f in (220e3, 380e3):
        print(f, {m: round(fundamental(f, 6350, 3100, 2e-3, m)['cg'], 1) for m in ('A0', 'S0')})
    cl, ct = 6391.54534145318, 3109.126351029605
    print('plate velocity limit', 2 * ct * math.sqrt(1 - ct * ct / cl / cl), 'S0 at 10 kHz', fundamental(1e4, cl, ct, 2e-3, 'S0')['cp'])
