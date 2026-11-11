"""Plane-stress scattering of a plane compressional wave by a traction-free circular hole (independent of the software).

Low-frequency S0 in a thin plate is a 2D compressional wave with the plane-stress constants λ' = 2λμ/(λ + 2μ), μ
(Diligent et al. 2002 use the same membrane model). With time factor e^{−iωt}, incident φ_i = e^{iαx}, the scattered
potentials are φ_s = Σ A_n H_n(αr) cos nθ, ψ_s = Σ B_n H_n(βr) sin nθ (H = Hankel of the first kind), and the two
traction conditions σ_rr = σ_rθ = 0 at r = a give A_n, B_n mode by mode (Pao & Mow 1973, ch. 4).
"""
import math
import numpy as np
from scipy.special import jv, jvp, hankel1, h1vp


class Hole:
    def __init__(self, lam, mu, rho, a, nmax=40):
        self.lp = 2 * lam * mu / (lam + 2 * mu); self.mu = mu; self.rho = rho; self.a = a; self.nmax = nmax
        self.cp = math.sqrt((self.lp + 2 * mu) / rho); self.cs = math.sqrt(mu / rho)
        self.nu = self.lp / (self.lp + 2 * mu)   # plane-stress Poisson ratio = λ/(2(λ+μ))

    def _fields(self, n, r, f, f1, f2, g, g1, g2):
        """u_r, u_θ (angular factors cos nθ, sin nθ) and σ_rr, σ_rθ from the radial functions of φ (f) and ψ (g)."""
        lp, mu = self.lp, self.mu
        ur = f1 + n * g / r; ut = -n * f / r - g1
        err = f2 + n * (g1 / r - g / r ** 2)
        ett = f1 / r + n * g / r ** 2 - n * n * f / r ** 2 - n * g1 / r
        grt = -2 * n * f1 / r + 2 * n * f / r ** 2 - n * n * g / r ** 2 - g2 + g1 / r
        return ur, ut, (lp + 2 * mu) * err + lp * ett, mu * grt, lp * err + (lp + 2 * mu) * ett

    def coefficients(self, w):
        al, be, a = w / self.cp, w / self.cs, self.a; out = []
        for n in range(self.nmax + 1):
            eps = (1 if n == 0 else 2) * 1j ** n
            inc = self._fields(n, a, eps * jv(n, al * a), eps * al * jvp(n, al * a), eps * al ** 2 * jvp(n, al * a, 2), 0, 0, 0)
            pa = self._fields(n, a, hankel1(n, al * a), al * h1vp(n, al * a), al ** 2 * h1vp(n, al * a, 2), 0, 0, 0)
            sb = self._fields(n, a, 0, 0, 0, hankel1(n, be * a), be * h1vp(n, be * a), be ** 2 * h1vp(n, be * a, 2))
            M = np.array([[pa[2], sb[2]], [pa[3], sb[3]]]); rhs = -np.array([inc[2], inc[3]])
            out.append(np.linalg.solve(M, rhs))
        return out

    def displacement(self, w, r, theta, scattered_only=False):
        """(u_x, u_y) at (r, θ) for an incident φ_i = e^{iαx}, i.e. incident u_x = iα e^{iαx}."""
        al, be = w / self.cp, w / self.cs; coef = self.coefficients(w); ur = 0j; ut = 0j
        for n, (A, B) in enumerate(coef):
            eps = (1 if n == 0 else 2) * 1j ** n
            f = A * hankel1(n, al * r) + (0 if scattered_only else eps * jv(n, al * r))
            f1 = A * al * h1vp(n, al * r) + (0 if scattered_only else eps * al * jvp(n, al * r))
            g = B * hankel1(n, be * r); g1 = B * be * h1vp(n, be * r)
            fr = self._fields(n, r, f, f1, 0, g, g1, 0); ur += fr[0] * math.cos(n * theta); ut += fr[1] * math.sin(n * theta)
        return ur * math.cos(theta) - ut * math.sin(theta), ur * math.sin(theta) + ut * math.cos(theta)

    def hoop_stress_ratio(self, w, theta):
        """σ_θθ at the hole edge over the incident σ_xx amplitude ((λ'+2μ)α² for φ_i = e^{iαx})."""
        al, be, a = w / self.cp, w / self.cs, self.a; coef = self.coefficients(w); s = 0j
        for n, (A, B) in enumerate(coef):
            eps = (1 if n == 0 else 2) * 1j ** n
            f = A * hankel1(n, al * a) + eps * jv(n, al * a); f1 = A * al * h1vp(n, al * a) + eps * al * jvp(n, al * a)
            f2 = A * al ** 2 * h1vp(n, al * a, 2) + eps * al ** 2 * jvp(n, al * a, 2)
            g = B * hankel1(n, be * a); g1 = B * be * h1vp(n, be * a); g2 = B * be ** 2 * h1vp(n, be * a, 2)
            s += self._fields(n, a, f, f1, f2, g, g1, g2)[4] * math.cos(n * theta)
        return s / ((self.lp + 2 * self.mu) * al ** 2)


if __name__ == '__main__':
    lam, mu, rho = 58.1e9, 26.1e9, 2700.; hole = Hole(lam, mu, rho, 4e-3)
    print('plane-stress c_p', round(hole.cp, 2), 'c_s', round(hole.cs, 2), 'ν', round(hole.nu, 4))
    for ka in (1e-3, .05, .5, 1., 2.):
        w = ka / hole.a * hole.cp; print(f'ka={ka:<5} |σθθ/σ0| at 90°: {abs(hole.hoop_stress_ratio(w, math.pi / 2)):.4f}   (Kirsch, biaxial plane wave: 3 − ν = {3 - hole.nu:.4f})')
    # boundary check: tractions vanish on r = a for the total field
    w = 1. / hole.a * hole.cp; coef = hole.coefficients(w); worst = 0.
    for n, (A, B) in enumerate(coef):
        eps = (1 if n == 0 else 2) * 1j ** n; al, be, a = w / hole.cp, w / hole.cs, hole.a
        f = A * hankel1(n, al * a) + eps * jv(n, al * a); f1 = A * al * h1vp(n, al * a) + eps * al * jvp(n, al * a); f2 = A * al ** 2 * h1vp(n, al * a, 2) + eps * al ** 2 * jvp(n, al * a, 2)
        g = B * hankel1(n, be * a); g1 = B * be * h1vp(n, be * a); g2 = B * be ** 2 * h1vp(n, be * a, 2)
        fr = hole._fields(n, a, f, f1, f2, g, g1, g2); worst = max(worst, abs(fr[2]) / ((hole.lp + 2 * hole.mu) * al ** 2), abs(fr[3]) / ((hole.lp + 2 * hole.mu) * al ** 2))
    print('max |traction| on the hole / incident stress:', f'{worst:.1e}')
    # series convergence at the top of the band used in analyze_hole.py (3 MHz) and the farthest receiver (r = 2a)
    w = 2 * math.pi * 3e6; worst = 0.
    for th in (0., 90., 180.):
        a40 = np.array(hole.displacement(w, 2 * hole.a, math.radians(th))); a80 = np.array(Hole(lam, mu, rho, 4e-3, nmax=80).displacement(w, 2 * hole.a, math.radians(th)))
        worst = max(worst, np.abs(a40 - a80).max() / np.abs(a80).max())
    print('3 MHz, r = 2a: nmax 40 vs 80 relative difference', f'{worst:.1e}')
