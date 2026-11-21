"""Reference solutions for a damped plane longitudinal wave. Nothing here imports the software's solvers.

The bar with symmetry on its lateral faces is a one-dimensional medium in uniaxial strain,

    ρ ü + α ρ u̇ = ∂ₓ[(1 + β ∂ₜ) P(uₓ)]                               (Rayleigh damping, tangent stiffness)

with the first Piola–Kirchhoff stress of the material in uniaxial strain, E = uₓ + uₓ²/2:

    linear      P = C₁₁ uₓ
    StVK        P = (1 + uₓ) C₁₁ E
    Murnaghan   P = (1 + uₓ) (C₁₁ E + (l + 2m) E²)                     (W = C₁₁E²/2 + (l + 2m)E³/3)

* `wavenumber`: the exact complex wavenumber of the linear equation, k² = (ω/c)² (1 − iα/ω)/(1 + iβω) for e^{i(ωt − kx)}.
  `propagate` carries a recorded linear waveform from one position to another with it.
* `Bar`: an independent numerical solution of the nonlinear equation — one-dimensional spectral elements of degree 6
  with the exact stress above, classical Runge–Kutta in time. Its own convergence is checked by halving the element.
* `second_harmonic`: the classical quasi-monochromatic result with attenuation,
      A₂(x) = (β_nl k² A₁(0)² / 8) (e^{−2a₁x} − e^{−a₂x}) / (a₂ − 2a₁),
  a₁, a₂ the attenuation at f and 2f. The damping drops out of the coupling because it multiplies the whole stress.
"""
import numpy as np


def wavenumber(omega, speed, alpha, beta):
    """Complex k of e^{i(ωt − kx)} travelling towards +x (Im k < 0: decay). ω > 0."""
    omega = np.asarray(omega, dtype=float)
    return omega / speed * np.sqrt((1 - 1j * alpha / omega) / (1 + 1j * beta * omega))


def propagate(times_us, values, distance_m, speed, alpha, beta):
    """The waveform `values` (uniform samples) after `distance_m` more of the damped linear medium."""
    n = len(values); spectrum = np.fft.rfft(values); f = np.fft.rfftfreq(n, (times_us[1] - times_us[0]) * 1e-6); k = np.zeros(len(f), dtype=complex)
    k[1:] = wavenumber(2 * np.pi * f[1:], speed, alpha, beta)
    return np.fft.irfft(spectrum * np.exp(-1j * k * distance_m), n)


def gll(degree):
    """Gauss–Lobatto–Legendre nodes, weights and the derivative matrix on [−1, 1]."""
    from numpy.polynomial import legendre
    interior = legendre.Legendre.basis(degree).deriv().roots(); x = np.concatenate([[-1.], np.sort(interior.real), [1.]])
    P = legendre.legval(x, [0] * degree + [1]); w = 2 / (degree * (degree + 1) * P ** 2)
    D = np.zeros((degree + 1, degree + 1))
    for i in range(degree + 1):
        for j in range(degree + 1):
            if i != j: D[i, j] = P[i] / (P[j] * (x[i] - x[j]))
    D[0, 0] = -degree * (degree + 1) / 4; D[-1, -1] = degree * (degree + 1) / 4
    return x, w, D


class Bar:
    """ρ ü + αρ u̇ = ∂ₓ[(1 + β∂ₜ) P(uₓ)] on 0 ≤ x ≤ L, traction −σ s(t) on x = 0 (pushing along +x), x = L free."""

    def __init__(self, length_m, element_m, material, law, alpha=0., beta=0., degree=6):
        self.law = law; self.alpha, self.beta = alpha, beta; self.rho = material['rho_kg_m3']
        self.c11 = material['lambda_pa'] + 2 * material['mu_pa']; self.third = material['l_pa'] + 2 * material['m_pa']
        elements = int(round(length_m / element_m)); h = length_m / elements; xi, w, D = gll(degree)
        self.ids = np.arange(elements)[:, None] * degree + np.arange(degree + 1)[None, :]; self.x = np.zeros(elements * degree + 1)
        self.x[self.ids] = (np.arange(elements)[:, None] + (xi[None, :] + 1) / 2) * h
        self.D = D * 2 / h; self.w = w * h / 2; self.mass = np.zeros(len(self.x)); np.add.at(self.mass, self.ids, self.rho * self.w[None, :])

    def stress(self, g, rate):
        """P and its rate dP/dt for the displacement gradient g and its rate."""
        if self.law == 'linear': return self.c11 * g, self.c11 * rate
        E = g + g * g / 2; S = self.c11 * E + (self.third * E * E if self.law == 'murnaghan' else 0.)
        dS = self.c11 + (2 * self.third * E if self.law == 'murnaghan' else 0.)
        return (1 + g) * S, (S + (1 + g) ** 2 * dS) * rate

    def acceleration(self, u, v, traction):
        g = u[self.ids] @ self.D.T; rate = v[self.ids] @ self.D.T; P, dP = self.stress(g, rate)
        force = np.zeros(len(u)); np.add.at(force, self.ids, -((P + self.beta * dP) * self.w[None, :]) @ self.D)
        force[0] += traction
        return force / self.mass - self.alpha * v

    def run(self, signal, amplitude_pa, duration_us, sample_us, points_m, substeps):
        """Displacement at `points_m` (nodes nearest to them must coincide: element boundaries) every `sample_us`."""
        taps = [int(np.argmin(np.abs(self.x - p))) for p in points_m]; assert all(abs(self.x[i] - p) < 1e-9 for i, p in zip(taps, points_m)), 'receivers must lie on nodes'
        frames = int(round(duration_us / sample_us)) + 1; dt = sample_us * 1e-6 / substeps; u = np.zeros(len(self.x)); v = np.zeros(len(self.x)); out = np.zeros((frames, len(taps)))
        load = lambda t: amplitude_pa * signal(t * 1e6)
        for frame in range(1, frames):
            for step in range(substeps):
                t = ((frame - 1) * substeps + step) * dt
                k1u, k1v = v, self.acceleration(u, v, load(t))
                k2u, k2v = v + dt / 2 * k1v, self.acceleration(u + dt / 2 * k1u, v + dt / 2 * k1v, load(t + dt / 2))
                k3u, k3v = v + dt / 2 * k2v, self.acceleration(u + dt / 2 * k2u, v + dt / 2 * k2v, load(t + dt / 2))
                k4u, k4v = v + dt * k3v, self.acceleration(u + dt * k3u, v + dt * k3v, load(t + dt))
                u = u + dt / 6 * (k1u + 2 * k2u + 2 * k3u + k4u); v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            out[frame] = u[taps]
        return out


def second_harmonic(x_m, a1_0, k1, beta_nl, a1, a2):
    """|A₂|(x) of the cumulative second harmonic with attenuation a₁ at f and a₂ at 2f (Np/m); a₁ = a₂ = 0 gives β k² A₁² x / 8."""
    x = np.asarray(x_m, dtype=float); m = a2 - 2 * a1
    growth = x * np.exp(-2 * a1 * x) if abs(m) < 1e-12 else (np.exp(-2 * a1 * x) - np.exp(-a2 * x)) / m
    return abs(beta_nl) * k1 * k1 * a1_0 * a1_0 / 8 * growth


if __name__ == '__main__':
    # self-checks: the wavenumber against the dispersion relation; the bar against the exact linear propagation
    c, alpha, beta = 6391.5, 32000., 3.25e-9; w = 2 * np.pi * .5e6; k = wavenumber(w, c, alpha, beta)
    residual = abs(-(w * w) + 1j * alpha * w + c * c * k * k * (1 + 1j * beta * w)) / (w * w); print('dispersion relation residual', residual)
    print('attenuation Np/m exact', -k.imag, 'first order', (alpha / 2 + beta * w * w / 2) / c)
