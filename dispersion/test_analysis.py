"""Signal-analysis helpers on synthetic data with known answers."""
import math
import unittest
import numpy as np
from .analysis import harmonic_coefficients, line_harmonics, accumulation_fit, amplitude_scaling, line_velocities, fk_spectrum, thickness_decomposition

class AnalysisTests(unittest.TestCase):
    def synthetic(self, beta=2e3, amplitude=1e-6, cp=5000., cg=4500., f=.5, x_mm=np.linspace(40, 80, 41), t_us=np.arange(0, 40, .05)):
        """A2/A1 is about 1e-4 here; the fundamental's window-edge leakage into 2f is near 1e-7 relative."""
        x = x_mm[:, None] * 1e-3; t = t_us[None, :] * 1e-6; k = 2 * np.pi * f * 1e6 / cp; omega = 2 * np.pi * f * 1e6
        envelope = np.exp(-((t - x / cg - 8e-6) / 3e-6)**2)
        a1 = amplitude * np.cos(k * x - omega * t); a2 = beta * amplitude**2 * x * np.cos(2 * (k * x - omega * t))
        return x_mm, t_us, (envelope * (a1 + a2)).T

    def test_harmonic_coefficients_recover_amplitude_and_phase(self):
        t = np.arange(0, 40, .05); f = .5; values = 3e-9 * np.cos(2 * np.pi * f * t + .4) + 5e-11 * np.cos(2 * np.pi * 2 * f * t - 1.)
        h = harmonic_coefficients(t, values, f)['harmonics']
        self.assertAlmostEqual(h[0]['amplitude'] / 3e-9, 1., places=4); self.assertAlmostEqual(math.radians(h[0]['phase_deg']), .4, places=3)
        self.assertAlmostEqual(h[1]['amplitude'] / 5e-11, 1., places=3); self.assertAlmostEqual(math.radians(h[1]['phase_deg']), -1., places=2)
        with self.assertRaises(ValueError): harmonic_coefficients(t, values, f, 10., 5.)

    def test_line_velocities_and_fk_peak(self):
        x, t, traces = self.synthetic(beta=0.)
        v = line_velocities(x, t, traces, .5)
        self.assertLess(abs(v['phase_velocity_m_s'] - 5000.) / 5000., .01); self.assertLess(abs(v['group_velocity_m_s'] - 4500.) / 4500., .05); self.assertGreater(v['phase_fit_r_squared'], .999)
        fk = fk_spectrum(x, t, traces, pad=4); magnitude = fk['magnitude']; i, j = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        self.assertAlmostEqual(fk['frequency_mhz'][i], .5, delta=.03); self.assertAlmostEqual(fk['wavenumber_rad_per_mm'][j], 2 * np.pi * .5e6 / 5000. * 1e-3, delta=.02)
        self.assertGreater(fk['wavenumber_rad_per_mm'][j], 0)

    def test_accumulation_fit_and_amplitude_scaling(self):
        x, t, traces = self.synthetic(beta=2e3, amplitude=1e-6); rows = line_harmonics(x, t, traces, .5, group_velocity_m_s=4500., burst_us=12., delay_us=2.)
        fit = accumulation_fit(rows); ratio = fit['fit_ratio_vs_x']
        # The window factor does not cancel in A2/A1^2, so the slope is proportional to beta (here about 1.9 beta), not equal to it.
        self.assertGreater(ratio['r_squared'], .99); self.assertGreater(ratio['slope_per_m'], 2e3); self.assertLess(ratio['slope_per_m'], 4e3)
        self.assertLess(abs(ratio['intercept']) / (ratio['slope_per_m'] * .06), .05); self.assertGreater(fit['fit_a2_vs_x']['r_squared'], .99)
        _, _, doubled = self.synthetic(beta=2e3, amplitude=2e-6); rows2 = line_harmonics(x, t, doubled, .5, group_velocity_m_s=4500., burst_us=12., delay_us=2.)
        scaling = amplitude_scaling(rows, rows2); self.assertTrue(scaling['all_pass'], scaling['checks'][:2])

    def test_thickness_decomposition_recovers_known_mixture(self):
        z = np.linspace(-1, 1, 15)
        s0 = {'name': 'S0', 'complex_ux': 1j * np.cosh(.3 * z), 'complex_uz': -.2 * np.sinh(.3 * z)}; a0 = {'name': 'A0', 'complex_ux': 1j * .4 * np.sinh(.5 * z), 'complex_uz': np.cosh(.5 * z)}
        ux = 2 * s0['complex_ux'] + .5j * a0['complex_ux']; uz = 2 * s0['complex_uz'] + .5j * a0['complex_uz']
        result = thickness_decomposition(ux, uz, [s0, a0])
        np.testing.assert_allclose(result['amplitudes'], [2, .5j], atol=1e-12); self.assertLess(result['relative_residual'], 1e-12)

if __name__ == '__main__': unittest.main()
