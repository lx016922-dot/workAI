"""Rayleigh–Lamb roots against closed-form limits, cutoffs and the c_p = c_L synchronism points."""
import math
import unittest
import numpy as np
from solid.schema import MATERIAL
from .lamb import bulk_speeds, rayleigh_speed, dispersion_curves, phase_velocities, mode_at, mode_shape, characteristic
from .matching import synchronous_pairs
from .design import excitation_design

cL, cT = bulk_speeds(MATERIAL)

class LambTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.curves = dispersion_curves(2., MATERIAL, fd_max_mhz_mm=8., points=320)

    def test_limits_and_cutoffs(self):
        c = self.curves; plate = 2 * cT * math.sqrt(1 - cT**2 / cL**2); rayleigh = rayleigh_speed(cL, cT)
        self.assertAlmostEqual(c['plate_speed_m_s'], plate); self.assertLess(rayleigh, cT); self.assertGreater(rayleigh, .85 * cT)
        s0 = mode_at(c, 'S0', .05); self.assertLess(abs(s0['phase_velocity_m_s'] - plate) / plate, 2e-3)
        a0 = mode_at(c, 'A0', .05); young = MATERIAL['mu_pa'] * (3 * MATERIAL['lambda_pa'] + 2 * MATERIAL['mu_pa']) / (MATERIAL['lambda_pa'] + MATERIAL['mu_pa'])
        nu = MATERIAL['lambda_pa'] / (2 * (MATERIAL['lambda_pa'] + MATERIAL['mu_pa'])); d = 2e-3; omega = 2 * math.pi * .05e6 / 2.
        bending = math.sqrt(omega) * (young * d**3 / (12 * (1 - nu**2)) / (MATERIAL['rho_kg_m3'] * d))**.25
        self.assertLess(abs(a0['phase_velocity_m_s'] - bending) / bending, .03)
        high = dispersion_curves(2., MATERIAL, fd_max_mhz_mm=40., fd_min_mhz_mm=30., points=12)
        for name in ('S0', 'A0'):
            cp = mode_at(high, name, 40.)['phase_velocity_m_s']; self.assertLess(abs(cp - rayleigh) / rayleigh, 2e-3, name)
        names = {m['name']: m for m in c['modes']}
        self.assertAlmostEqual(names['A1']['cutoff_fd_mhz_mm'], cT * 1e-3 / 2, delta=.03)        # cos(qh) = 0 at k = 0
        self.assertAlmostEqual(names['S2']['cutoff_fd_mhz_mm'], cL * 1e-3 / 2, delta=.03)        # cos(ph) = 0 at k = 0
        self.assertIn('S1b', names); self.assertTrue(names['S1b']['backward_branch']); self.assertFalse(names['S1']['backward_branch'])
        self.assertAlmostEqual(names['S1']['zero_group_velocity_point_fd_mhz_mm'], 2.87, delta=.03)
        self.assertTrue(all(g < 0 for g in names['S1b']['group_velocity_m_s']))

    def test_roots_satisfy_the_characteristic_equation_and_are_complete(self):
        f = 2e6; omega = 2 * math.pi * f
        for family in ('S', 'A'):
            roots = phase_velocities(f, 2e-3, cL, cT, family)
            self.assertGreaterEqual(len(roots), 2)
            for cp in roots:
                k = omega / cp; scale = max(abs(characteristic(omega, np.array([k * (1 + 1e-3)]), 1e-3, cL, cT, family)[0]), 1e-300)
                self.assertLess(abs(characteristic(omega, np.array([k]), 1e-3, cL, cT, family)[0]) / scale, 1e-6)
        # c_p = c_L on symmetric modes exactly where sin(q h) = 0: fd = n / sqrt(1/cT^2 - 1/cL^2)
        fd_star = 1e-3 / math.sqrt(1 / cT**2 - 1 / cL**2)
        self.assertAlmostEqual(mode_at(self.curves, 'S1', fd_star)['phase_velocity_m_s'], cL, delta=cL * 5e-3)
        self.assertAlmostEqual(mode_at(self.curves, 'S2', 2 * fd_star)['phase_velocity_m_s'], cL, delta=cL * 5e-3)

    def test_mode_shapes_have_the_right_parity_and_quadrature(self):
        z = np.linspace(-1e-3, 1e-3, 21)
        for name, family in (('S0', 'S'), ('A0', 'A'), ('S1', 'S'), ('A1', 'A')):
            at = mode_at(self.curves, name, 4.); shape = mode_shape(2e6, at['phase_velocity_m_s'], 2e-3, cL, cT, family, z)
            ux = np.array(shape['ux_quadrature']); uz = np.array(shape['uz_inphase']); self.assertLess(shape['quadrature_residual'], 1e-9)
            if family == 'S': np.testing.assert_allclose(ux, ux[::-1], atol=1e-9); np.testing.assert_allclose(uz, -uz[::-1], atol=1e-9)
            else: np.testing.assert_allclose(ux, -ux[::-1], atol=1e-9); np.testing.assert_allclose(uz, uz[::-1], atol=1e-9)
            self.assertAlmostEqual(max(np.max(abs(ux)), np.max(abs(uz))), 1.)

    def test_synchronous_pairs_and_design(self):
        pairs = synchronous_pairs(self.curves, tolerance=.01); fd_star = 1e-3 / math.sqrt(1 / cT**2 - 1 / cL**2)
        # S1 (f) and S2 (2f) share c_p = c_L at fd* and 2 fd*; the curves touch, so either a crossing or a tolerance interval must be reported there.
        matches = [r for r in pairs['pairs'] if r['primary'] == 'S1' and r['secondary'] == 'S2' and abs(r['fd_mhz_mm'] - fd_star) < .05]
        self.assertTrue(matches, pairs['pairs']); self.assertAlmostEqual(matches[0]['phase_velocity_m_s'], cL, delta=cL * 5e-3); self.assertLess(abs(matches[0]['phase_velocity_mismatch']), 1e-3)
        low = [r for r in pairs['pairs'] if r['kind'] == 'within_tolerance' and r['primary'] == 'S0' and r['secondary'] == 'S0']
        self.assertTrue(low and low[0]['fd_start_mhz_mm'] <= .1)
        design = excitation_design(self.curves, 'S0', .5, distance_mm=100., cycles=5.)
        self.assertAlmostEqual(design['wavelength_mm'], design['phase_velocity_m_s'] / .5e6 * 1e3); self.assertEqual(design['half_model_boundary']['type'], 'symmetry')
        self.assertGreater(design['wedge_angle_deg'], 20.); self.assertLess(design['direct_window_us'][0], design['group_arrival_us']); self.assertGreater(design['direct_window_us'][1], design['group_arrival_us'] + design['burst_duration_us'])
        self.assertEqual(excitation_design(self.curves, 'A0', .5)['half_model_boundary']['type'], 'antisymmetry')
        with self.assertRaises(ValueError): excitation_design(self.curves, 'S1', .5)

if __name__ == '__main__': unittest.main()
