"""User-defined strain-energy materials: constants, symmetry, checks and scene validation."""
import copy
import math
import unittest

import numpy as np

from physics import materials as M
from physics.scene import validate_scene
from solid.schema import MATERIAL, default_scene

MURNAGHAN = '0.5*(lam+2*mu)*i1**2 - 2*mu*i2 + (l+2*m)/3*i1**3 - 2*m*i1*i2 + n*i3'
AL = {'lam': 58.1, 'mu': 26.1, 'l': -143., 'm': -297., 'n': -345.}


def cfrp(angle=0., cubic=False):
    m = copy.deepcopy(M.presets()['cfrp_ud']['material']); m['fiber_angle_deg'] = angle
    if cubic: m['parameters'].update(a1=-30., a4=-200., a6=-900., a8=15., a9=40.)
    return m


def rotation_x(angle):
    c, s = math.cos(angle), math.sin(angle); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


class Constants(unittest.TestCase):
    def test_murnaghan_written_as_an_energy_reproduces_the_closed_form_and_the_network(self):
        c2e, c3e, s0 = M.taylor_constants(M.compile_energy(MURNAGHAN, AL))
        c2, c3 = M.murnaghan_tensors(dict(MATERIAL, model='murnaghan'))
        self.assertLess(np.max(abs(c2e - c2)) / np.max(abs(c2)), 1e-12); self.assertLess(np.max(abs(c3e - c3)) / np.max(abs(c3)), 1e-10)
        self.assertLess(np.max(abs(s0)), 1e-20)
        from graphnet.model import load
        w = load(3)['weights']
        self.assertLess(np.max(abs(2e11 * w['quadratic_tensor'] - c2)) / np.max(abs(c2)), 1e-12)
        self.assertLess(np.max(abs(6e11 * w['cubic_tensor'] - c3)) / np.max(abs(c3)), 1e-12)
        energy = {'model': 'energy', 'rho_kg_m3': 2700., 'energy': MURNAGHAN, 'parameters': AL}
        table = M.describe(energy)['check_stress_mpa']
        self.assertAlmostEqual(table[0][0][0], 109.563, places=3); self.assertAlmostEqual(table[1][0][1], 52.2, places=3); self.assertAlmostEqual(table[1][2][2], -.249, places=3)
        self.assertAlmostEqual(table[2][0][0], 120.89172, places=4); self.assertAlmostEqual(table[2][1][2], 15.57726, places=4)

    def test_transversely_isotropic_cfrp_has_fibre_axis_symmetry_and_rotates_with_the_fibre_angle(self):
        c2, c3, _ = M.tensors(cfrp(cubic=True))
        for angle in (.3, 1.1):
            Q = M.mandel_rotation(rotation_x(angle))
            self.assertLess(np.max(abs(Q @ c2 @ Q.T - c2)) / np.max(abs(c2)), 1e-9)
            self.assertLess(np.max(abs(np.einsum('ia,jb,kc,abc->ijk', Q, Q, Q, c3) - c3)) / np.max(abs(c3)), 1e-7)
        C2, _ = M.voigt(c2, c3); p = M.presets()['cfrp_ud']['material']['parameters']
        for (i, j), key in {(0, 0): 'C11', (0, 1): 'C12', (1, 1): 'C22', (1, 2): 'C23', (4, 4): 'C55', (5, 5): 'C55'}.items():
            self.assertAlmostEqual(C2[i, j] / 1e9, p[key], places=6)
        self.assertAlmostEqual(C2[3, 3] / 1e9, (p['C22'] - p['C23']) / 2, places=6)
        turned, _, _ = M.tensors(cfrp(90.)); C2t, _ = M.voigt(turned, turned[..., None] * 0 + np.zeros((6, 6, 6)))
        self.assertAlmostEqual(C2t[1, 1], C2[0, 0], delta=1e-6 * C2[0, 0]); self.assertAlmostEqual(C2t[0, 0], C2[1, 1], delta=1e-6 * C2[0, 0])
        speeds = M.describe(cfrp(30.))['speeds_m_s']
        self.assertAlmostEqual(speeds['fiber'][0], math.sqrt(p['C11'] * 1e9 / 1560.), delta=1e-6 * speeds['fiber'][0])
        fast, slow = M.speed_bounds(cfrp()); self.assertGreaterEqual(fast, speeds['fiber'][0] - 1e-6); self.assertLess(slow, math.sqrt(p['C55'] * 1e9 / 1560.))

    def test_engineering_constants_round_trip(self):
        C = M.ti_stiffness_from_engineering(135., 9., 5., .3, .45)
        c2, c3, _ = M.tensors(dict(cfrp(), parameters={**C, **{f'a{k}': 0. for k in range(1, 10)}}))
        S = np.linalg.inv(M.voigt(c2, c3)[0]) * 1e9
        self.assertAlmostEqual(1 / S[0, 0], 135., places=6); self.assertAlmostEqual(1 / S[1, 1], 9., places=6)
        self.assertAlmostEqual(-S[0, 1] / S[0, 0], .3, places=8); self.assertAlmostEqual(-S[1, 2] / S[1, 1], .45, places=8); self.assertAlmostEqual(1 / S[5, 5], 5., places=6)

    def test_truncation_report_for_an_exponential_energy(self):
        m = {'model': 'energy', 'rho_kg_m3': 1500., 'energy': MURNAGHAN + ' + k*(exp(i4) - 1 - i4 - i4**2/2)', 'parameters': dict(AL, k=50.)}
        d = M.describe(m); self.assertLess(d['truncation']['remainder_relative'], 1e-2); self.assertGreater(d['truncation']['remainder_relative'], 0)
        self.assertEqual(d['truncation']['relative_to'], 'cubic')

    def test_invalid_energies_are_refused(self):
        base = {'model': 'energy', 'rho_kg_m3': 1500., 'parameters': {'k': 1.}}
        for expression, word in [('k*E11**2 + E11', '无应力'), ('k*(E11**2', '语法'), ('k*q**2', '未知'), ("__import__('os')", '函数'), ('k.real*E11**2', 'Attribute'),
                                 ('k*E11**2 - 50*E22**2', '正定'), ('k', '应变'), ('abs(E11)', '函数'), ('k*E11**2 if E11 else 0', 'IfExp'), ('[E11]', 'List')]:
            with self.assertRaises(ValueError, msg=expression) as caught: M.validate_material(dict(base, energy=expression))
            self.assertIn(word, str(caught.exception), expression)
        with self.assertRaises(ValueError): M.validate_material(dict(base, energy='k*E11**2', parameters={'E11': 1.}))
        with self.assertRaises(ValueError): M.validate_material(dict(base, energy='k*E11**2', extra=1))
        with self.assertRaises(ValueError): M.validate_material(dict(MATERIAL, model='murnaghan', mu_pa=-1.))


class Scenes(unittest.TestCase):
    def scene(self, material):
        s = default_scene('plate'); s['schema'] = 'workai.hyperelastic.solid.v2'; s['geometry'].update(length_mm=10., width_mm=3., thickness_mm=2.)
        s['source'].update(center_mm=[0., 1.5, 1.], frequency_mhz=.2, cycles=3., delay_us=.2); s['numerics']['h_mm'] = .25
        s['acquisition'] = {'duration_us': 20., 'sample_step_us': .05}; s['material'] = material
        return s

    def test_v2_scenes_carry_strain_energy_materials_and_screen_the_mesh_with_the_slowest_shear_wave(self):
        out = validate_scene(self.scene(cfrp(20.))); self.assertEqual(out['material']['model'], 'energy'); self.assertEqual(out['material']['fiber_angle_deg'], 20.)
        self.assertEqual(validate_scene(self.scene(dict(MATERIAL)))['material'], MATERIAL)
        coarse = self.scene(dict(MATERIAL)); coarse['numerics']['h_mm'] = .45
        validate_scene(coarse)  # aluminium: shear 3109 m/s resolves this mesh
        coarse['material'] = cfrp()
        with self.assertRaises(ValueError) as caught: validate_scene(coarse)
        self.assertIn('网格', str(caught.exception))
        other = self.scene(dict(MATERIAL, l_pa=-1e11))
        with self.assertRaises(ValueError): validate_scene(other)  # legacy keys stay the reference aluminium
        self.assertEqual(validate_scene(self.scene(dict(MATERIAL, model='murnaghan', l_pa=-1e11)))['material']['l_pa'], -1e11)


if __name__ == '__main__': unittest.main()
