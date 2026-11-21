"""Guided waves of a plate (offline/guided.py): the modes against the Rayleigh–Lamb equations and the SH formula, the
group velocity, the phase-matched pair S1–S2, the symmetry classes of an anisotropic plate and what a model gets."""

import math
import unittest
import numpy as np
from dispersion.lamb import bulk_speeds, phase_velocities
from physics.materials import presets, stiffness4, tensors
from solid.schema import MATERIAL
from .domain import templates
from .guided import Plate, describe, elements_for, plate_of, validity

C = stiffness4(tensors(MATERIAL)[0])
RHO = MATERIAL['rho_kg_m3']
CL, CT = bulk_speeds(MATERIAL)


def plate(thickness, top_hz, angle=0.0, c=C, rho=RHO):
    return Plate(thickness, rho, c, angle, elements_for(thickness, CT, top_hz))


class Isotropic(unittest.TestCase):
    def test_lamb_and_sh_modes_agree_with_the_closed_forms(self):
        d = 2e-3
        for f in (0.2e6, 1e6, 3.5585e6):
            modes = plate(d, f).at(f)
            omega = 2 * math.pi * f
            for family in 'SA':
                expected = phase_velocities(f, d, CL, CT, family)
                found = sorted(m['phase_velocity_m_s'] for m in modes if m['family'] == family)
                self.assertEqual(len(found), len(expected), (f, family))
                for a, b in zip(found, expected):
                    self.assertAlmostEqual(a / b, 1.0, delta=1e-7)
                self.assertEqual(sorted(m['name'] for m in modes if m['family'] == family), [f'{family}{n}' for n in range(len(expected))])
            sh = sorted(n for n in range(40) if omega / CT > n * math.pi / d)
            for m in (m for m in modes if m['family'] == 'SH'):
                n = int(m['name'][2:])
                self.assertIn(n, sh)
                self.assertAlmostEqual(
                    m['phase_velocity_m_s'] / (omega / math.sqrt(omega**2 / CT**2 - (n * math.pi / d) ** 2)), 1.0, delta=1e-7
                )
                self.assertAlmostEqual(max(m['surface']), m['surface'][1])  # displacement across the propagation direction only
            self.assertEqual(len([m for m in modes if m['family'] == 'SH']), len(sh))
            self.assertEqual(modes, sorted(modes, key=lambda m: m['phase_velocity_m_s']))

    def test_the_group_velocity_is_the_slope_of_the_branch(self):
        p = plate(1e-3, 1e6)
        cls = next(c for c in p.classes if c['family'] == 'S')
        k = 900.0
        omega, group, shapes = p.solve(cls, k, vectors=True)
        step = 1e-3
        slope = (p.solve(cls, k + step) - p.solve(cls, k - step)) / (2 * step)
        np.testing.assert_allclose(group[:4], slope[:4], rtol=1e-6)
        self.assertAlmostEqual(float(np.max(np.abs(shapes[:, 0].conj() * p.mass @ shapes[:, 0]))), 1.0, places=9)  # unit mass norm

    def test_the_synchronous_pair_s1_s2_is_found_with_no_mismatch(self):
        d = 2e-3
        f = CT / math.sqrt(1 - CT**2 / CL**2) / d  # the frequency where S1 at f and S2 at 2f both travel at the longitudinal speed
        pairs = plate(d, 2.2 * f).pairs(f)
        best = next(r for r in pairs if (r['primary'], r['secondary']) == ('S1', 'S2'))
        self.assertLess(abs(best['mismatch']), 1e-7)
        self.assertAlmostEqual(best['phase_velocity_primary_m_s'] / CL, 1.0, delta=1e-7)
        # SH2 travels at the longitudinal speed at this frequency too (the same condition), so two pairs lead the table, in either order
        self.assertEqual({(r['primary'], r['secondary']) for r in pairs[:2]}, {('S1', 'S2'), ('SH2', 'S2')})
        self.assertLess(abs(pairs[1]['mismatch']), 1e-4)
        self.assertTrue(
            all(row['secondary'].startswith('S') and not row['secondary'].startswith('SH') for row in pairs)
        )  # second harmonics are symmetric Lamb modes
        s0 = next(r for r in plate(1e-3, 0.5e6).pairs(0.2e6) if (r['primary'], r['secondary']) == ('S0', 'S0'))
        self.assertLess(abs(s0['mismatch']), 0.005)
        self.assertGreater(s0['coherence_length_m'], 1.0)  # nearly matched at low frequency


class Anisotropic(unittest.TestCase):
    def test_classes_follow_the_symmetry_of_the_direction(self):
        material = presets()['cfrp_ud']['material']
        c = stiffness4(tensors(material)[0])
        rho = material['rho_kg_m3']
        along = Plate(1e-3, rho, c, 0.0, 6)
        skew = Plate(1e-3, rho, c, 27.0, 6)
        self.assertTrue(along.mirror_symmetric and along.sagittal_symmetric)
        self.assertEqual({cls['family'] for cls in along.classes}, {'S', 'A', 'SH'})
        self.assertTrue(skew.mirror_symmetric)
        self.assertFalse(skew.sagittal_symmetric)  # off the fibre axes the Lamb and SH families mix
        self.assertEqual({cls['family'] for cls in skew.classes}, {'Sq', 'Aq'})
        self.assertEqual(skew.name(skew.classes[0], 2), 'S2')
        fast = max(m['phase_velocity_m_s'] for m in along.at(0.2e6))
        slow = max(m['phase_velocity_m_s'] for m in Plate(1e-3, rho, c, 90.0, 6).at(0.2e6))
        self.assertGreater(fast, 2 * slow)  # along the fibres the plate wave is much faster than across them
        for p in (along, skew):  # the classes together are the whole problem
            k = 1500.0
            K = p.K0 + k * k * p.K2 + 1j * k * p.K1
            root = 1 / np.sqrt(p.mass)
            everything = np.sqrt(np.maximum(np.linalg.eigvalsh(K * root[:, None] * root[None, :]), 0.0))
            parts = np.sort(np.concatenate([p.solve(cls, k) for cls in p.classes]))
            np.testing.assert_allclose(parts, everything, rtol=1e-9, atol=1e-3 * everything[6])


class Model(unittest.TestCase):
    def scene(self):
        s = templates()['plate']
        s['schema'] = 'workai.hyperelastic.solid.v2'
        s['geometry'].update(thickness_mm=1.0)
        s['source'].update(frequency_mhz=0.2)
        s['boundaries'] = []
        return s

    def test_a_model_gets_its_curves_modes_and_pairs(self):
        s = self.scene()
        d = describe(s)
        self.assertEqual((d['frequency_mhz'], d['frequency_max_mhz'], d['thickness_mm']), (0.2, 0.52, 1.0))
        self.assertEqual(d['symmetry'], {'mid_plane': True, 'sagittal_plane': True})
        self.assertEqual([m['name'] for m in d['modes']['f']], ['A0', 'SH0', 'S0'])
        self.assertEqual(set(d['families']), {'S', 'A', 'SH'})
        s0 = next(m for m in d['modes']['f'] if m['name'] == 'S0')
        self.assertAlmostEqual(s0['wavelength_mm'] * s0['wavenumber_rad_mm'], 2 * math.pi)
        self.assertAlmostEqual(s0['phase_velocity_m_s'], 2 * math.pi * 0.2e6 / (s0['wavenumber_rad_mm'] * 1e3))
        self.assertTrue(s0['secondary'])
        for curve in d['curves']:
            self.assertEqual(len(curve['frequency_mhz']), len(curve['wavenumber_rad_mm']))
            self.assertEqual(curve['wavenumber_rad_mm'][0], 0.0)
        self.assertEqual(d['pairs'][0]['primary'] + d['pairs'][0]['secondary'], 'S0S0')
        self.assertLess(abs(d['pairs'][0]['mismatch']), 0.01)
        self.assertIs(plate_of(s), plate_of(self.scene()))  # the same plate is computed once

    def test_half_a_plate_with_its_mid_plane_as_a_symmetry_plane(self):
        from .guided import mid_plane

        whole = self.scene()
        half = self.scene()
        half['geometry']['thickness_mm'] = 0.5
        half['boundaries'] = (half.get('boundaries') or []) + [{'type': 'symmetry', 'face': 'zmin'}]
        self.assertEqual((mid_plane(whole), mid_plane(half)), ((1, None), (2, 'sym')))
        a, b = describe(whole), describe(half)
        self.assertEqual((b['thickness_mm'], b['model_thickness_mm'], b['mid_plane']), (1.0, 0.5, 'sym'))
        self.assertIsNone(a['mid_plane'])
        names = lambda d: [m['name'] for m in d['modes']['f']]
        self.assertEqual(names(a), ['A0', 'SH0', 'S0'])
        self.assertEqual(names(b), ['SH0', 'S0'])  # no antisymmetric modes in the half
        for name in ('S0', 'SH0'):
            x, y = (next(m for m in d['modes']['f'] if m['name'] == name) for d in (a, b))
            self.assertAlmostEqual(x['wavenumber_rad_mm'], y['wavenumber_rad_mm'], places=9)
            self.assertAlmostEqual(x['group_velocity_m_s'], y['group_velocity_m_s'], places=4)
        self.assertEqual({c['family'] for c in b['curves']}, {'S', 'SH'})
        self.assertIn('1 mm 厚的板的一半', b['validity']['note'])
        anti = self.scene()
        anti['geometry']['thickness_mm'] = 0.5
        anti['boundaries'] = (anti.get('boundaries') or []) + [{'type': 'antisymmetry', 'face': 'zmax'}]
        self.assertEqual(mid_plane(anti), (2, 'anti'))
        self.assertEqual([m['name'] for m in describe(anti)['modes']['f']], ['A0'])
        both = self.scene()
        both['boundaries'] = (both.get('boundaries') or []) + [{'type': 'symmetry', 'face': 'zmin'}, {'type': 'symmetry', 'face': 'zmax'}]
        with self.assertRaisesRegex(ValueError, '不是自由表面的平板'):
            describe(both)
        with self.assertRaises(ValueError):
            Plate(1e-3, RHO, C, 0.0, 8, parity='even')

    def test_what_the_infinite_plate_says_about_a_model(self):
        s = self.scene()
        self.assertFalse(validity(s)['plane_strain'])
        self.assertIn('对称面', validity(s)['note'])
        s['boundaries'] = [{'type': 'symmetry', 'face': 'ymin'}, {'type': 'symmetry', 'face': 'ymax'}]
        self.assertTrue(validity(s)['plane_strain'])
        pipe = templates()['pipe']
        with self.assertRaises(ValueError) as refused:
            plate_of(pipe)
        self.assertIn('圆管', str(refused.exception))
        with self.assertRaises(ValueError):
            describe(s, frequency_max_mhz=0.0)


if __name__ == '__main__':
    unittest.main()
