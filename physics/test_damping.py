"""Rayleigh damping of the material: the scene field, the decay it stands for, and the solvers that refuse what they do not model."""
import math
import unittest

from physics.damping import attenuation_np_m, coefficients, decay_per_s, describe, from_decay, only_mass, validate_damping
from physics.scene import validate_scene
from solid.schema import default_scene


def scene(**damping):
    s = default_scene('plate'); s['schema'] = 'workai.hyperelastic.solid.v2'
    if damping: s['damping'] = damping
    return s


class Damping(unittest.TestCase):
    def test_validation(self):
        self.assertIsNone(validate_damping(None)); self.assertIsNone(validate_damping({'mass_per_s': 0, 'stiffness_s': 0}))
        self.assertEqual(validate_damping({'mass_per_s': 100, 'stiffness_s': 1e-9}), {'mass_per_s': 100., 'stiffness_s': 1e-9})
        for bad in ({'mass_per_s': -1, 'stiffness_s': 0}, {'mass_per_s': 0, 'stiffness_s': 1e-3}, {'mass_per_s': 1}, {'mass_per_s': 1, 'stiffness_s': 0, 'q': 5}, 3):
            with self.assertRaises(ValueError): validate_damping(bad)

    def test_the_scene_carries_it(self):
        self.assertNotIn('damping', validate_scene(scene())); self.assertNotIn('damping', validate_scene(scene(mass_per_s=0., stiffness_s=0.)))
        checked = validate_scene(scene(mass_per_s=2e4, stiffness_s=3e-9)); self.assertEqual(checked['damping'], {'mass_per_s': 2e4, 'stiffness_s': 3e-9}); self.assertEqual(coefficients(checked), (2e4, 3e-9))
        self.assertEqual(coefficients(validate_scene(scene())), (0., 0.))
        flat = scene(mass_per_s=2e4, stiffness_s=0.); flat['dimension'] = '2d_plane_strain'; flat['source'].update(face='xmin', direction=[1., 0., 0.])
        with self.assertRaisesRegex(ValueError, '平面应变'): validate_scene(flat)
        with self.assertRaises(ValueError): validate_scene(scene(mass_per_s='x', stiffness_s=0.))

    def test_decay_rate_and_attenuation(self):
        d = {'mass_per_s': 2e4, 'stiffness_s': 4e-9}; f = .5e6; w = 2 * math.pi * f
        self.assertAlmostEqual(decay_per_s(d, f), 1e4 + 2e-9 * w * w); self.assertEqual(decay_per_s(None, f), 0.)
        self.assertAlmostEqual(attenuation_np_m(d, f, 5000.), decay_per_s(d, f) / 5000.); self.assertIsNone(attenuation_np_m(d, f, 0.))
        row = describe(d, (f,))['at'][0]; self.assertAlmostEqual(row['loss_factor'], 2 * decay_per_s(d, f) / w); self.assertAlmostEqual(row['q'] * row['loss_factor'], 1.)
        self.assertIsNone(describe(None))

    def test_two_decay_rates_give_the_two_coefficients(self):
        d = {'mass_per_s': 2e4, 'stiffness_s': 4e-9}; back = from_decay((.5e6, decay_per_s(d, .5e6)), (1e6, decay_per_s(d, 1e6)))
        self.assertAlmostEqual(back['mass_per_s'], 2e4, places=6); self.assertAlmostEqual(back['stiffness_s'] * 1e9, 4., places=9)
        only = from_decay((.5e6, 3e4)); self.assertEqual(only['mass_per_s'], 0.); self.assertAlmostEqual(decay_per_s(only, .5e6), 3e4)
        same = from_decay((.5e6, 3e4), (1e6, 3e4)); self.assertAlmostEqual(same['mass_per_s'], 6e4); self.assertEqual(same['stiffness_s'], 0.)     # the same rate at both: mass-proportional
        four = from_decay((.5e6, 3e4), (1e6, 12e4)); self.assertAlmostEqual(four['mass_per_s'], 0., places=6); self.assertGreater(four['stiffness_s'], 0.)   # as ω²: stiffness-proportional
        for impossible in (((.5e6, 3e4), (1e6, 2e4)), ((.5e6, 3e4), (1e6, 13e4))):   # falling with frequency, or rising faster than ω²
            with self.assertRaisesRegex(ValueError, '质量比例加刚度比例'): from_decay(*impossible)

    def test_solvers_without_the_stiffness_term_say_so(self):
        only_mass({'beta': 0.}, 'x'); only_mass(None, 'x')
        with self.assertRaisesRegex(ValueError, '只支持质量比例'): only_mass({'beta': 1e-9}, '参考求解器')


if __name__ == '__main__': unittest.main()
