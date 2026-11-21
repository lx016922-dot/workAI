"""Material damping in the offline software: what a request may ask for, what the preview says, the attenuation of the plate's modes."""
import copy
import math
import unittest
import numpy as np
from physics.damping import decay_per_s
from .domain import templates, validate_request
from .guided import describe
from .preview import preview
from .scan import ROOTS
from .test_network_v2 import v2_plate


def damped(alpha=2e4, beta=3e-9):
    s = v2_plate(); s['damping'] = {'mass_per_s': alpha, 'stiffness_s': beta}; return s


class Requests(unittest.TestCase):
    def test_a_request_carries_the_damping_and_refuses_what_cannot_run(self):
        checked = validate_request({'scene': damped(), 'degree': 3}); self.assertEqual(checked['scene']['damping'], {'mass_per_s': 2e4, 'stiffness_s': 3e-9})
        self.assertEqual(validate_request(copy.deepcopy(checked))['scene'], checked['scene'])
        self.assertNotIn('damping', validate_request({'scene': damped(0., 0.), 'degree': 3})['scene'])
        self.assertEqual(validate_request({'scene': damped(), 'mesh': 'auto'})['scene']['damping']['stiffness_s'], 3e-9)
        # the stiffness-proportional term: automatic layer and the network engine only; mass-proportional damping runs anywhere
        with self.assertRaisesRegex(ValueError, '自动时间层'): validate_request({'scene': damped(), 'degree': 3, 'time_stepping': 'frozen'})
        with self.assertRaisesRegex(ValueError, '快速引擎'): validate_request({'scene': damped(), 'mesh': 'auto', 'engine': 'coarse'})
        self.assertEqual(validate_request({'scene': damped(beta=0.), 'degree': 3, 'time_stepping': 'frozen'})['time_stepping'], 'frozen')
        with self.assertRaises(ValueError): validate_request({'scene': dict(damped(), damping={'mass_per_s': -1., 'stiffness_s': 0.}), 'degree': 3})
        self.assertIn('damping', ROOTS)

    def test_the_preview_says_what_the_damping_does(self):
        out = preview({'scene': damped(), 'degree': 3}); self.assertTrue(out['ok'], out['errors']); self.assertIn('材料阻尼', out['scope'])
        d = out['damping']; f = damped()['source']['frequency_mhz'] * 1e6; self.assertEqual([round(r['frequency_mhz'] * 1e6) for r in d['at']], [round(f), round(2 * f)])
        self.assertAlmostEqual(d['at'][1]['decay_per_s'], decay_per_s(damped()['damping'], 2 * f)); self.assertGreater(d['at'][0]['shear_np_m'], d['at'][0]['longitudinal_np_m'])
        self.assertEqual(out['estimate']['scheme'], 'verlet2'); self.assertIn('阻尼', out['estimate']['layer_reason'])
        self.assertNotIn('damping', preview({'scene': v2_plate(), 'degree': 3}))
        strong = preview({'scene': damped(beta=5e-8), 'degree': 3}); self.assertTrue(any('损耗因子' in w for w in strong['warnings']))
        refused = preview({'scene': damped(), 'degree': 3, 'time_stepping': 'frozen'}); self.assertFalse(refused['ok']); self.assertIn('自动时间层', refused['errors'][0])

    def test_every_mode_of_the_plate_gets_its_attenuation(self):
        scene = validate_request({'scene': damped(), 'degree': 3})['scene']; d = describe(scene); f = scene['source']['frequency_mhz'] * 1e6
        self.assertAlmostEqual(d['damping']['decay_per_s']['2f'], decay_per_s(scene['damping'], 2 * f))
        for key, frequency in (('f', f), ('2f', 2 * f)):
            for m in d['modes'][key]: self.assertAlmostEqual(m['attenuation_np_m'], decay_per_s(scene['damping'], frequency) / abs(m['group_velocity_m_s']))
        plain = describe(validate_request({'scene': v2_plate(), 'degree': 3})['scene']); self.assertNotIn('damping', plain); self.assertNotIn('attenuation_np_m', plain['modes']['f'][0])


class Accumulation(unittest.TestCase):
    def rows(self, K, m, a1, offset, x):
        return [{'usable': True, 'distance_mm': float(d), 'ratio_per_m': K * (1 - math.exp(-m * (d + offset) * 1e-3)) / m, 'a1_m': 1e-9 * math.exp(-a1 * d * 1e-3)} for d in x]

    def test_the_slope_without_attenuation_is_recovered(self):
        from .nonlinear import _attenuation
        damping = {'mass_per_s': 2e4, 'stiffness_s': 3e-9}; f = .5; c1, c2 = 5000., 4000.
        scene = {'damping': damping, 'source': {'center_mm': [0., 1., 1.]}}; a1 = decay_per_s(damping, f * 1e6) / c1; a2 = decay_per_s(damping, 2 * f * 1e6) / c2; m = a2 - 2 * a1; K = 3.7e6
        x = np.linspace(20., 200., 10); rows = self.rows(K, m, a1, 15., x); points = [[35. + d - 20., 1., 1.] for d in x]; notes = []
        pair = {'primary': 'S0', 'secondary': 'S0', 'mismatch': 1e-4, 'group_velocity_primary_m_s': c1, 'group_velocity_secondary_m_s': c2}
        out = _attenuation(scene, f, [1., 0., 0.], points, x, rows, rows, x * 1e-3, np.array([r['a1_m'] for r in rows]), pair, {'theory': None}, 5100., notes)
        self.assertAlmostEqual(out['a1_measured_np_m'], a1, places=9); self.assertAlmostEqual(out['a1_np_m'], a1); self.assertAlmostEqual(out['a2_np_m'], a2); self.assertAlmostEqual(out['m_np_m'], m)
        self.assertAlmostEqual(out['source_offset_mm'], 15.); self.assertAlmostEqual(out['corrected']['slope_per_m2'] / K, 1., places=9); self.assertLess(abs(out['corrected']['intercept_per_m']), 1e-6 * K)
        self.assertEqual((out['mode_f'], out['mode_2f'], out['group_velocity_source'], notes), ('S0', 'S0', 'theory', []))
        raw = np.polyfit(x * 1e-3, [r['ratio_per_m'] for r in rows], 1)[0]; self.assertLess(raw, .9 * K)   # the uncorrected slope is lower
        # a mismatched pair and points upstream of the source get no correction, and say why
        for change, word in (({'pair': dict(pair, mismatch=.05)}, '相位失配'), ({'points': [[-50. + d, 1., 1.] for d in x]}, '下游')):
            fresh = self.rows(K, m, a1, 15., x); notes = []
            out = _attenuation(scene, f, [1., 0., 0.], change.get('points', points), x, fresh, fresh, x * 1e-3, np.array([r['a1_m'] for r in fresh]), change.get('pair', pair), {'theory': None}, 5100., notes)
            self.assertNotIn('corrected', out); self.assertIn(word, notes[0]); self.assertNotIn('ratio_corrected_per_m', fresh[0])
        # no damping in the model: only what A1 shows
        plain = _attenuation({'source': scene['source']}, f, [1., 0., 0.], points, x, rows, rows, x * 1e-3, np.array([r['a1_m'] for r in rows]), pair, {'theory': None}, 5100., [])
        self.assertEqual(set(plain), {'a1_measured_np_m', 'a1_measured_r_squared'})
        # without a recognised pair the measured group velocity stands in, with a note
        notes = []; fresh = self.rows(K, m, a1, 15., x)
        out = _attenuation(scene, f, [1., 0., 0.], points, x, fresh, fresh, x * 1e-3, np.array([r['a1_m'] for r in fresh]), None, {'theory': None}, 5100., notes)
        self.assertEqual(out['group_velocity_source'], 'measured'); self.assertAlmostEqual(out['a1_np_m'], decay_per_s(damping, f * 1e6) / 5100.); self.assertIn('二次模态', notes[0])


if __name__ == '__main__': unittest.main()
