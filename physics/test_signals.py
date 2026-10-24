import unittest
import numpy as np
from solid.schema import default_scene
from solid.fem import pulse
from . import signals

def source(**changes):
    s = default_scene()['source']; s.update(changes); return s

class SignalTests(unittest.TestCase):
    def test_default_tone_burst_matches_version1_pulse(self):
        s = source(frequency_mhz=.3, cycles=2.5, delay_us=.7, phase_deg=33.)
        times = np.linspace(-1, 14, 1501)
        expected = np.array([pulse(t, s) for t in times])
        np.testing.assert_allclose(signals.evaluate(s, times), expected, rtol=0, atol=1e-15)
        self.assertAlmostEqual(signals.evaluate(s, 2.), pulse(2., s), places=15)
        self.assertEqual(signals.support_us(s), (.7, .7 + 2.5 / .3))

    def test_normalized_families_have_unit_peak_and_compact_support(self):
        cases = [{'type': 'gaussian_pulse', 'bandwidth': .6}, {'type': 'ricker'},
                 {'type': 'chirp', 'end_frequency_mhz': .6, 'duration_us': 12., 'window': 'hann'},
                 {'type': 'multi_tone', 'components': [{'frequency_mhz': .2, 'weight': 1., 'phase_deg': 0.}, {'frequency_mhz': .35, 'weight': .5, 'phase_deg': 90.}], 'duration_us': 10., 'window': 'hann'}]
        for signal in cases:
            s = source(delay_us=1., cycles=1.); s['signal'] = signals.validate_signal(signal, s)
            start, end = signals.support_us(s); grid = np.linspace(start, end, 20001)
            peak = np.max(np.abs(signals.evaluate(s, grid)))
            self.assertAlmostEqual(peak, 1., places=3, msg=signal['type'])
            self.assertEqual(signals.evaluate(s, start - 1e-6), 0.); self.assertEqual(signals.evaluate(s, end + 1e-6), 0.)
            self.assertGreater(signals.describe(s)['bandwidth_30db_mhz'], 0.)

    def test_windows_and_rejections(self):
        s = source(cycles=4.)
        for window, extra in [('hann', {}), ('rectangular', {}), ('tukey', {'tukey_alpha': .5}), ('gaussian', {'gaussian_width': .2})]:
            s['signal'] = signals.validate_signal({'type': 'tone_burst', 'window': window, **extra}, s)
            values = signals.evaluate(s, np.linspace(0, 20, 4001)); self.assertLessEqual(np.max(np.abs(values)), 1 + 1e-12)
        rectangular = dict(s, signal={'type': 'tone_burst', 'window': 'rectangular'})
        self.assertGreater(signals.bandwidth_mhz(rectangular), signals.bandwidth_mhz(dict(s, signal={'type': 'tone_burst', 'window': 'hann'})))
        for bad in [{'type': 'square'}, {'type': 'tone_burst', 'window': 'kaiser'}, {'type': 'tone_burst', 'window': 'tukey'},
                    {'type': 'chirp', 'end_frequency_mhz': .4, 'duration_us': 1., 'window': 'hann'},
                    {'type': 'multi_tone', 'components': [{'frequency_mhz': .4, 'weight': 1., 'phase_deg': 0.}], 'duration_us': 10., 'window': 'hann'},
                    {'type': 'samples', 'sample_step_us': .1, 'values': [0.] * 16, 'normalize': True},
                    {'type': 'samples', 'sample_step_us': .1, 'values': [2.] * 16, 'normalize': False}]:
            with self.assertRaises(ValueError): signals.validate_signal(bad, s)

    def test_sampled_record_is_interpolated_and_normalized(self):
        raw = (3 * np.sin(np.linspace(0, 4 * np.pi, 41))).tolist()
        s = source(delay_us=2., cycles=1.); s['signal'] = signals.validate_signal({'type': 'samples', 'sample_step_us': .25, 'values': raw, 'normalize': True}, s)
        self.assertAlmostEqual(max(abs(v) for v in s['signal']['values']), 1.)
        grid = 2. + .25 * np.arange(41)
        np.testing.assert_allclose(signals.evaluate(s, grid), np.array(raw) / 3, atol=1e-15)
        self.assertAlmostEqual(signals.evaluate(s, 2.125), (s['signal']['values'][0] + s['signal']['values'][1]) / 2)
        self.assertEqual(signals.support_us(s), (2., 12.)); self.assertEqual(signals.evaluate(s, 12.5), 0.)

if __name__ == '__main__': unittest.main()
