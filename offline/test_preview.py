"""Editor preview: never raises, reports validation errors, signals, mesh size, 2f resolution and the original-acceptance scope."""
import copy
import math
import unittest

from offline.domain import templates
from offline.preview import preview
from offline.test_network_v2 import v2_plate


class Preview(unittest.TestCase):
    def test_valid_v2_scene(self):
        scene = v2_plate(); result = preview({'scene': scene, 'degree': 4}, '.')
        self.assertTrue(result['ok'], result['errors']); self.assertEqual(result['errors'], [])
        self.assertEqual([s['name'] for s in result['signals']], ['主激励', 'top patch'])
        for s in result['signals']:
            self.assertEqual(len(s['times_us']), len(s['values_mpa'])); self.assertEqual(len(s['spectrum_mhz']), len(s['spectrum_db']))
            self.assertLessEqual(len(s['times_us']), 800); self.assertAlmostEqual(max(s['spectrum_db']), 0, places=6)
        mesh = result['mesh']; self.assertGreater(mesh['nodes'], 0); self.assertEqual(mesh['degree'], 4); self.assertEqual(len(mesh['grid_mm']), 3)
        self.assertEqual(result['record']['samples'], 401); self.assertEqual(result['record']['uncompressed_bytes'], 401 * mesh['nodes'] * 3 * 8 * 3)
        m = scene['material']; shear = math.sqrt(m['mu_pa'] / m['rho_kg_m3'])
        for row, sig in zip(result['resolution'], result['signals']):
            self.assertAlmostEqual(row['nodes_per_2f_shear_wavelength'], shear / (2 * sig['band_30db_mhz'] * 1e6) / (scene['numerics']['h_mm'] / 4 * 1e-3))
        self.assertIn('2 个激励源', result['scope']); self.assertIn('非 X 端面加载', result['scope'])

    def test_short_absorbing_layers_are_flagged(self):
        from physics.materials import fastest_speed
        scene = v2_plate(); scene['boundaries'] = [b for b in scene.get('boundaries', []) if b['type'] != 'absorbing']
        wavelength = fastest_speed(scene['material']) / (min(s['frequency_mhz'] for s in [scene['source'], *scene.get('sources', [])]) * 1e6) * 1e3
        from offline.domain import validate_request
        from offline.preview import absorber_warnings
        for length, flagged in ((8., True), (round(2.05 * wavelength, 1), False)):
            s = copy.deepcopy(scene); s['geometry']['length_mm'] = max(s['geometry']['length_mm'], 2.2 * length)
            s['boundaries'].append({'type': 'absorbing', 'face': 'xmax', 'length_mm': length, 'ramp': 'cubic', 'attenuation_db': 60.})
            warnings = absorber_warnings(validate_request({'scene': s, 'degree': 4})['scene'])
            self.assertEqual(any('吸收层 xmax' in w for w in warnings), flagged, (length, warnings))
            if flagged:  # and the editor preview shows it
                result = preview({'scene': s, 'degree': 4}, '.'); self.assertTrue(result['ok'], result['errors']); self.assertTrue(any('吸收层 xmax' in w for w in result['warnings']))

    def test_long_records_with_coarse_meshes_are_flagged(self):
        from offline.preview import long_record_warnings
        signals = [{'name': '主激励', 'frequency_mhz': .2}]
        self.assertEqual(long_record_warnings(1000., signals, [{'name': '主激励', 'nodes_per_2f_shear_wavelength': 15.4}]), [])
        self.assertEqual(len(long_record_warnings(1000., signals, [{'name': '主激励', 'nodes_per_2f_shear_wavelength': 11.5}])), 1)
        self.assertEqual(long_record_warnings(30., signals, [{'name': '主激励', 'nodes_per_2f_shear_wavelength': 11.5}]), [])  # 6 periods: short record
        scene = v2_plate(); scene['acquisition'] = {'duration_us': 1000., 'sample_step_us': .05}
        result = preview({'scene': scene, 'degree': 4}, '.'); n = result['resolution'][0]['nodes_per_2f_shear_wavelength']
        self.assertEqual(any('载波周期' in w for w in result['warnings']), n < 15, (n, result['warnings']))

    def test_invalid_scenes_report_errors_without_raising(self):
        bad = v2_plate(); bad['numerics']['h_mm'] = 3.
        result = preview({'scene': bad, 'degree': 4}, '.')
        self.assertFalse(result['ok']); self.assertTrue(any('网格' in e for e in result['errors'])); self.assertIsNone(result['mesh'])
        self.assertEqual(len(result['signals']), 2)  # signals still previewed for the editor
        broken = v2_plate(); broken['sources'][0]['signal'] = {'type': 'chirp'}
        self.assertFalse(preview({'scene': broken, 'degree': 4})['ok'])
        for body in (None, {}, {'scene': 3}, {'scene': {'schema': 'x'}, 'degree': 4}):
            self.assertFalse(preview(body)['ok'])

    def test_original_scope_for_version_1_templates(self):
        plate = templates()['plate']
        self.assertEqual(preview({'scene': plate, 'degree': 4})['scope'], [])
        wide = copy.deepcopy(plate); wide['geometry']['length_mm'] = 40.
        self.assertEqual(preview({'scene': wide, 'degree': 4})['scope'], ['长度 40 mm（原验收 8–12）'])


if __name__ == '__main__': unittest.main()
