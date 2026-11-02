import copy
import json
import unittest
from pathlib import Path
from solid.schema import default_scene, validate_scene as validate_v1
from . import scene as scene_v2

EXAMPLE = Path(__file__).resolve().parents[1] / 'docs/examples/plate-v2-absorbing-comb.json'

def v2(kind='plate', **numerics):
    s = default_scene(kind); s['schema'] = scene_v2.V2; s['numerics'].update(numerics); return s

class SceneV2Tests(unittest.TestCase):
    def test_version1_scenes_pass_through_unchanged(self):
        for kind in ['plate', 'pipe', 'hole', 'notch']:
            raw = default_scene(kind); self.assertEqual(scene_v2.validate_scene(raw), validate_v1(raw))
            self.assertFalse(scene_v2.is_v2(validate_v1(raw))); self.assertEqual(scene_v2.sources(validate_v1(raw)), [validate_v1(raw)['source']])

    def test_minimal_version2_equals_core_plus_empty_extensions(self):
        core = validate_v1(default_scene()); out = scene_v2.validate_scene(v2())
        self.assertEqual(out['schema'], scene_v2.V2); self.assertEqual(out['sources'], []); self.assertEqual(out['boundaries'], [])
        for key in core:
            if key != 'schema': self.assertEqual(out[key], core[key])

    def test_example_scene_validates_and_normalizes(self):
        raw = json.loads(EXAMPLE.read_text()); out = scene_v2.validate_scene(raw)
        self.assertEqual(len(out['sources']), 1); self.assertEqual(out['sources'][0]['signal']['type'], 'gaussian_pulse')
        self.assertEqual(out['source']['profile']['type'], 'comb'); self.assertEqual([b['type'] for b in out['boundaries']], ['absorbing', 'absorbing', 'symmetry'])
        self.assertEqual(out, scene_v2.validate_scene(out))
        scene_v2.spectral_admissible(out); self.assertEqual(len(scene_v2.summary(out)['sources']), 2)

    def test_boundary_rejections(self):
        def reject(mutate, message=None):
            s = v2(); mutate(s)
            with self.assertRaises(ValueError) as caught: scene_v2.validate_scene(s)
            if message: self.assertIn(message, str(caught.exception))
        reject(lambda s: s.update(boundaries=[{'type': 'absorbing', 'face': 'xmin', 'length_mm': 2., 'ramp': 'cubic', 'attenuation_db': 60.}]), '加载面')
        reject(lambda s: s.update(boundaries=[{'type': 'symmetry', 'face': 'ymin'}, {'type': 'absorbing', 'face': 'ymin', 'length_mm': 1., 'ramp': 'cubic', 'attenuation_db': 60.}]), '同一面')
        reject(lambda s: (s['boundary'].update(fixed_face='xmax'), s.update(boundaries=[{'type': 'symmetry', 'face': 'xmax'}])), '固定面')
        reject(lambda s: s.update(boundaries=[{'type': 'absorbing', 'face': 'xmax', 'length_mm': 7., 'ramp': 'cubic', 'attenuation_db': 60.}]), '一半')
        reject(lambda s: s.update(boundaries=[{'type': 'absorbing', 'face': 'xmax', 'length_mm': 3., 'ramp': 'steep', 'attenuation_db': 60.}]), '渐变')
        reject(lambda s: s.update(boundaries=[{'type': 'absorbing', 'face': 'xmax', 'length_mm': 3., 'ramp': 'cubic'}]))
        reject(lambda s: s.update(boundaries=[{'type': 'slip', 'face': 'xmax'}]), '边界类型')
        pipe = v2('pipe'); pipe['boundaries'] = [{'type': 'symmetry', 'face': 'ymin'}]
        with self.assertRaises(ValueError): scene_v2.validate_scene(pipe)

    def test_source_rejections(self):
        def reject(mutate, message=None):
            s = v2(); mutate(s)
            with self.assertRaises(ValueError) as caught: scene_v2.validate_scene(s)
            if message: self.assertIn(message, str(caught.exception))
        reject(lambda s: s['source'].update(signal={'type': 'chirp', 'end_frequency_mhz': .4, 'duration_us': 20., 'window': 'hann'}), '覆盖完整激励')
        reject(lambda s: s['source'].update(profile={'type': 'comb', 'pitch_mm': 2., 'teeth': 3, 'tooth_width_mm': 1., 'along': 'x', 'alternating_sign': False}), '法向')
        reject(lambda s: s.update(sources=[dict(s['source'])]), 'name')
        reject(lambda s: s.update(sources=[dict(s['source'], name='a'), dict(s['source'], name='a')]), '重复')
        reject(lambda s: s['source'].update(signal={'type': 'ricker'}, frequency_mhz=.25), '带宽')
        ok = v2(); ok['source'].update(signal={'type': 'ricker'}, cycles=7., frequency_mhz=.25); ok['numerics']['h_mm'] = .5
        out = scene_v2.validate_scene(ok); self.assertEqual(out['source']['cycles'], 7.); self.assertEqual(out['source']['signal'], {'type': 'ricker'})

    def test_spectral_admissibility(self):
        s = v2(); s['source'].update(face='surface_patch', radius_mm=1., patch_normal=[0., 0., 1.], center_mm=[6., 2., 2.])
        out = scene_v2.validate_scene(s)
        with self.assertRaises(ValueError): scene_v2.spectral_admissible(out)
        pipe = scene_v2.validate_scene(v2('pipe'))
        pipe['sources'] = [dict(pipe['source'], name='side', face='ymin')]
        with self.assertRaises(ValueError): scene_v2.spectral_admissible(pipe)

if __name__ == '__main__': unittest.main()
