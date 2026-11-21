"""Transducers in the offline software: sources that stand for a wedge and a bonded wafer, exact element loads, and receivers of finite size."""

import math
import unittest
import numpy as np
from graphnet.inputs import build
from physics.loads import strips
from spectral3d.feature_mesh import make_mesh
from spectral3d.loads import MAX_PARTS, prepare_excitation, source_loads
from .domain import validate_request
from .nonlinear import aperture
from .preview import preview
from .test_network_v2 import v2_plate

WEDGE = {
    'type': 'wedge',
    'along': 'x',
    'length_mm': 6.0,
    'width_mm': 0.0,
    'trace_velocity_m_s': 5400.0,
    'reverse': False,
    'apodization': 'uniform',
}
PATCH = {'type': 'patch', 'shape': 'strip', 'length_mm': 5.0, 'width_mm': 0.0, 'edge_mm': 0.6}


def plate(profile, direction):
    s = v2_plate()
    s.pop('sources')
    g = s['geometry']
    g.update(length_mm=16.0)
    s['source'].update(
        face='zmax', center_mm=[8.0, g['width_mm'] / 2, g['thickness_mm']], radius_mm=0.0, direction=direction, profile=profile
    )
    s['boundaries'] = [{'type': 'symmetry', 'face': 'ymax'}]
    return s


class Sources(unittest.TestCase):
    def test_requests_take_transducer_profiles(self):
        checked = validate_request({'scene': plate(WEDGE, [0.0, 0.0, -1.0]), 'degree': 3})
        self.assertEqual(checked['scene']['source']['profile'], WEDGE)
        self.assertEqual(validate_request({'scene': plate(PATCH, [1.0, 0.0, 0.0]), 'mesh': 'auto'})['scene']['source']['profile'], PATCH)
        with self.assertRaisesRegex(ValueError, '面内'):
            validate_request({'scene': plate(PATCH, [0.0, 0.0, -1.0]), 'degree': 3})
        out = preview({'scene': plate(WEDGE, [0.0, 0.0, -1.0]), 'degree': 3})
        self.assertTrue(out['ok'], out['errors'])
        self.assertEqual(out['signals'][0]['type'], 'tone_burst')

    def test_element_loads_are_exact_wherever_the_mesh_lines_fall(self):
        for h in (1.6, 1.1):
            scene = validate_request({'scene': plate(WEDGE, [0.0, 0.0, -1.0]), 'degree': 3})['scene']
            mesh = make_mesh(scene, 3, h)
            width = scene['geometry']['width_mm']
            parts, area, weighted = source_loads(mesh, scene['source'])
            self.assertEqual(len(parts), strips(WEDGE, scene['source']['frequency_mhz']))
            self.assertLessEqual(len(parts), MAX_PARTS)
            self.assertAlmostEqual(area * 1e6, 6.0 * width, places=9)
            self.assertAlmostEqual(weighted * 1e6, 6.0 * width, places=9)  # footprint × width, whatever the element size
            total = sum(force for force, _ in parts)
            self.assertAlmostEqual(total[:, 2].sum(), -scene['source']['amplitude_mpa'] * 6.0 * width, places=9)
            self.assertEqual(np.abs(total[:, :2]).max(), 0.0)
            x = mesh['points'][np.abs(total[:, 2]) > 0, 0] * 1e3
            self.assertGreaterEqual(x.min(), 5.0 - h - 1e-9)
            self.assertLessEqual(x.max(), 11.0 + h + 1e-9)
            delays = [part['delay_us'] for _, part in parts]
            self.assertAlmostEqual(delays[-1] - delays[0], 6e-3 / 5400.0 * 1e6)
            wafer = validate_request({'scene': plate(PATCH, [1.0, 0.0, 0.0]), 'degree': 3})['scene']
            parts, area, weighted = source_loads(mesh, wafer['source'])
            self.assertEqual(len(parts), 1)
            self.assertAlmostEqual(area * 1e6, 2 * 0.6 * width, places=9)
            self.assertAlmostEqual(weighted, 0.0, places=15)  # two edge zones, equal and opposite
            force = parts[0][0]
            self.assertLess(abs(force[:, 0].sum()), 1e-12)
            self.assertEqual(np.abs(force[:, 1:]).max(), 0.0)
            right = mesh['points'][:, 0] > 8e-3
            self.assertAlmostEqual(
                force[right, 0].sum(), wafer['source']['amplitude_mpa'] * 0.6 * width, places=9
            )  # the wafer pushes outwards: +x on its +x edge
        inputs = build(scene, mesh)
        self.assertEqual(len(inputs['sources']), len(inputs['loads']))
        self.assertEqual(inputs['loads'].shape[0], strips(WEDGE, scene['source']['frequency_mhz']))
        row = prepare_excitation(mesh, scene)['summary']['sources'][0]
        self.assertEqual((row['profile'], row['parts']), ('wedge', inputs['loads'].shape[0]))


class Receivers(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(0.0, 60.0, 0.05)
        self.x = np.arange(0.0, 60.0, 0.5)
        self.f = 0.5
        self.v = 5400.0
        wave = lambda v: (
            np.sin(2 * np.pi * self.f * (self.t[:, None] - self.x[None, :] * 1e-3 / v * 1e6))
            * np.exp(-(((self.t[:, None] - self.x[None, :] * 1e-3 / v * 1e6 - 20.0) / 6.0) ** 2))
        )
        self.matched = wave(self.v)
        self.other = wave(2330.0)

    def amplitude(self, v):
        return np.abs(v).max(0)

    def test_a_point_changes_nothing(self):
        out, kept, used = aperture(self.t, {'linear': self.matched}, self.x, None, 0.05)
        self.assertIs(out['linear'], self.matched)
        self.assertEqual((kept, used), (list(range(len(self.x))), {'type': 'point'}))

    def test_a_wedge_keeps_the_wave_it_is_matched_to(self):
        out, kept, used = aperture(
            self.t,
            {'linear': self.matched, 'murnaghan': self.other},
            self.x,
            {'type': 'wedge', 'length_mm': 12.0, 'trace_velocity_m_s': self.v},
            0.05,
        )
        self.assertEqual((kept[0], kept[-1], used['points'], used['length_mm']), (12, len(self.x) - 13, 25, 12.0))
        self.assertTrue(
            np.allclose(out['linear'], self.matched[:, kept], atol=2e-3)
        )  # the matched wave comes through as it is at the centre
        k = 2 * math.pi * self.f * 1e6 * (1 / 2330.0 - 1 / self.v)
        offsets = np.arange(-12, 13) * 0.5e-3
        expected = abs(np.mean(np.exp(-1j * k * offsets)))  # what the footprint leaves of the other wave
        self.assertAlmostEqual(self.amplitude(out['murnaghan'])[20] / self.amplitude(self.other)[kept[20]], expected, delta=0.02)
        self.assertLess(expected, 0.25)
        hann, _, used = aperture(
            self.t,
            {'linear': self.other},
            self.x,
            {'type': 'wedge', 'length_mm': 12.0, 'trace_velocity_m_s': self.v, 'apodization': 'hann'},
            0.05,
        )
        self.assertEqual(used['apodization'], 'hann')
        self.assertLess(self.amplitude(hann['linear'])[20], self.amplitude(self.other)[kept[20]])

    def test_a_wafer_reads_the_difference_of_its_edges(self):
        out, kept, used = aperture(self.t, {'linear': self.matched}, self.x, {'type': 'patch', 'length_mm': 5.0}, 0.05)
        self.assertEqual((used['length_mm'], used['points']), (5.0, 11))
        self.assertTrue(np.array_equal(out['linear'], self.matched[:, 10:] - self.matched[:, :-10]))
        k = 2 * math.pi * self.f * 1e6 / self.v
        self.assertAlmostEqual(self.amplitude(out['linear'])[30], 2 * abs(math.sin(k * 2.5e-3)), delta=0.02)  # 2 sin(ka) of a unit wave

    def test_what_cannot_be_done_is_refused(self):
        for receiver, word in (
            ({'type': 'laser'}, '接收方式'),
            ({'type': 'wedge', 'length_mm': 12.0}, '迹线速度'),
            ({'type': 'patch', 'length_mm': 0.3}, '小于两个点距'),
            ({'type': 'patch', 'length_mm': 59.5}, '放不下'),
            ({'type': 'patch', 'length_mm': -1.0}, '正的长度'),
            ({'type': 'point', 'length_mm': 1.0}, '不带其他参数'),
        ):
            with self.assertRaisesRegex(ValueError, word):
                aperture(self.t, {'linear': self.matched}, self.x, receiver, 0.05)
        uneven = self.x.copy()
        uneven[5] += 0.1
        with self.assertRaisesRegex(ValueError, '等距'):
            aperture(self.t, {'linear': self.matched}, uneven, {'type': 'patch', 'length_mm': 5.0}, 0.05)


if __name__ == '__main__':
    unittest.main()
