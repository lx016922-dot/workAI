"""Transducer profiles (physics/loads.py): a wedge as a travelling traction, a bonded wafer as edge shear."""
import unittest
import numpy as np

from physics.loads import MAX_STRIPS, STRIPS_PER_WAVELENGTH, expand, expand_all, region_weights, strips, travel_us, validate_profile
from physics.scene import validate_scene
from solid.schema import default_scene

SOURCE = {'face': 'zmax', 'center_mm': [50., 1., 2.], 'radius_mm': 0., 'direction': [0., 0., -1.], 'amplitude_mpa': 1., 'frequency_mhz': .5, 'cycles': 5., 'delay_us': 1., 'phase_deg': 0.}
WEDGE = {'type': 'wedge', 'along': 'x', 'length_mm': 20., 'width_mm': 0., 'trace_velocity_m_s': 5400., 'reverse': False, 'apodization': 'uniform'}
PATCH = {'type': 'patch', 'shape': 'strip', 'length_mm': 10., 'width_mm': 0., 'edge_mm': 1.}


def line(n=2001, lo=30., hi=70.):
    x = np.linspace(lo, hi, n); return x, np.stack([x, np.ones(n), 2 * np.ones(n)], 1) * 1e-3


def grid(step=.1):
    g = np.stack(np.meshgrid(np.arange(40., 60. + step / 2, step), np.arange(-9., 11. + step / 2, step), indexing='ij'), -1).reshape(-1, 2)
    return np.column_stack([g, 2 * np.ones(len(g))]) * 1e-3, (step * 1e-3) ** 2


class Wedge(unittest.TestCase):
    def source(self, **changes): return dict(SOURCE, profile=validate_profile(dict(WEDGE, **changes), SOURCE))

    def test_validation(self):
        self.assertEqual(validate_profile(WEDGE, SOURCE), WEDGE)
        for bad in ({'along': 'z'}, {'trace_velocity_m_s': 50.}, {'apodization': 'cosine'}, {'reverse': 1}, {'length_mm': 0.}, {'extra': 1}):
            with self.assertRaises(ValueError): validate_profile(dict(WEDGE, **bad), SOURCE)
        with self.assertRaisesRegex(ValueError, 'radius_mm'): validate_profile(WEDGE, dict(SOURCE, radius_mm=3.))
        with self.assertRaisesRegex(ValueError, '个迹线波长'): validate_profile(dict(WEDGE, length_mm=200.), SOURCE)   # 18.5 wavelengths: more than the strips can carry

    def test_the_strips_carry_the_delay_along_the_footprint(self):
        s = self.source(); parts = expand(s); wavelength = 5400. / .5e6 * 1e3
        self.assertEqual(len(parts), strips(s['profile'], .5)); self.assertGreaterEqual(len(parts) - 1, 20. / wavelength * STRIPS_PER_WAVELENGTH); self.assertLessEqual(len(parts), MAX_STRIPS)
        self.assertAlmostEqual(parts[0]['delay_us'], 1.); self.assertAlmostEqual(parts[-1]['delay_us'], 1. + 20e-3 / 5400. * 1e6); self.assertAlmostEqual(travel_us(s), 20e-3 / 5400. * 1e6)
        self.assertTrue(all(p['profile']['type'] == 'wedge_strip' and p['direction'] == SOURCE['direction'] for p in parts)); self.assertEqual(travel_us(SOURCE), 0.)
        x, points = line(); W = np.array([region_weights(points, p, 2) for p in parts]); inside = (x > 40.01) & (x < 59.99)
        self.assertTrue(np.allclose(W.sum(0)[inside], 1.)); self.assertTrue(np.all(W.sum(0)[(x < 39.99) | (x > 60.01)] == 0)); self.assertTrue(np.all(W >= 0))
        # the delay a point sees (the weighted mean over the strips) is its distance from the entry edge over the trace velocity
        delays = np.array([p['delay_us'] for p in parts]); seen = (W * delays[:, None]).sum(0)[inside]; self.assertTrue(np.allclose(seen, 1. + (x[inside] - 40.) * 1e-3 / 5400. * 1e6, atol=1e-9))
        # reversed: the wave enters at the other end
        back = expand(self.source(reverse=True)); B = np.array([region_weights(points, p, 2) for p in back]); seen = (B * delays[:, None]).sum(0)[inside]
        self.assertTrue(np.allclose(seen, 1. + (60. - x[inside]) * 1e-3 / 5400. * 1e6, atol=1e-9))

    def test_the_load_is_a_wave_at_the_trace_wavenumber(self):
        s = self.source(length_mm=32.4); parts = expand(s); x, points = line(4001, 20., 80.); w = 2 * np.pi * .5e6; k = w / 5400.
        field = sum(region_weights(points, p, 2) * np.exp(-1j * w * (p['delay_us'] - 1.) * 1e-6) for p in parts)      # the complex amplitude of the traction at f
        inside = (x > 34.) & (x < 66.); exact = np.exp(-1j * k * (x - 33.8) * 1e-3)
        ratio = field[inside] / exact[inside]; self.assertLess(np.abs(np.angle(ratio)).max(), .01)                   # the phase is right to 0.01 rad; the interpolation lowers the amplitude a little
        self.assertGreater(np.abs(ratio).min(), 1 - (np.pi / STRIPS_PER_WAVELENGTH) ** 2 / 2 - 1e-6); self.assertLessEqual(np.abs(ratio).max(), 1 + 1e-12)
        spectrum = lambda q: abs(np.sum(field * np.exp(1j * q * x * 1e-3))); self.assertGreater(spectrum(k), 300 * spectrum(-k))   # forward, not backward
        hann = expand(self.source(apodization='hann')); H = np.array([region_weights(line()[1], p, 2) for p in hann]).sum(0); xs = line()[0]
        self.assertAlmostEqual(H[np.argmin(np.abs(xs - 50.))], 1.); self.assertAlmostEqual(H[np.argmin(np.abs(xs - 45.))], .5, places=6); self.assertAlmostEqual(H[np.argmin(np.abs(xs - 40.))], 0.)

    def test_a_width_limits_the_footprint_across(self):
        parts = expand(self.source(width_mm=1.)); x, points = line(); off = points.copy(); off[:, 1] = 1.6e-3
        self.assertGreater(np.abs(np.array([region_weights(points, p, 2) for p in parts])).max(), .9); self.assertEqual(np.abs(np.array([region_weights(off, p, 2) for p in parts])).max(), 0.)


class Patch(unittest.TestCase):
    def source(self, direction=(1., 0., 0.), **changes):
        s = dict(SOURCE, direction=list(direction)); return dict(s, profile=validate_profile(dict(PATCH, **changes), s))

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, '面内'): validate_profile(PATCH, SOURCE)                     # the default direction is the face normal
        with self.assertRaisesRegex(ValueError, '坐标轴'): validate_profile(PATCH, dict(SOURCE, direction=[.6, .8, 0.]))
        self.assertEqual(validate_profile(dict(PATCH, shape='disc'), dict(SOURCE, direction=[.6, .8, 0.]))['shape'], 'disc')
        for bad in ({'shape': 'ring'}, {'edge_mm': 6.}, {'shape': 'rectangle', 'width_mm': 1.}, {'length_mm': -1.}):
            with self.assertRaises(ValueError): validate_profile(dict(PATCH, **bad), dict(SOURCE, direction=[1., 0., 0.]))

    def traction(self, source):
        points, dA = grid(); field = sum(np.outer(region_weights(points, p, 2), p['direction']) for p in expand(source)); return points, field, dA

    def test_edge_shear_is_self_balanced_and_points_outwards(self):
        centre = np.array([50., 1.]) * 1e-3
        for shape, width, parts, edge_length in (('strip', 0., 1, 2 * 20e-3), ('rectangle', 6., 2, 2 * 6e-3 + 2 * 10e-3), ('disc', 0., 2, 2 * np.pi * 4.5e-3)):
            s = self.source(shape=shape, width_mm=width); self.assertEqual(len(expand(s)), parts); points, F, dA = self.traction(s); r = points[:, :2] - centre
            self.assertLess(np.abs(F.sum(0) * dA).max(), 1e-12); self.assertGreater(float((F[:, :2] * r).sum()), 0.); self.assertTrue(np.all(F[:, 2] == 0))
            # the traction fills the edge zone: its area is the edge width times the length of the edge (the strip spans the 20 mm of the test grid)
            if shape != 'rectangle': self.assertAlmostEqual(np.linalg.norm(F, axis=1).sum() * dA / (1e-3 * edge_length), 1., delta=.03)
            self.assertLess(abs(float((r[:, 0] * F[:, 1] - r[:, 1] * F[:, 0]).sum())) * dA, 1e-12)   # no net moment
        points, F, dA = self.traction(self.source(shape='disc')); r = points[:, :2] - centre; loaded = np.linalg.norm(F, axis=1) > 0; d = np.linalg.norm(r[loaded], axis=1)
        self.assertTrue(np.all((d >= 4e-3 - 1e-9) & (d <= 5e-3 + 1e-9))); self.assertTrue(np.allclose(np.sum(F[loaded][:, :2] * r[loaded], 1) / d, np.linalg.norm(F[loaded], axis=1), rtol=1e-9))   # radial, in the ring
        along_y = self.source(direction=(0., 1., 0.)); points, F, dA = self.traction(along_y); self.assertTrue(np.all(F[:, 0] == 0)); self.assertGreater(np.abs(F[:, 1]).max(), 0)

    def test_other_profiles_expand_to_themselves(self):
        plain = dict(SOURCE, profile={'type': 'gaussian', 'sigma_mm': 2.}); self.assertEqual(expand(plain), [plain]); self.assertEqual(expand(SOURCE), [SOURCE])
        self.assertEqual(len(expand_all([SOURCE, self.source(shape='rectangle', width_mm=6.), dict(SOURCE, profile=validate_profile(WEDGE, SOURCE))])), 1 + 2 + strips(WEDGE, .5))
        with self.assertRaisesRegex(ValueError, 'expand'): region_weights(line()[1], dict(SOURCE, profile=dict(WEDGE)), 2)


class Scenes(unittest.TestCase):
    def scene(self, profile, direction=(0., 0., -1.), duration=40.):
        s = default_scene('plate'); s['schema'] = 'workai.hyperelastic.solid.v2'; s['geometry'].update(length_mm=100., width_mm=2., thickness_mm=2.)
        s['source'].update(face='zmax', center_mm=[50., 1., 2.], radius_mm=0., direction=list(direction), frequency_mhz=.5, cycles=5., delay_us=0., profile=profile)
        s['acquisition'] = {'duration_us': duration, 'sample_step_us': .05}; s['numerics']['h_mm'] = .35
        s['receivers'] = [{'name': 'P', 'point_mm': [80., 1., 2.], 'direction': [0., 0., 1.]}]
        return s

    def test_a_scene_takes_both_and_the_record_must_cover_the_travelling_load(self):
        self.assertEqual(validate_scene(self.scene(WEDGE))['source']['profile'], WEDGE); self.assertEqual(validate_scene(self.scene(PATCH, (1., 0., 0.)))['source']['profile'], PATCH)
        with self.assertRaisesRegex(ValueError, '斜楔底面'): validate_scene(self.scene(WEDGE, duration=12.))   # the burst is 10 μs and the footprint takes 3.7 μs more
        flat = self.scene(WEDGE); flat['dimension'] = '2d_plane_strain'; self.assertEqual(validate_scene(flat)['dimension'], '2d_plane_strain')
        for profile, direction in ((dict(WEDGE, width_mm=1.), (0., 0., -1.)), (dict(PATCH, shape='disc'), (1., 0., 0.))):
            wide = self.scene(profile, direction); wide['dimension'] = '2d_plane_strain'
            with self.assertRaisesRegex(ValueError, '二维平面应变'): validate_scene(wide)


if __name__ == '__main__': unittest.main()
