"""Plane-strain solver: kernel and transient equivalence with the 3-D solver, symmetry, Lamb S0 check."""
import copy
import unittest
import numpy as np
from solid.schema import default_scene, MATERIAL
from physics.scene import validate_scene, V2
from spectral3d.mesh import make_mesh as make_mesh_3d
from spectral3d.solver import internal_force as internal_force_3d, solve as solve_3d
from spectral3d.loads import prepare_excitation as prepare_3d
from .mesh import make_mesh, interpolation
from .solver import internal_force, solve
from .loads import prepare_excitation, receiver_points, face_load
from .run import simulate, prepared_scene

C = np.array([MATERIAL[k] for k in ['lambda_pa', 'mu_pa', 'l_pa', 'm_pa', 'n_pa']])

def plane(length, thickness, h, **source):
    s = default_scene('plate'); s['schema'] = V2; s['dimension'] = '2d_plane_strain'; s['constitutive'] = 'linear'
    s['geometry'].update(length_mm=length, width_mm=2., thickness_mm=thickness); s['numerics']['h_mm'] = h
    s['source'].update(center_mm=[0., 0., thickness / 2], **source)
    s['receivers'] = [{'name': 'P', 'point_mm': [length / 3, 0., thickness / 2], 'direction': [1., 0., 0.]}]
    return s

class PlaneStrainTests(unittest.TestCase):
    def test_kernel_matches_the_three_dimensional_kernel_for_y_independent_fields(self):
        s = default_scene('plate'); s['geometry'].update(length_mm=6., width_mm=3., thickness_mm=2.)
        mesh3 = make_mesh_3d(s, 3, 1.5); mesh2 = make_mesh(s, 3, 1.5); p3 = mesh3['points']; p2 = mesh2['points']
        field = lambda xz: np.stack([1e-7 * np.sin(xz[:, 0] * 1e3) * np.cos(2e3 * xz[:, 1]) + 2e-5 * xz[:, 0] * xz[:, 1], 1e-7 * np.cos(.7e3 * xz[:, 0]) * xz[:, 1] * 1e3 + 1e-3 * xz[:, 0]**2], 1)
        u2 = field(p2); u3 = field(p3[:, [0, 2]]); u3 = np.stack([u3[:, 0], np.zeros(len(p3)), u3[:, 1]], 1)
        key = lambda xz: tuple(np.round(xz * 1e9).astype(int)); index = {key(q): i for i, q in enumerate(p2)}
        for law in range(3):
            f2, w2, s2, j2 = internal_force(u2, mesh2['elements'], mesh2['derivative'], mesh2['size'], mesh2['area_weight'], C, law)
            f3, w3, s3, j3 = internal_force_3d(u3, mesh3['elements'], mesh3['derivative'], mesh3['size'], mesh3['volume_weight'], C, law)
            summed = np.zeros_like(f2)
            for i, q in enumerate(p3): summed[index[key(q[[0, 2]])]] += f3[i, [0, 2]]
            self.assertAlmostEqual(w3 / (w2 * 3e-3), 1., places=12); np.testing.assert_allclose(summed, f2 * 3e-3, rtol=1e-12, atol=np.max(abs(f2)) * 3e-3 * 1e-13)
            self.assertAlmostEqual(s2, s3); self.assertAlmostEqual(j2, j3)
        rotation = np.array([[np.cos(.4), np.sin(.4)], [-np.sin(.4), np.cos(.4)]]); rigid = p2 @ rotation.T - p2
        f, energy, strain, j = internal_force(rigid, mesh2['elements'], mesh2['derivative'], mesh2['size'], mesh2['area_weight'], C, 2)
        self.assertLess(np.max(abs(f)), 1e-6); self.assertLess(abs(energy), 1e-18); self.assertAlmostEqual(j, 1.)

    def test_transient_matches_three_dimensional_plate_with_symmetric_y_faces(self):
        raw = plane(12., 2., 2.); two = prepared_scene(raw, 3)
        three = copy.deepcopy(raw); three['dimension'] = '3d'; three['receivers'][0]['point_mm'][1] = 1.; three['source']['center_mm'][1] = 1.
        three['boundaries'] = [{'type': 'symmetry', 'face': 'ymin'}, {'type': 'symmetry', 'face': 'ymax'}]
        three = _prepare3d(three, 3)
        mesh2 = make_mesh(two, 3, 2.); mesh3 = make_mesh_3d(three, 3, 2.)
        result2 = solve(two, mesh2, prepare_excitation(mesh2, two), receiver_points(two), keep_fields=True)
        result3 = solve_3d(three, mesh3, keep_fields=True, excitation=prepare_3d(mesh3, three))
        self.assertEqual(result2['diagnostics']['internal_steps'], result3['diagnostics']['internal_steps'])
        key = lambda xz: tuple(np.round(xz * 1e9).astype(int)); index = {key(q): i for i, q in enumerate(mesh2['points'])}
        matched = np.array([index[key(q[[0, 2]])] for q in mesh3['points']])
        peak = np.max(abs(result3['fields_m']))
        np.testing.assert_allclose(result2['fields_m'][:, matched], result3['fields_m'][:, :, [0, 2]], atol=peak * 1e-10, rtol=0)
        self.assertLess(np.max(abs(result3['fields_m'][:, :, 1])), peak * 1e-12)
        np.testing.assert_allclose(result2['probes_m'][:, 0], result3['probes_m'][:, 0, [0, 2]], atol=peak * 1e-10, rtol=0)

    def test_half_thickness_symmetry_isolates_the_symmetric_family(self):
        full = prepared_scene(plane(12., 2., 1.), 3)
        half = plane(12., 1., 1.); half['boundaries'] = [{'type': 'symmetry', 'face': 'zmax'}]; half['source']['center_mm'] = [0., 0., .5]; half['receivers'][0]['point_mm'] = [4., 0., .5]
        half = prepared_scene(half, 3)
        full_run = simulate(full, 3, keep_fields=True); half_run = simulate(half, 3, keep_fields=True)
        key = lambda xz: tuple(np.round(xz * 1e9).astype(int)); index = {key(q): i for i, q in enumerate(full_run['mesh']['points'])}
        matched = np.array([index[key(q)] for q in half_run['mesh']['points']])
        peak = np.max(abs(full_run['computed']['fields_m']))
        np.testing.assert_allclose(half_run['computed']['fields_m'], full_run['computed']['fields_m'][:, matched], atol=peak * 1e-10, rtol=0)
        self.assertEqual(half_run['computed']['diagnostics']['constraints'][0]['type'], 'symmetry')

    def test_s0_phase_velocity_matches_rayleigh_lamb_and_absorber_removes_the_echo(self):
        from dispersion.lamb import dispersion_curves, mode_at
        from dispersion.analysis import line_velocities, harmonic_coefficients
        raw = plane(120., 2., 1., frequency_mhz=.5, cycles=5.); raw['acquisition'].update(duration_us=36., sample_step_us=.05)
        raw['boundaries'] = [{'type': 'absorbing', 'face': 'xmax', 'length_mm': 30., 'ramp': 'cubic', 'attenuation_db': 60.}]
        raw['receiver_lines'] = [{'name': 'top', 'from_mm': [40., 0., 2.], 'to_mm': [80., 0., 2.], 'count': 41, 'direction': [1., 0., 0.]}]
        run = simulate(raw, 4); probes = run['computed']['probes_m']; times = run['computed']['times_us']; receivers = run['receivers']
        line = [i for i, rx in enumerate(receivers) if rx['group'] == 'top']; distances = np.array([receivers[i]['distance_mm'] for i in line])
        traces = probes[:, line, 0]
        curves = dispersion_curves(2., MATERIAL, fd_max_mhz_mm=1.5, points=60); lamb = mode_at(curves, 'S0', .5 * 2.)
        measured = line_velocities(distances, times, traces, .5)
        self.assertLess(abs(measured['phase_velocity_m_s'] - lamb['phase_velocity_m_s']) / lamb['phase_velocity_m_s'], .02, (measured, lamb))
        self.assertGreater(measured['phase_fit_r_squared'], .999)
        self.assertLess(abs(measured['group_velocity_m_s'] - lamb['group_velocity_m_s']) / lamb['group_velocity_m_s'], .05, (measured, lamb))
        d = run['computed']['diagnostics']; self.assertLessEqual(d['external_work_balance_relative'], .01); self.assertGreater(d['dissipated_energy_j_per_m'], 0)
        late = harmonic_coefficients(times, probes[:, line[0], 0], .5, 30., times[-1])['harmonics'][0]['amplitude']
        direct = harmonic_coefficients(times, probes[:, line[0], 0], .5, 4., 20.)['harmonics'][0]['amplitude']
        self.assertLess(late, .05 * direct)

    def test_loads_and_line_receivers(self):
        scene = prepared_scene(plane(12., 2., 1.), 2); mesh = make_mesh(scene, 2, 1.)
        top = dict(scene['source'], face='zmax', center_mm=[6., 0., 2.], direction=[0., 0., 1.], profile={'type': 'comb', 'pitch_mm': 4., 'teeth': 3, 'tooth_width_mm': 1., 'along': 'x', 'alternating_sign': False})
        force, length, weighted = face_load(mesh, top); self.assertAlmostEqual(weighted, 3e-3, places=12); np.testing.assert_allclose(force.sum(0), [0, weighted * 1e6], atol=1e-9)
        disc = dict(scene['source'], face='zmax', center_mm=[6., 5., 2.], direction=[0., 0., 1.], radius_mm=1.); _, disc_length, _ = face_load(mesh, disc); self.assertAlmostEqual(disc_length, 2e-3, places=12)
        lined = copy.deepcopy(scene); lined['receiver_lines'] = [{'name': 'L', 'from_mm': [1., 0., 2.], 'to_mm': [11., 0., 2.], 'count': 6, 'direction': [0., 0., 1.]}]
        rows = receiver_points(prepared_scene(lined, 2)); self.assertEqual(len(rows), 7); self.assertEqual(rows[-1]['distance_mm'], 10.); self.assertEqual(rows[3]['group'], 'L')
        for rx in rows: interpolation(mesh, rx['point_mm'])
        with self.assertRaises(ValueError): interpolation(mesh, [6., 0., 2.5])
        for mutate, message in [(lambda s: s['source'].update(face='ymin'), 'xmin/xmax/zmin/zmax'), (lambda s: s.update(dimension='3d'), 'receiver_lines'),
                                (lambda s: s['receivers'][0].update(direction=[0., 1., 0.]), 'y 分量'), (lambda s: s['geometry'].update(kind='pipe'), '板与贯穿')]:
            bad = copy.deepcopy(lined); mutate(bad)
            with self.assertRaises(ValueError) as caught: prepared_scene(bad, 2)
            self.assertIn(message, str(caught.exception))


def _prepare3d(raw, degree):
    s = copy.deepcopy(raw); h = s['numerics']['h_mm']; s['numerics']['h_mm'] = h / degree; out = validate_scene(s); out['numerics']['h_mm'] = h; return out

if __name__ == '__main__': unittest.main()
