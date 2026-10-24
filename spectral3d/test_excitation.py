"""Version-2 excitation on GLL meshes: equivalence, symmetry, superposition, absorption."""
import copy
import unittest
import numpy as np
from solid.schema import default_scene
from physics.scene import validate_scene, V2
from physics.signals import evaluate
from .mesh import make_mesh, boundary_data
from .loads import prepare_excitation, face_load, loaded_surface_faces
from .result import surface_triangles
from .solver import solve

def prepared(raw, degree):
    """Validate like the worker: the resolution screen sees h / degree."""
    s = copy.deepcopy(raw); h = s['numerics']['h_mm']; s['numerics']['h_mm'] = h / degree
    out = validate_scene(s); out['numerics']['h_mm'] = h; return out

def plate(length, width, thickness, h, **source):
    s = default_scene('plate'); s['schema'] = V2; s['constitutive'] = 'linear'
    s['geometry'].update(length_mm=length, width_mm=width, thickness_mm=thickness); s['numerics']['h_mm'] = h
    s['source'].update(center_mm=[0., width / 2, thickness / 2], **source)
    s['receivers'] = [{'name': 'P', 'point_mm': [length / 3, width / 2, thickness / 2], 'direction': [1., 0., 0.]}]
    return s

def run(scene, degree, keep_fields=False):
    mesh = make_mesh(scene, degree, scene['numerics']['h_mm'])
    return mesh, solve(scene, mesh, cfl=.3, keep_fields=keep_fields, excitation=prepare_excitation(mesh, scene))

class ExcitationTests(unittest.TestCase):
    def test_version2_without_extensions_matches_version1_path(self):
        raw = plate(12., 4., 2., 2.); v1 = copy.deepcopy(raw); v1['schema'] = 'workai.hyperelastic.solid.v1'
        v1 = prepared(v1, 3); v2 = prepared(raw, 3); self.assertEqual(v2['sources'], [])
        mesh = make_mesh(v1, 3, 2.)
        legacy = solve(v1, mesh, keep_fields=True); modern = solve(v2, mesh, keep_fields=True, excitation=prepare_excitation(mesh, v2))
        self.assertNotIn('excitation_model', legacy['diagnostics']); self.assertEqual(modern['diagnostics']['excitation_model'], 'multi_source_v2')
        # Only the signal sampling differs (vectorized vs scalar transcendental calls), so agreement is to round-off growth.
        np.testing.assert_allclose(modern['fields_m'], legacy['fields_m'], rtol=0, atol=np.max(np.abs(legacy['fields_m'])) * 1e-9)
        self.assertAlmostEqual(modern['diagnostics']['loaded_area_mm2'], legacy['diagnostics']['loaded_area_mm2'], places=9)
        load, _, area = boundary_data(mesh, v1); force, face_area, weighted = face_load(mesh, v2['source'])
        np.testing.assert_allclose(force, load, rtol=1e-13, atol=0); self.assertAlmostEqual(face_area, area); self.assertAlmostEqual(weighted, area)

    def test_symmetry_and_antisymmetry_half_models_reproduce_the_full_plate(self):
        for direction, kind in [([1., 0., 0.], 'symmetry'), ([0., 1., 0.], 'antisymmetry')]:
            full = prepared(plate(12., 4., 2., 2., direction=direction), 3)
            half = plate(12., 2., 2., 2., direction=direction); half['boundaries'] = [{'type': kind, 'face': 'ymax'}]
            half['receivers'][0]['point_mm'] = [4., 1., 1.]; half = prepared(half, 3)
            full_mesh, full_result = run(full, 3, True); half_mesh, half_result = run(half, 3, True)
            self.assertEqual(half_result['diagnostics']['constraints'][0]['type'], kind)
            key = lambda p: tuple(np.round(p * 1e9).astype(int))
            index = {key(p): i for i, p in enumerate(full_mesh['points'])}
            matched = np.array([index[key(p)] for p in half_mesh['points']])
            peak = np.max(np.abs(full_result['fields_m']))
            np.testing.assert_allclose(half_result['fields_m'], full_result['fields_m'][:, matched], atol=peak * 1e-10, rtol=0)
            self.assertLessEqual(full_result['diagnostics']['external_work_balance_relative'], .01)

    def test_two_sources_superpose_in_the_linear_control(self):
        base = plate(12., 4., 2., 2.); base['acquisition']['duration_us'] = 20.
        second = dict(base['source'], name='far end', face='xmax', center_mm=[12., 2., 1.], direction=[-1., 0., .1], amplitude_mpa=.4,
                      frequency_mhz=.2, cycles=1., delay_us=1.5, signal={'type': 'gaussian_pulse', 'bandwidth': 1.})
        both = copy.deepcopy(base); both['sources'] = [second]
        only_second = copy.deepcopy(base); only_second['source']['amplitude_mpa'] = 1e-5; only_second['sources'] = [second]
        results = [run(prepared(s, 3), 3) for s in (base, only_second, both)]
        first, second_only, combined = (r['probes_m'] for _, r in results)
        # The tiny residual primary in `only_second` is removed by scaling.
        residual = first * 1e-5; np.testing.assert_allclose(combined, first + second_only - residual, rtol=1e-10, atol=np.max(np.abs(combined)) * 1e-11)
        self.assertEqual(len(results[2][1]['diagnostics']['sources']), 2); self.assertEqual(results[2][1]['diagnostics']['sources'][1]['name'], 'far end')
        mesh = results[2][0]; faces = loaded_surface_faces(surface_triangles(mesh['elements']), prepare_excitation(mesh, prepared(both, 3))['loads'])
        self.assertGreater(len(faces), 0)

    def test_absorbing_layer_suppresses_the_far_end_echo(self):
        bar = plate(60., 2., 2., 1.15, frequency_mhz=.5, cycles=4.); bar['acquisition'].update(duration_us=32., sample_step_us=.05)
        bar['receivers'][0]['point_mm'] = [10., 1., 1.]
        damped = copy.deepcopy(bar); damped['boundaries'] = [{'type': 'absorbing', 'face': 'xmax', 'length_mm': 20., 'ramp': 'cubic', 'attenuation_db': 60.}]
        free = run(prepared(bar, 3), 3)[1]; absorbed = run(prepared(damped, 3), 3)[1]
        times = free['times_us']; late = times >= 18.; early = (times >= 1.) & (times <= 12.)
        direct = np.max(np.abs(free['probes_m'][early, 0, 0])); echo_free = np.max(np.abs(free['probes_m'][late, 0, 0])); echo_absorbed = np.max(np.abs(absorbed['probes_m'][late, 0, 0]))
        self.assertGreater(echo_free, .3 * direct, 'the free bar must show a clear end reflection')
        self.assertLess(echo_absorbed, .1 * echo_free, f'absorbing layer left {echo_absorbed / echo_free:.3f} of the free echo')
        np.testing.assert_allclose(absorbed['probes_m'][early], free['probes_m'][early], atol=direct * 1e-6, rtol=0)
        d = absorbed['diagnostics']; self.assertLessEqual(d['external_work_balance_relative'], .01); self.assertGreater(d['dissipated_energy_j'], .5 * abs(d['energy_samples'][-1][3]))
        self.assertEqual(d['energy_columns'][-1], 'dissipated_j'); self.assertEqual(d['damping'][0]['face'], 'xmax'); self.assertGreater(d['damping'][0]['alpha_max_per_s'], 1e6)

    def test_profiles_scale_the_loaded_area(self):
        scene = prepared(plate(12., 4., 2., 1.), 2); mesh = make_mesh(scene, 2, 1.)
        comb = dict(scene['source'], face='zmax', center_mm=[6., 2., 2.], direction=[0., 0., 1.],
                    profile={'type': 'comb', 'pitch_mm': 4., 'teeth': 3, 'tooth_width_mm': 1., 'along': 'x', 'alternating_sign': True})
        force, area, weighted = face_load(mesh, comb)
        self.assertAlmostEqual(weighted, 4e-6, places=12)   # teeth +1 -1 +1 of 1 mm x 4 mm; edges on nodes count half
        self.assertGreater(area, weighted); np.testing.assert_allclose(force.sum(0), [0, 0, weighted * 1e6], atol=1e-9)
        gaussian = dict(comb, profile={'type': 'gaussian', 'sigma_mm': 1.}); _, _, g = face_load(mesh, gaussian)
        self.assertLess(g, area); self.assertGreater(g, 0)
        disc = dict(scene['source'], face='zmax', center_mm=[6., 2., 2.], direction=[0., 0., 1.], radius_mm=1.5); _, disc_area, _ = face_load(mesh, disc)
        self.assertLess(disc_area, 48e-6); self.assertGreater(disc_area, 2e-6)

    def test_implicit_hole_solver_accepts_damping_and_balances_energy(self):
        from .implicit import solve as implicit
        s = default_scene('hole'); s['schema'] = V2; s['constitutive'] = 'linear'; s['numerics']['h_mm'] = 1.5; s['source']['cycles'] = 1.
        s['acquisition'].update(duration_us=6., sample_step_us=.05)
        s['boundaries'] = [{'type': 'absorbing', 'face': 'xmax', 'length_mm': 4., 'ramp': 'quadratic', 'attenuation_db': 40.}]
        scene = prepared(s, 3); mesh = make_mesh(scene, 3, 1.5); excitation = prepare_excitation(mesh, scene)
        result = implicit(scene, mesh, substeps=4, excitation=excitation); d = result['diagnostics']
        self.assertEqual(d['excitation_model'], 'multi_source_v2'); self.assertGreater(d['dissipated_energy_j'], 0); self.assertLessEqual(d['external_work_balance_relative'], .01)
        legacy_scene = copy.deepcopy(scene); legacy_scene['schema'] = 'workai.hyperelastic.solid.v1'; legacy_scene.pop('sources'); legacy_scene.pop('boundaries')
        legacy = implicit(legacy_scene, mesh, substeps=4); self.assertEqual(legacy['diagnostics']['excitation_model'] if 'excitation_model' in legacy['diagnostics'] else 'legacy', 'legacy')
        self.assertFalse(np.allclose(legacy['probes_m'], result['probes_m']))

if __name__ == '__main__': unittest.main()
