"""Version-2 excitation on tetrahedral meshes with hand-built, mirror-symmetric boxes."""
import copy
import itertools
import unittest
import numpy as np
from .schema import default_scene
from .mesh import prepare_mesh, nodal_load, prepare_excitation, face_selection
from .fem import solve
from physics.scene import validate_scene, V2

def box_mesh(length, width, thickness, cell):
    """Kuhn (six tetrahedra per cube) box in millimetres; conforming and translation invariant."""
    nx, ny, nz = (int(round(v / cell)) for v in (length, width, thickness))
    grid = np.array([[i, j, k] for i in range(nx + 1) for j in range(ny + 1) for k in range(nz + 1)], dtype=float) * cell
    index = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k
    tets = []
    for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
        corner = lambda d: index(i + d[0], j + d[1], k + d[2])
        for order in itertools.permutations(range(3)):
            path = [np.zeros(3, dtype=int)]
            for axis in order: path.append(path[-1] + np.eye(3, dtype=int)[axis])
            tets.append([corner(p) for p in path])
    return grid, np.array(tets)

def mirrored(points, tetra, axis, bound):
    """Reflect a mesh about the plane coordinate = bound; shared plane nodes are merged."""
    reflected = points.copy(); reflected[:, axis] = 2 * bound - reflected[:, axis]
    key = lambda p: tuple(np.round(p * 1e9).astype(int)); index = {key(p): i for i, p in enumerate(points)}; merged = list(points); mapping = []
    for p in reflected:
        if key(p) in index: mapping.append(index[key(p)])
        else: index[key(p)] = len(merged); merged.append(p); mapping.append(index[key(p)])
    mapping = np.array(mapping)
    return np.array(merged), np.vstack([tetra, mapping[tetra]])

def scene_v2(length, width, thickness, **source):
    s = default_scene('plate'); s['schema'] = V2; s['constitutive'] = 'linear'; s['numerics']['h_mm'] = .3
    s['geometry'].update(length_mm=length, width_mm=width, thickness_mm=thickness)
    s['source'].update(center_mm=[0., width / 2, thickness / 2], **source)
    s['receivers'] = [{'name': 'P', 'point_mm': [length / 3, width / 2, thickness / 2], 'direction': [1., 0., 0.]}]
    return s

def run(scene, points, tetra):
    mesh = prepare_mesh(points * 1e-3, tetra); excitation = prepare_excitation(mesh, scene)
    fields, diagnostics = solve(scene, mesh, None, None, output_precision='float64', excitation=excitation)
    return mesh, fields, diagnostics

class TetraExcitationTests(unittest.TestCase):
    def test_version2_without_extensions_matches_version1_path(self):
        raw = scene_v2(6., 2., 2.); v1 = copy.deepcopy(raw); v1['schema'] = 'workai.hyperelastic.solid.v1'
        v1 = validate_scene(v1); v2 = validate_scene(raw); points, tetra = box_mesh(6., 2., 2., 1.); mesh = prepare_mesh(points * 1e-3, tetra)
        load, selected = nodal_load(mesh, v1['source']); legacy, first = solve(v1, mesh, load, np.array([], dtype=int), output_precision='float64')
        modern, second = solve(v2, mesh, None, None, output_precision='float64', excitation=prepare_excitation(mesh, v2))
        np.testing.assert_allclose(modern, legacy, rtol=0, atol=np.max(np.abs(legacy)) * 1e-9); self.assertAlmostEqual(first['external_work_balance_relative'], second['external_work_balance_relative'], places=9)
        self.assertEqual(second['excitation_model'], 'multi_source_v2'); self.assertNotIn('excitation_model', first)

    def test_symmetry_half_model_matches_mirrored_full_model(self):
        for direction, kind in [([1., 0., 0.], 'symmetry'), ([0., 1., 0.], 'antisymmetry')]:
            half_points, half_tetra = box_mesh(6., 1., 2., 1.); full_points, full_tetra = mirrored(half_points, half_tetra, 1, 1.)
            full = validate_scene(scene_v2(6., 2., 2., direction=direction))
            half = scene_v2(6., 1., 2., direction=direction); half['boundaries'] = [{'type': kind, 'face': 'ymax'}]; half['receivers'][0]['point_mm'] = [2., .5, 1.]
            half = validate_scene(half)
            _, full_fields, full_diag = run(full, full_points, full_tetra); _, half_fields, half_diag = run(half, half_points, half_tetra)
            self.assertEqual(half_diag['constraints'][0], {'type': kind, 'face': 'ymax', 'nodes': 21})
            peak = np.max(np.abs(full_fields)); np.testing.assert_allclose(half_fields, full_fields[:, :len(half_points)], atol=peak * 1e-9, rtol=0)
            self.assertLessEqual(full_diag['external_work_balance_relative'], .01)

    def test_two_sources_superpose_in_the_linear_control(self):
        points, tetra = box_mesh(6., 2., 2., 1.); base = scene_v2(6., 2., 2.); base['acquisition']['duration_us'] = 20.
        second = dict(base['source'], name='top patch', face='zmax', center_mm=[3., 1., 2.], radius_mm=1.2, direction=[0., 0., 1.], amplitude_mpa=.3, frequency_mhz=.2, cycles=1., delay_us=2., phase_deg=0.,
                      signal={'type': 'gaussian_pulse', 'bandwidth': 1.})
        both = copy.deepcopy(base); both['sources'] = [second]; tiny = copy.deepcopy(both); tiny['source']['amplitude_mpa'] = 1e-5
        a = run(validate_scene(base), points, tetra)[1]; b = run(validate_scene(tiny), points, tetra)[1]; ab = run(validate_scene(both), points, tetra)[1]
        np.testing.assert_allclose(ab, a + b - a * 1e-5, rtol=1e-9, atol=np.max(np.abs(ab)) * 1e-10)

    def test_absorbing_layer_suppresses_the_far_end_echo(self):
        points, tetra = box_mesh(60., 2., 2., 1.); bar = scene_v2(60., 2., 2., frequency_mhz=.5, cycles=4.); bar['acquisition'].update(duration_us=32., sample_step_us=.1)
        bar['receivers'][0]['point_mm'] = [10., 1., 1.]
        damped = copy.deepcopy(bar); damped['boundaries'] = [{'type': 'absorbing', 'face': 'xmax', 'length_mm': 20., 'ramp': 'cubic', 'attenuation_db': 60.}]
        mesh, free_fields, _ = run(validate_scene(bar), points, tetra); _, damped_fields, d = run(validate_scene(damped), points, tetra)
        node = int(np.argmin(np.linalg.norm(mesh['points'] - np.array([.010, .001, .001]), axis=1))); times = np.arange(len(free_fields)) * .1
        late = times >= 18.; early = (times >= 1.) & (times <= 12.)
        direct = np.max(np.abs(free_fields[early, node, 0])); echo_free = np.max(np.abs(free_fields[late, node, 0])); echo_damped = np.max(np.abs(damped_fields[late, node, 0]))
        self.assertGreater(echo_free, .3 * direct); self.assertLess(echo_damped, .1 * echo_free, f'absorbing layer left {echo_damped / echo_free:.3f} of the echo')
        np.testing.assert_allclose(damped_fields[early, node], free_fields[early, node], atol=direct * 1e-6, rtol=0)
        self.assertLessEqual(d['external_work_balance_relative'], .01); self.assertGreater(d['dissipated_energy_j'], 0); self.assertEqual(d['damping'][0]['ramp'], 'cubic')

    def test_profile_and_conflict_checks(self):
        points, tetra = box_mesh(6., 2., 2., .5); mesh = prepare_mesh(points * 1e-3, tetra); scene = validate_scene(scene_v2(6., 2., 2.))
        comb = dict(scene['source'], face='zmax', center_mm=[3., 1., 2.], direction=[0., 0., 1.], profile={'type': 'comb', 'pitch_mm': 2., 'teeth': 3, 'tooth_width_mm': 1., 'along': 'x', 'alternating_sign': False})
        force, selected = nodal_load(mesh, comb); self.assertAlmostEqual(force.sum(0)[2], 3 * 1e-3 * 2e-3 * 1e6, places=9); self.assertEqual(selected.sum(), 3 * 2 * 4 * 2)
        plain, mask = nodal_load(mesh, scene['source']); self.assertEqual(mask.sum(), face_selection(mesh, 'xmin').sum())
        conflict = copy.deepcopy(scene); conflict['source']['direction'] = [0., 1., 0.]; conflict['boundaries'] = [{'type': 'symmetry', 'face': 'ymax'}]
        with self.assertRaises(ValueError): prepare_excitation(mesh, validate_scene(conflict))

if __name__ == '__main__': unittest.main()
