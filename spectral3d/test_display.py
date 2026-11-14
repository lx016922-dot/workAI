"""Display mesh (surface, thinning, clustering) and the array form of the frozen plate mesher."""
import unittest
import numpy as np
from solid.schema import default_scene
from spectral3d.display import display_mesh, frames_shown, loaded_triangles, surface_faces
from spectral3d.feature_mesh import FROZEN_NODES, large_plate_mesh, make_mesh, plate_node_count
from spectral3d.mesh import make_mesh as frozen_mesh
from spectral3d.result import surface_triangles


def scene(kind='plate', **geometry):
    s = default_scene(kind); s['geometry'].update(geometry); return s


class Display(unittest.TestCase):
    def test_full_display_is_the_whole_surface(self):
        for kind in ('plate', 'notch'):
            mesh = frozen_mesh(scene(kind), 4, 1.5); display = display_mesh(mesh); reference = surface_triangles(mesh['elements'])
            self.assertEqual(display['stride'], 1); self.assertIsNone(display['cell_mm']); self.assertEqual(len(display['triangles']), len(reference))
            self.assertEqual(display['surface_faces'] * 2 * 16, len(reference))
            key = lambda t: sorted(tuple(sorted(row)) for row in np.asarray(t).tolist())
            self.assertEqual(key(display['nodes'][display['triangles']]), key(reference))
            self.assertTrue(np.array_equal(display['nodes'], np.unique(reference)))

    def test_thinning_and_clustering_fit_the_budget(self):
        mesh = frozen_mesh(scene(), 4, 1.5); faces = len(surface_faces(mesh['elements']))
        thin = display_mesh(mesh, triangle_budget=faces * 2 * 4); self.assertEqual(thin['stride'], 2); self.assertEqual(len(thin['triangles']), faces * 8)
        corners = display_mesh(mesh, triangle_budget=faces * 2); self.assertEqual(corners['stride'], 4); self.assertIsNone(corners['cell_mm']); self.assertEqual(len(corners['triangles']), faces * 2)
        merged = display_mesh(mesh, triangle_budget=faces); self.assertIsNotNone(merged['cell_mm']); self.assertLessEqual(len(merged['triangles']), faces); self.assertGreater(len(merged['triangles']), 8)
        for display in (thin, corners, merged):
            t = display['triangles']; self.assertTrue(np.all((t >= 0) & (t < len(display['nodes'])))); self.assertTrue(np.all((t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])))
            self.assertTrue(np.all(np.diff(display['nodes']) > 0))
            box = mesh['points'][display['nodes']]; self.assertTrue(np.allclose(box.min(axis=0), mesh['points'].min(axis=0))); self.assertTrue(np.allclose(box.max(axis=0), mesh['points'].max(axis=0)))
        limited = display_mesh(mesh, node_budget=200); self.assertLessEqual(len(limited['nodes']), 200)
        force = np.zeros_like(mesh['points']); top = np.isclose(mesh['points'][:, 2], mesh['points'][:, 2].max()); force[top, 2] = 1.
        loaded = loaded_triangles(corners, [(force, None)]); self.assertTrue(loaded); self.assertLess(len(loaded), len(corners['triangles']))
        self.assertTrue(np.all(top[corners['nodes'][corners['triangles'][loaded]]]))

    def test_animation_frames(self):
        self.assertEqual(len(frames_shown(601, 1000, 4)), 281); self.assertEqual(frames_shown(601, 1000, 4)[[0, -1]].tolist(), [0, 600])
        self.assertEqual(len(frames_shown(100, 1000, 4)), 100); self.assertLess(len(frames_shown(601, 60000, 8)), 281)


class LargePlate(unittest.TestCase):
    def test_array_form_equals_the_frozen_mesh(self):
        for kind in ('plate', 'notch'):
            for degree, h in ((3, 1.4), (4, .9), (5, 1.1)):
                s = scene(kind); a = frozen_mesh(s, degree, h); b = large_plate_mesh(s, degree, h)
                self.assertEqual(set(a), set(b))
                for key in a: self.assertTrue(np.array_equal(a[key], b[key]) and np.asarray(a[key]).dtype == np.asarray(b[key]).dtype, (kind, degree, key))
            self.assertGreaterEqual(plate_node_count(scene(kind), 4, .9)[0], len(frozen_mesh(scene(kind), 4, .9)['points']))
        self.assertEqual(plate_node_count(scene(), 4, .9)[0], len(frozen_mesh(scene(), 4, .9)['points']))

    def test_curved_geometry_equals_the_frozen_function(self):
        from physics.scene import validate_scene
        from spectral3d.curved_mesh import prepare
        from spectral3d.feature_mesh import prepare_curved
        s = scene(length_mm=16., width_mm=4., thickness_mm=2.); s['schema'] = 'workai.hyperelastic.solid.v2'; s['numerics']['h_mm'] = .5
        s['source'].update(center_mm=[0., 2., 1.]); s['receivers'] = [{'name': 'P1', 'point_mm': [12., 2., 1.], 'direction': [1., 0., 0.]}]
        s['defects'] = [{'type': 'hole', 'axis': 'z', 'x_mm': 8., 'y_mm': 2., 'radius_mm': .5}]
        validated = validate_scene(s); mesh = make_mesh(validated, 3, 1.4); self.assertEqual(str(mesh['mapping_kind']), 'isoparametric')
        a = prepare(mesh['points'], mesh['elements'], 3, validated['material']['rho_kg_m3'], 1.4); b = prepare_curved(mesh['points'], mesh['elements'], 3, validated['material']['rho_kg_m3'], 1.4)
        self.assertEqual(set(a), set(b))
        for key in a: self.assertTrue(np.array_equal(a[key], b[key]), key)

    def test_plates_beyond_the_frozen_size_are_meshed(self):
        s = scene(length_mm=120., width_mm=24., thickness_mm=1.6)
        with self.assertRaises(ValueError): frozen_mesh(s, 4, 1.15)
        mesh = make_mesh(s, 4, 1.15); self.assertGreater(len(mesh['points']), FROZEN_NODES); self.assertEqual(len(mesh['points']), plate_node_count(s, 4, 1.15)[0])
        self.assertAlmostEqual(float(mesh['mass'].sum()), 120e-3 * 24e-3 * 1.6e-3 * s['material']['rho_kg_m3'], delta=1e-12)
        display = display_mesh(mesh); self.assertLessEqual(len(display['triangles']), 50000); self.assertLessEqual(len(display['nodes']), 60000)


if __name__ == '__main__': unittest.main()
