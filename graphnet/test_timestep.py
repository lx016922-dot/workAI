"""Time layer plan (stability, phase error, measured ω_max) and compact records."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
from spectral3d.mesh import make_mesh
from graphnet.model import load, energy_layer, speed_bound
from graphnet.inputs import network_scene, build
from graphnet.rollout import rollout, read_record
from graphnet.timestep import VERLET, PHASE_TOLERANCE_RAD, STABILITY_FRACTION, largest_frequency, phase_error, plan, stability_limit
from graphnet.test_graphnet import plate, excited
from graphwave.inference import inverse_mapping

YOSHIDA = (1.3512071919596575, -1.702414383919315, 1.3512071919596575)


def setup(scene, degree=3, h=1.6):
    model = load(degree, long_record=True); scene = network_scene(scene, degree, model['material']); mesh = make_mesh(scene, degree, h)
    return model, scene, mesh


class Layers(unittest.TestCase):
    def test_stability_limits_and_orders_of_the_two_layers(self):
        self.assertAlmostEqual(stability_limit(VERLET), 2., places=6); self.assertAlmostEqual(stability_limit(YOSHIDA), 1.5734, places=3)
        self.assertEqual(phase_error(VERLET, 2.1), math.inf); self.assertEqual(phase_error(YOSHIDA, 1.6), math.inf)
        # halving ω·Δt divides the phase error by 4 (second order) and by 16 (fourth order)
        self.assertAlmostEqual(phase_error(VERLET, .1) / phase_error(VERLET, .05), 4., delta=.02)
        self.assertAlmostEqual(phase_error(YOSHIDA, .1) / phase_error(YOSHIDA, .05), 16., delta=.2)
        self.assertAlmostEqual(phase_error(VERLET, .1), .1 ** 2 / 24, delta=1e-6)
        model = load(3, long_record=True); self.assertTrue(np.allclose(model['temporal']['weights'], YOSHIDA, rtol=0, atol=1e-15))

    def test_measured_largest_frequency_matches_the_dense_eigenvalue(self):
        from graphwave.buffered_inference import graph_message
        model, scene, mesh = setup(excited(plate(length=4.)), 3, 1.6); inputs = build(scene, mesh)
        q, c, _ = energy_layer(scene, model); inverse = inverse_mapping(mesh); D = model['weights']['derivative_3']
        omega = largest_frequency(mesh, D, inverse, q, c, inputs['free'])
        n = len(mesh['points']) * 3; scale = (np.repeat(1 / np.sqrt(mesh['mass']), 3) * inputs['free'].ravel()); K = np.zeros((n, n))
        for i in range(n):
            unit = np.zeros(n); unit[i] = scale[i]; K[:, i] = graph_message(unit.reshape(-1, 3), mesh['elements'], D, inverse, mesh['volume_weight'], q, c, 0)[0].ravel() * scale
        exact = math.sqrt(np.linalg.eigvalsh((K + K.T) / 2)[-1])
        self.assertGreaterEqual(omega, exact); self.assertLess(omega / exact, 1.03)   # an upper estimate, within the 3 % margin

    def test_the_plan_picks_the_cheapest_layer_that_meets_both_conditions(self):
        narrow = plate(duration=30., v2=True); narrow['source'].update(cycles=5.)   # a five-cycle burst: narrow band
        model, scene, mesh = setup(narrow); composition = model['temporal']['weights']; inputs = build(scene, mesh)
        q, c, _ = energy_layer(scene, model); omega = largest_frequency(mesh, model['weights']['derivative_3'], inverse_mapping(mesh), q, c, inputs['free'])
        frozen = plan(scene, mesh, composition, speed_bound(scene), mode='frozen')
        self.assertEqual((frozen['scheme'], frozen['time_order'], len(frozen['composition'])), ('composition4', 4, 3))
        short = plan(scene, mesh, composition, speed_bound(scene), omega)
        self.assertEqual(short['scheme'], 'verlet2'); self.assertLess(short['evaluations_per_control'], frozen['evaluations_per_control'] / 2)
        self.assertLessEqual(short['stability_number'], STABILITY_FRACTION * 2 + 1e-9); self.assertLessEqual(short['phase_error_bound_rad'], PHASE_TOLERANCE_RAD)
        # a longer record of the same model: the second-order layer would need too many layers, the fourth-order one takes over with a larger spacing than the trained one
        longer = copy.deepcopy(scene); longer['acquisition'] = {'duration_us': 400., 'sample_step_us': .05}
        row = plan(longer, mesh, composition, speed_bound(longer), omega); trained = plan(longer, mesh, composition, speed_bound(longer), mode='frozen')
        self.assertEqual(row['scheme'], 'composition4'); self.assertLessEqual(row['phase_error_bound_rad'], PHASE_TOLERANCE_RAD)
        self.assertLessEqual(row['stability_number'], STABILITY_FRACTION * stability_limit(YOSHIDA) + 1e-9)
        self.assertLess(row['evaluations_per_control'], trained['evaluations_per_control']); self.assertGreater(row['layer_step_us'], trained['layer_step_us'])
        # never dearer than the trained layer, whatever the record length; its own bound is then reported
        for duration, step in ((5000., .2), (20000., .2)):
            late = copy.deepcopy(scene); late['acquisition'] = {'duration_us': duration, 'sample_step_us': step}
            row = plan(late, mesh, composition, speed_bound(late), omega); trained = plan(late, mesh, composition, speed_bound(late), mode='frozen')
            self.assertLessEqual(row['evaluations_per_control'], trained['evaluations_per_control']); self.assertIn('phase_error_bound_rad', row)
        contact = dict(mesh, contact_a=np.array([0]), contact_b=np.array([1]))   # interface stiffness is tied to the layer spacing: keep the trained layer
        self.assertEqual(plan(scene, contact, composition, speed_bound(scene), omega)['subdivisions'], frozen['subdivisions'])
        with self.assertRaises(ValueError): plan(scene, mesh, composition, speed_bound(scene), omega, mode='other')
        with self.assertRaises(ValueError): plan(scene, mesh, composition, speed_bound(scene))   # auto needs the measured frequency


class Records(unittest.TestCase):
    def test_automatic_layer_and_compact_record_agree_with_the_frozen_full_record(self):
        model, scene, mesh = setup(dict(excited(plate()), acquisition={'duration_us': 15., 'sample_step_us': .05}))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); identity = {'test': 'timestep'}
            frozen = rollout(scene, mesh, model, root / 'frozen', identity); a = read_record(root / 'frozen')
            self.assertEqual((frozen['time_plan']['mode'], frozen['record_mode'], frozen['time_order']), ('frozen', 'full', 4))
            # compact record of the same layer: receivers bit-identical, display and snapshots are the full field at the kept nodes and samples
            compact = rollout(scene, mesh, model, root / 'compact', identity, record='compact'); b = read_record(root / 'compact')
            self.assertTrue(np.array_equal(a['probes'], b['probes'])); self.assertTrue(np.array_equal(a['energies'], b['energies']))
            display = np.load(root / 'compact' / 'display.npz'); frames = compact['display']['frames']; nodes = display['nodes']
            self.assertEqual(b['display'].dtype, np.float32); self.assertEqual(b['display'].shape, (3, len(frames), len(nodes), 3))
            linear, stvk, full = (a[k][frames][:, nodes] for k in ('linear', 'stvk', 'murnaghan'))
            for stored, exact in zip(b['display'], (linear, stvk - linear, full - stvk)): self.assertTrue(np.array_equal(stored, exact.astype(np.float32)))
            self.assertEqual(compact['snapshot_frames'][-1], compact['frame_count'] - 1)
            for law, name in enumerate(('linear', 'stvk', 'murnaghan')): self.assertTrue(np.array_equal(b['snapshots'][law], a[name][compact['snapshot_frames']]))
            size = lambda d: sum(f.stat().st_size for f in (root / d / 'fields').glob('*.npz'))
            self.assertLess(size('compact'), size('frozen') / 2)
            # automatic layer: fewer element messages, same answer to well below the mesh error, energy balance closed
            # (this broad-band scene gets the fourth-order layer at a larger spacing; the narrow-band one below gets one stage)
            auto = rollout(scene, mesh, model, root / 'auto', identity, time_plan='auto'); c = read_record(root / 'auto')
            self.assertEqual((auto['time_plan']['scheme'], auto['time_order']), ('composition4', 4)); self.assertGreater(auto['layer_step_us'], frozen['layer_step_us'])
            self.assertLess(auto['time_plan']['evaluations_per_control'], frozen['time_plan']['evaluations_per_control'] / 2)
            relative = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y))
            self.assertLess(relative(c['murnaghan'], a['murnaghan']), 1e-3); self.assertLess(relative(c['murnaghan'] - c['linear'], a['murnaghan'] - a['linear']), 1e-3)
            e = c['energies'][2]; stored = e[:, 1] + e[:, 2] + e[:, 4]; self.assertLess(np.max(np.abs(stored - e[:, 3])) / np.max(np.abs(e[:, 3])), 1e-3)
            # a compact automatic record resumes bit-identically and refuses another setting
            partial = rollout(scene, mesh, model, root / 'resumed', identity, stop_after_chunks=1, time_plan='auto', record='compact'); self.assertEqual(partial['phase'], 'running')
            rollout(scene, mesh, model, root / 'resumed', identity, time_plan='auto', record='compact'); rollout(scene, mesh, model, root / 'whole', identity, time_plan='auto', record='compact')
            x, y = read_record(root / 'resumed'), read_record(root / 'whole')
            for key in ('probes', 'energies', 'display', 'snapshots'): self.assertTrue(np.array_equal(x[key], y[key]), key)
            self.assertTrue(np.array_equal(x['probes'], c['probes']))
            with self.assertRaises(ValueError): rollout(scene, mesh, model, root / 'whole', identity, time_plan='frozen', record='compact')
            with self.assertRaises(ValueError): rollout(scene, mesh, model, root / 'other', identity, record='partial')

    def test_one_stage_layer_on_a_narrow_band_record_agrees_with_the_frozen_layer(self):
        narrow = excited(plate()); narrow['source'].update(cycles=5.); narrow['sources'][0].update(cycles=5.); narrow['acquisition'] = {'duration_us': 30., 'sample_step_us': .05}
        model, scene, mesh = setup(narrow)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); frozen = rollout(scene, mesh, model, root / 'frozen', {'t': 1}); auto = rollout(scene, mesh, model, root / 'auto', {'t': 1}, time_plan='auto')
            a, c = read_record(root / 'frozen'), read_record(root / 'auto')
            self.assertEqual((auto['time_plan']['scheme'], auto['time_order'], len(auto['composition'])), ('verlet2', 2, 1))
            self.assertLess(auto['time_plan']['evaluations_per_control'], frozen['time_plan']['evaluations_per_control'] / 2)
            relative = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y))
            self.assertLess(relative(c['murnaghan'], a['murnaghan']), 1e-3); self.assertLess(relative(c['murnaghan'] - c['linear'], a['murnaghan'] - a['linear']), 1e-3)
            self.assertLess(relative(c['probes'], a['probes']), 1e-3)
            e = c['energies'][2]; stored = e[:, 1] + e[:, 2] + e[:, 4]; self.assertLess(np.max(np.abs(stored - e[:, 3])) / np.max(np.abs(e[:, 3])), 1e-3)


if __name__ == '__main__': unittest.main()
