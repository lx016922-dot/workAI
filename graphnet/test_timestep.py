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
        with self.assertRaises(ValueError): plan(scene, mesh, composition, speed_bound(scene), omega, mode='other')
        with self.assertRaises(ValueError): plan(scene, mesh, composition, speed_bound(scene))   # auto needs the measured frequency


class Interfaces(unittest.TestCase):
    """Contact interfaces use the automatic layer too: same interface stiffness, their springs in the measured ω_max."""
    def setUp(self):
        from graphnet.test_defects import bar
        from spectral3d.feature_mesh import make_mesh as featured
        self.model = load(3, long_record=True); self.scene = network_scene(bar(1, True), 3, self.model['material']); self.mesh = featured(self.scene, 3, 1.)
        self.opening = network_scene(bar(-1, True), 3, self.model['material'])

    def test_closed_springs_raise_the_measured_frequency_within_their_own_bound(self):
        from graphnet.contact import STABILITY, arrays, stiffness
        from graphnet.timestep import frozen_plan
        from graphwave.buffered_inference import graph_message
        model, scene, mesh = self.model, self.scene, self.mesh; inputs = build(scene, mesh); q, c, _ = energy_layer(scene, model); D = model['weights']['derivative_3']; inverse = inverse_mapping(mesh)
        step = frozen_plan(scene, mesh, model['temporal']['weights'], speed_bound(scene))['layer_step_us'] * 1e-6
        a, b, normal, _ = arrays(mesh); k, summary = stiffness(mesh, scene, step); self.assertEqual(len(a), 16); self.assertAlmostEqual(summary['omega_dt_max'], STABILITY)
        free = largest_frequency(mesh, D, inverse, q, c, inputs['free']); closed = largest_frequency(mesh, D, inverse, q, c, inputs['free'], contact=(a, b, normal, k))
        self.assertGreater(closed, free); self.assertLess(closed, 1.02 * math.hypot(free, STABILITY / step))   # a spring adds at most its own ω² to ω_max²
        n = len(mesh['points']) * 3; scale = np.repeat(1 / np.sqrt(mesh['mass']), 3) * inputs['free'].ravel(); K = np.zeros((n, n))
        for i in range(n):
            unit = np.zeros(n); unit[i] = scale[i]; u = unit.reshape(-1, 3); force = graph_message(u, mesh['elements'], D, inverse, mesh['volume_weight'], q, c, 0)[0]
            push = (k * np.sum((u[b] - u[a]) * normal, axis=1))[:, None] * normal; np.add.at(force, b, push); np.add.at(force, a, -push); K[:, i] = force.ravel() * scale
        exact = math.sqrt(np.linalg.eigvalsh((K + K.T) / 2)[-1]); self.assertGreaterEqual(closed, exact); self.assertLess(closed / exact, 1.03)

    def test_automatic_layer_keeps_the_interface_and_agrees_with_the_trained_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, scene in (('pressed', self.scene), ('opening', self.opening)):
                frozen = rollout(scene, self.mesh, self.model, root / f'{name}-frozen', {'t': 1}); auto = rollout(scene, self.mesh, self.model, root / f'{name}-auto', {'t': 1}, time_plan='auto')
                a, c = read_record(root / f'{name}-frozen'), read_record(root / f'{name}-auto')
                self.assertEqual(auto['time_plan']['scheme'], 'verlet2'); self.assertLess(auto['time_plan']['evaluations_per_control'], frozen['time_plan']['evaluations_per_control'] / 2)
                self.assertLessEqual(auto['time_plan']['stability_number'], STABILITY_FRACTION * 2 + 1e-9)
                # the same interface: stiffness of the trained layer spacing, whatever the layer
                for key in ('pairs', 'stiffness_per_area_n_m3', 'compression_reflection', 'stiffness_layer_step_us'): self.assertEqual(auto['contact'][key], frozen['contact'][key], key)
                self.assertAlmostEqual(frozen['contact']['layer_omega_dt_max'], .6); self.assertAlmostEqual(auto['contact']['layer_omega_dt_max'], .6 * auto['layer_step_us'] / frozen['layer_step_us'])
                relative = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y))
                before = relative(c['probes'][:, 0, 2], a['probes'][:, 0, 2]); self.assertLess(before, 1e-3, name)
                if name == 'pressed': self.assertLess(relative(c['probes'][:, 1, 2], a['probes'][:, 1, 2]), 1e-3)   # compression passes the interface
                else: self.assertLess(np.max(np.abs(c['probes'][:, 1, 2])), 1e-6 * np.max(np.abs(c['probes'][:, 0, 2])))   # tension opens it: nothing arrives
                e = c['energies'][2]; stored = e[:, 1] + e[:, 2] + e[:, 4]; self.assertLess(np.max(np.abs(stored - e[:, 3])) / np.max(np.abs(e[:, 3])), 1e-3)


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

    def test_selected_controls_equal_the_same_controls_of_a_run_with_all_three(self):
        from graphnet.rollout import display_parts
        self.assertEqual([p[0] for p in display_parts(('linear', 'stvk', 'murnaghan'))], ['linear', 'geometric', 'material'])
        self.assertEqual(display_parts(('linear', 'murnaghan')), (('linear', 'linear', None), ('nonlinear', 'murnaghan', 'linear')))
        self.assertEqual(display_parts(('murnaghan',)), (('full', 'murnaghan', None),))
        model, scene, mesh = setup(excited(plate()))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); identity = {'test': 'controls'}
            rollout(scene, mesh, model, root / 'all', identity); a = read_record(root / 'all')
            for name, controls, rows in (('pair', ('linear', 'murnaghan'), [0, 2]), ('full', ('murnaghan',), [2])):
                m = rollout(scene, mesh, model, root / name, identity, controls=controls); b = read_record(root / name)
                self.assertEqual(m['controls'], list(controls)); self.assertEqual(len(m['max_strain']), len(controls)); self.assertNotIn('stvk', b)
                for control in controls: self.assertTrue(np.array_equal(b[control], a[control]), (name, control))
                self.assertTrue(np.array_equal(b['probes'], a['probes'][:, :, rows])); self.assertTrue(np.array_equal(b['energies'], a['energies'][rows]))
                size = lambda d: sum(f.stat().st_size for f in (root / d).rglob('*.npz') if f.name != 'mesh.npz')
                self.assertAlmostEqual(size(name) / size('all'), len(controls) / 3, delta=.02)
                # the compact record of the same controls: receivers bit-identical, display holds the response and its increment
                c = rollout(scene, mesh, model, root / (name + '-compact'), identity, controls=controls, record='compact'); d = read_record(root / (name + '-compact'))
                self.assertTrue(np.array_equal(d['probes'], b['probes'])); frames = c['display']['frames']; nodes = np.load(root / (name + '-compact') / 'display.npz')['nodes']
                self.assertEqual(c['display']['parts'], ['linear', 'nonlinear'] if len(controls) == 2 else ['full'])
                expected = [a['linear'], a['murnaghan'] - a['linear']] if len(controls) == 2 else [a['murnaghan']]
                for stored, exact in zip(d['display'], expected): self.assertTrue(np.array_equal(stored, exact[frames][:, nodes].astype(np.float32)))
            with self.assertRaises(ValueError): rollout(scene, mesh, model, root / 'all', identity, controls=('murnaghan',))   # another setting: not the saved record
            for bad in ((), ('murnaghan', 'linear'), ('linear', 'linear'), ('other',)):
                with self.assertRaises(ValueError): rollout(scene, mesh, model, root / 'bad', identity, controls=bad)

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
