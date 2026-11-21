"""Material damping in the long-record layer: the damped layer's own numbers, the plan, and records that keep the
properties of every other record (the same bits on any threads and after a resume, a closed energy balance)."""
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
from spectral3d.feature_mesh import make_mesh
from graphnet.model import load, speed_bound
from graphnet.inputs import build, network_scene
from graphnet.rollout import read_record, rollout
from graphnet.test_graphnet import excited, plate
from graphnet.test_parallel import same
from graphnet.timestep import VERLET, damped_layer, phase_error, plan, stability_limit


def damped(alpha=2e4, beta=3e-9, duration=12.):
    s = excited(plate()); s['acquisition'] = {'duration_us': duration, 'sample_step_us': .05}; s['damping'] = {'mass_per_s': alpha, 'stiffness_s': beta}
    return s


class Layer(unittest.TestCase):
    def test_the_damped_layer_keeps_second_order(self):
        for kappa in (.05, .5, 2.):
            for y in (.01, .05):
                radius, phase, decay = damped_layer(y, kappa); self.assertLess(radius, 1.)
                self.assertLess(abs(phase - phase_error(VERLET, y)), .1 * phase_error(VERLET, y))   # the viscous force adds no first-order phase error
                self.assertLess(abs(decay - y * y / 2), .1 * y * y)                                 # the decay rate is right to second order
        self.assertEqual(damped_layer(3., .5)[1:], (math.inf, math.inf))

    def test_the_stability_limit_falls_with_the_damping(self):
        limits = [stability_limit(VERLET, kappa) for kappa in (0., .05, .5, 2., 8.)]
        self.assertAlmostEqual(limits[0], 2., places=6); self.assertTrue(all(a > b for a, b in zip(limits, limits[1:]))); self.assertGreater(limits[-1], .3)
        for kappa, limit in zip((.05, .5, 2.), limits[1:]):
            self.assertLessEqual(damped_layer(limit * .99, kappa)[0], 1 + 1e-9); self.assertGreater(damped_layer(limit * 1.02, kappa)[0], 1.)
        with self.assertRaises(ValueError): stability_limit((1.35, -1.7, 1.35), .5)

    def test_the_plan(self):
        model = load(3, long_record=True); trained = np.asarray(model['temporal']['weights'])
        scene = network_scene(damped(), 3, model['material']); mesh = make_mesh(scene, 3, 1.6); speed = speed_bound(scene); omega = 2e8
        chosen = plan(scene, mesh, trained, speed, omega, 'auto', beta=3e-9); free = plan(scene, mesh, trained, speed, omega, 'auto')
        self.assertEqual(chosen['scheme'], 'verlet2'); self.assertGreaterEqual(chosen['subdivisions'], free['subdivisions'])
        self.assertLessEqual(chosen['stability_number'], .75 * chosen['stability_limit'] + 1e-9); self.assertLess(chosen['stability_limit'], 2.)
        self.assertLessEqual(max(chosen['phase_error_bound_rad'], chosen['decay_error_bound_np']), .01); self.assertIn('阻尼', chosen['reason'])
        self.assertNotIn('decay_error_bound_np', free)
        strong = plan(scene, mesh, trained, speed, omega, 'auto', beta=3e-8); self.assertGreater(strong['subdivisions'], chosen['subdivisions'])
        with self.assertRaisesRegex(ValueError, '自动时间层'): plan(scene, mesh, trained, speed, omega, 'frozen', beta=3e-9)
        self.assertEqual(plan(scene, mesh, trained, speed, None, 'frozen')['scheme'], 'composition4')


class Records(unittest.TestCase):
    def test_damped_records_are_the_same_bits_on_any_threads_and_after_a_resume(self):
        model = load(3, long_record=True); scene = network_scene(damped(), 3, model['material']); mesh = make_mesh(scene, 3, 1.6)
        self.assertEqual(build(scene, mesh)['beta'], 3e-9); self.assertGreaterEqual(float(build(scene, mesh)['alpha'].min()), 2e4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); a = rollout(scene, mesh, model, root / 'per', {'t': 1}, spread=False, time_plan='auto'); b = rollout(scene, mesh, model, root / 'spread', {'t': 1}, spread=True, time_plan='auto')
            x, y = read_record(root / 'per'), read_record(root / 'spread'); self.assertTrue(same(x, y)); self.assertEqual(a['dissipated_energy_j'], b['dissipated_energy_j'])
            self.assertEqual(a['time_plan']['scheme'], 'verlet2'); self.assertEqual(a['material_damping']['stiffness_s'], 3e-9)
            for first in (False, True):   # stopped after one chunk and continued with the other layer
                rollout(scene, mesh, model, root / f'r{first}', {'t': 1}, spread=first, time_plan='auto', stop_after_chunks=1)
                self.assertTrue(same(read_record(rollout(scene, mesh, model, root / f'r{first}', {'t': 1}, spread=not first, time_plan='auto') and root / f'r{first}'), x), first)
            compact = rollout(scene, mesh, model, root / 'compact', {'t': 1}, time_plan='auto', record='compact', controls=('linear', 'murnaghan'))
            self.assertTrue(np.array_equal(read_record(root / 'compact')['probes'][:, :, 0], x['probes'][:, :, 0])); self.assertEqual(compact['controls'], ['linear', 'murnaghan'])
            # the energy balance closes with the dissipation counted: kinetic + strain + dissipated = work
            e = x['energies']; balance = np.abs(e[:, :, 1] + e[:, :, 2] + e[:, :, 4] - e[:, :, 3]).max() / np.abs(e[:, :, 3]).max(); self.assertLess(balance, 2e-3)
            # more is dissipated than by the absorbing layer alone, and the signal is smaller
            plain = dict(scene); plain.pop('damping'); rollout(plain, mesh, model, root / 'plain', {'t': 1}, time_plan='auto'); p = read_record(root / 'plain')
            self.assertTrue(np.all(e[:, -1, 4] > p['energies'][:, -1, 4])); self.assertLess(np.abs(x['probes'][:, 0, 0]).max(), np.abs(p['probes'][:, 0, 0]).max())
            self.assertNotEqual(a['fingerprint'], rollout(plain, mesh, model, root / 'plain', {'t': 1}, time_plan='auto')['fingerprint'])

    def test_the_trained_layer_takes_mass_proportional_damping_only(self):
        model = load(3, long_record=True)
        with tempfile.TemporaryDirectory() as tmp:
            scene = network_scene(damped(duration=10.), 3, model['material']); mesh = make_mesh(scene, 3, 1.6)
            with self.assertRaisesRegex(ValueError, '自动时间层'): rollout(scene, mesh, model, Path(tmp) / 'frozen', {'t': 1})
            with self.assertRaisesRegex(ValueError, '单级时间层'): rollout(scene, mesh, model, Path(tmp) / 'given', {'t': 1}, time_plan={'composition': model['temporal']['weights'], 'subdivisions': 30})
            mass = network_scene(damped(beta=0., duration=10.), 3, model['material']); m = rollout(mass, mesh, model, Path(tmp) / 'mass', {'t': 1})
            self.assertEqual((m['time_plan']['scheme'], m['phase']), ('composition4', 'completed')); self.assertGreater(min(m['dissipated_energy_j']), 0.)


if __name__ == '__main__': unittest.main()
