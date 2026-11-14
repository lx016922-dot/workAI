"""Learned coarse operators through the chunked record: the same answer as the training-time simulation, on the model's own layer."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from graphnet.inputs import network_scene
from graphnet.model import load
from graphnet.rollout import read_record, rollout
from coarsegraph.record import model_layer, projected, setup, simulate
from coarsegraph.run import load_model
from coarsegraph.simulate import Coarse

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / 'coarsegraph/models/p4-h3-harmonic/model.json'


def label_scene(duration=18.):
    """A training-like scene (top Gaussian, absorbing ends, half width) with plain receivers."""
    s = json.loads((REPO / 'coarsegraph/specs/labels-a-top-gaussian.json').read_text(encoding='utf-8'))['base_scene']; s.pop('receiver_lines')
    s['source']['frequency_mhz'] = .3; s['acquisition']['duration_us'] = duration
    s['receivers'] = [{'name': f'top{i}', 'point_mm': [x, 1., 2.], 'direction': [0., 0., 1.]} for i, x in enumerate((22., 28., 34.))] + [{'name': 'mid', 'point_mm': [28., 1., 1.], 'direction': [1., 0., 0.]}]
    return s


class Record(unittest.TestCase):
    def setUp(self):
        self.network = load(4, long_record=True); self.raw = label_scene(); self.scene = network_scene(self.raw, 4, self.network['material'])

    def test_record_reproduces_the_training_time_simulation(self):
        model, (D, q, m) = load_model(MODEL); base = load(3)
        c = Coarse(self.scene, model['degree'], model['h_mm'], model['cfl'])
        relative = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y))
        with tempfile.TemporaryDirectory() as tmp:
            for learned in (True, False):
                operators = c.operators(D, q, m) if learned else (base['weights']['derivative_4'], c.mesh['volume_weight'], c.mesh['mass'])
                laws = ('linear', 'murnaghan') if learned else ('linear',)   # the GLL baseline: one control is enough to see it is the same grid
                reference = {law: c.run(*operators, law, base['weights'])[0] for law in laws}
                manifest = simulate(self.scene, MODEL, Path(tmp) / str(learned), self.network, learned, controls=laws)
                values = projected(read_record(Path(tmp) / str(learned)), self.scene)
                self.assertEqual((manifest['time_plan']['mode'], manifest['time_plan']['subdivisions'], manifest['nodes']), ('given', c.subdivisions, len(c.mesh['mass'])))
                self.assertAlmostEqual(manifest['layer_step_us'], c.dt * 1e6); self.assertLessEqual(manifest['time_plan']['stability_number'], 1.5)
                for law in reference: self.assertLess(relative(values[law], reference[law]), 1e-5, (learned, law))
                if learned: self.assertLess(relative(values['murnaghan'] - values['linear'], reference['murnaghan'] - reference['linear']), 1e-5)
            # learning changes the answer: the two runs above are different computations
            a, b = (projected(read_record(Path(tmp) / str(flag)), self.scene)['linear'] for flag in (True, False)); self.assertGreater(relative(a, b), 1e-4)

    def test_setup_carries_the_learned_operators_and_the_models_layer(self):
        model, (D, q, m) = load_model(MODEL); coarse, mesh, network, layer, _ = setup(self.scene, MODEL, self.network)
        plain, grid, same, _, _ = setup(self.scene, MODEL, self.network, learned=False)
        self.assertEqual((coarse['numerics']['h_mm'], coarse['numerics']['cfl']), (model['h_mm'], model['cfl'])); self.assertEqual(layer, model_layer(coarse, mesh))
        self.assertTrue(np.array_equal(network['weights']['derivative_4'], np.asarray(D))); self.assertIs(same, self.network)
        self.assertFalse(np.array_equal(mesh['volume_weight'], grid['volume_weight'])); self.assertFalse(np.array_equal(mesh['mass'], grid['mass']))
        self.assertAlmostEqual(float(mesh['mass'].sum()), float(grid['mass'].sum()), delta=1e-12 * float(grid['mass'].sum()))   # the learned mass weights keep the mass
        self.assertTrue(np.array_equal(mesh['points'], grid['points'])); self.assertEqual(layer['composition'], [1.])

    def test_a_given_layer_is_checked_recorded_and_resumed(self):
        coarse, mesh, network, layer, _ = setup(self.scene, MODEL, self.network)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError) as unstable: rollout(coarse, mesh, network, root / 'unstable', {'t': 1}, time_plan=dict(layer, subdivisions=1))
            self.assertIn('不稳定', str(unstable.exception))
            for bad in ({'composition': [.5], 'subdivisions': 7}, {'composition': [1.], 'subdivisions': 0}, {'composition': [1.], 'subdivisions': 7.5}):
                with self.assertRaises(ValueError): rollout(coarse, mesh, network, root / 'bad', {'t': 1}, time_plan=bad)
            long = copy.deepcopy(coarse); long['acquisition']['duration_us'] = 30.   # more than one chunk
            first = rollout(long, mesh, network, root / 'resumed', {'t': 1}, time_plan=layer, record='compact', stop_after_chunks=1); self.assertEqual(first['phase'], 'running')
            rollout(long, mesh, network, root / 'resumed', {'t': 1}, time_plan=layer, record='compact'); rollout(long, mesh, network, root / 'whole', {'t': 1}, time_plan=layer, record='compact')
            a, b = read_record(root / 'resumed'), read_record(root / 'whole')
            for key in ('probes', 'energies', 'display', 'snapshots'): self.assertTrue(np.array_equal(a[key], b[key]), key)
            with self.assertRaises(ValueError): rollout(long, mesh, network, root / 'whole', {'t': 1}, time_plan=dict(layer, subdivisions=layer['subdivisions'] + 1), record='compact')   # another layer: not this record


if __name__ == '__main__': unittest.main()
