"""The layer spread over all threads gives the same bits as one thread per control."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from numba import config
from spectral3d.feature_mesh import make_mesh
from graphnet.model import load
from graphnet.inputs import network_scene
from graphnet.parallel import ARRAYS, choose, layer_time, prepare
from graphnet.rollout import GUARD_MESSAGE, read_record, rollout
from graphnet.test_defects import bar, plate as cracked_plate
from graphnet.test_graphnet import excited, plate

ALL = ('linear', 'stvk', 'murnaghan')


def same(a, b):
    return all(np.array_equal(a[key], b[key]) for key in a if key != 'manifest') and set(a) == set(b)


class Choice(unittest.TestCase):
    def test_the_faster_way_is_chosen_for_the_threads_and_controls(self):
        pair, one = ('linear', 'murnaghan'), ('murnaghan',)
        self.assertEqual([layer_time(c, 4, False) for c in (ALL, pair, one)], [1., 1., 1.])   # a thread per control: the complete control sets the time
        self.assertAlmostEqual(layer_time(ALL, 1, False), 2.05); self.assertAlmostEqual(layer_time(ALL, 2, False), 1.025)
        self.assertAlmostEqual(layer_time(ALL, 4, True), 1.2 * 2.05 / 4); self.assertAlmostEqual(layer_time(one, 4, True), .3)
        expected = {(ALL, 1): (False, 1), (ALL, 2): (False, 2), (ALL, 3): (True, 3), (ALL, 4): (True, 4), (pair, 1): (False, 1), (pair, 2): (True, 2), (pair, 8): (True, 8),
                    (one, 1): (False, 1), (one, 2): (True, 2), (one, 4): (True, 4)}
        for (controls, threads), choice in expected.items(): self.assertEqual(choose(controls, threads, 50000), choice, (controls, threads))
        self.assertEqual(choose(one, 8, 200), (False, 1))   # a tiny mesh is not worth the parallel regions

    def test_index_arrays(self):
        model = load(3, long_record=True); scene = network_scene(excited(plate()), 3, model['material']); mesh = make_mesh(scene, 3, 1.6)
        from graphnet.inputs import build
        inputs = build(scene, mesh); x = prepare(mesh, inputs['loads'], inputs['alpha'], np.array([0, 2]), 4); nodes = len(mesh['mass']); elements = len(mesh['elements'])
        self.assertEqual(set(x), set(ARRAYS)); flat = mesh['elements'].reshape(-1)
        for node in (0, nodes // 2, nodes - 1):   # a node's entries are its (element, local index) pairs in element order
            entries = x['entries'][x['pointer'][node]:x['pointer'][node + 1]]
            self.assertTrue(np.all(flat[entries] == node)); self.assertTrue(np.all(np.diff(entries) > 0)); self.assertEqual(len(entries), int(np.sum(flat == node)))
        self.assertEqual(x['pointer'][-1], flat.size)
        for control in (0, 1):   # every element of every control is in exactly one task
            rows = x['element_tasks'][x['element_tasks'][:, 0] == control]; self.assertEqual(rows[0, 1], 0); self.assertEqual(rows[-1, 2], elements); self.assertTrue(np.all(rows[1:, 1] == rows[:-1, 2]))
        self.assertGreater(np.sum(x['element_tasks'][:, 0] == 1), np.sum(x['element_tasks'][:, 0] == 0))   # the complete control costs more: more, smaller tasks
        self.assertEqual((x['node_blocks'][0, 0], x['node_blocks'][-1, 1]), (0, nodes)); self.assertTrue(np.all(x['node_blocks'][1:, 0] == x['node_blocks'][:-1, 1]))
        self.assertTrue(np.all(inputs['alpha'][x['absorbing']] != 0)); self.assertEqual(len(x['absorbing']), int(np.sum(inputs['alpha'] != 0)))
        self.assertEqual(len(x['loaded']), int(np.sum(np.any(inputs['loads'] != 0, axis=(0, 2)))))


class SameBits(unittest.TestCase):
    def run_both(self, scene, mesh, model, **settings):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); a = rollout(scene, mesh, model, root / 'per', {'t': 1}, spread=False, **settings); b = rollout(scene, mesh, model, root / 'spread', {'t': 1}, spread=True, **settings)
            self.assertEqual((a['layer_spread'], b['layer_spread'], a['fingerprint']), (False, True, b['fingerprint'])); self.assertEqual(b['layer_threads'], config.NUMBA_NUM_THREADS)
            x, y = read_record(root / 'per'), read_record(root / 'spread'); self.assertTrue(same(x, y), settings)
            for key in ('max_strain', 'min_J', 'dissipated_energy_j'): self.assertEqual(a[key], b[key])
            return root, x

    def test_damped_multi_source_records_with_every_setting(self):
        model = load(3, long_record=True); scene = network_scene(dict(excited(plate()), acquisition={'duration_us': 15., 'sample_step_us': .05}), 3, model['material']); mesh = make_mesh(scene, 3, 1.6)
        for settings in ({}, {'time_plan': 'auto'}, {'controls': ('linear', 'murnaghan'), 'record': 'compact'}, {'controls': ('murnaghan',), 'time_plan': 'auto', 'record': 'compact'}):
            self.run_both(scene, mesh, model, **settings)
        with tempfile.TemporaryDirectory() as tmp:   # any number of threads, and a record resumed with the other layer
            root = Path(tmp); rollout(scene, mesh, model, root / 'per', {'t': 1}, spread=False); reference = read_record(root / 'per')
            for threads in (1, 2, 3):
                m = rollout(scene, mesh, model, root / f'spread{threads}', {'t': 1}, spread=True, threads=threads)
                self.assertEqual(m['layer_threads'], min(threads, config.NUMBA_NUM_THREADS)); self.assertTrue(same(read_record(root / f'spread{threads}'), reference), threads)
            for first in (False, True):
                rollout(scene, mesh, model, root / f'mixed{first}', {'t': 1}, spread=first, stop_after_chunks=1); m = rollout(scene, mesh, model, root / f'mixed{first}', {'t': 1}, spread=not first)
                self.assertEqual(m['layer_spread'], not first); self.assertTrue(same(read_record(root / f'mixed{first}'), reference), first)
            automatic = rollout(scene, mesh, model, root / 'auto', {'t': 1}); self.assertEqual((automatic['layer_spread'], automatic['layer_threads']), choose(ALL, config.NUMBA_NUM_THREADS, len(mesh['mass'])))
            self.assertTrue(same(read_record(root / 'auto'), reference))

    def test_undamped_plate_matches_the_frozen_network(self):
        from graphwave.long_rollout import rollout as frozen
        model = load(3, long_record=True); s1 = network_scene(plate(duration=15.), 3, model['material']); mesh = make_mesh(s1, 3, 1.6)
        v2 = network_scene(plate(duration=15., v2=True), 3, model['material']); root, record = self.run_both(v2, mesh, model)
        with tempfile.TemporaryDirectory() as tmp:
            frozen(s1, mesh, model['weights'], model['temporal'], Path(tmp) / 'frozen', {'t': 1}); chunks = sorted((Path(tmp) / 'frozen' / 'fields').glob('*.npz'))
            for law in ALL: self.assertTrue(np.array_equal(record[law], np.concatenate([np.load(c)[law] for c in chunks])), law)

    def test_contact_interfaces_curved_elements_and_the_guard(self):
        model = load(3, long_record=True)
        closed = network_scene(bar(1, True), 3, model['material']); mesh = make_mesh(closed, 3, 1.); self.assertEqual(len(mesh['contact_a']), 16)
        self.run_both(closed, mesh, model); self.run_both(network_scene(bar(-1, True), 3, model['material']), mesh, model, controls=('murnaghan',))   # pressed together, and opening
        holed = cracked_plate(); holed['defects'] = [{'type': 'hole', 'axis': 'z', 'x_mm': 6., 'y_mm': 1., 'radius_mm': .4}]
        scene = network_scene(holed, 3, model['material']); mesh = make_mesh(scene, 3, 1.); self.assertEqual(str(mesh['mapping_kind']), 'isoparametric'); self.run_both(scene, mesh, model, time_plan='auto')
        soft = plate(duration=21., v2=True); soft['material'] = {'model': 'murnaghan', 'rho_kg_m3': 2700., 'lambda_pa': 2e9, 'mu_pa': 2e9, 'l_pa': 0., 'm_pa': 0., 'n_pa': 0.}
        soft['source'].update(amplitude_mpa=10., frequency_mhz=.1); soft['numerics']['h_mm'] = .8; scene = network_scene(soft, 3, model['material']); mesh = make_mesh(scene, 3, .8); messages = []
        with tempfile.TemporaryDirectory() as tmp:
            for spread in (False, True):
                with self.assertRaises(ValueError) as caught: rollout(scene, mesh, model, Path(tmp) / str(spread), {'t': 1}, spread=spread)
                messages.append(str(caught.exception))
        self.assertEqual(messages[0], messages[1]); self.assertIn(GUARD_MESSAGE, messages[0])


if __name__ == '__main__': unittest.main()
