import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from solid.schema import default_scene
from spectral3d.mesh import make_mesh
from graphnet.model import load, material_check
from graphnet.inputs import network_scene, build, signal_table
from graphnet.predict import predict, layers
from graphnet.recurrence import forward
from graphwave.inference import inverse_mapping

REPO = Path(__file__).resolve().parents[1]


def plate(length=8., duration=10., v2=False):
    s = default_scene('plate'); s['geometry'].update(length_mm=length, width_mm=3., thickness_mm=2.)
    s['source'].update(center_mm=[0., 1.5, 1.], frequency_mhz=.25, cycles=2., delay_us=.2)
    s['acquisition'] = {'duration_us': duration, 'sample_step_us': .05}; s['numerics']['h_mm'] = 1.6
    s['receivers'] = [{'name': 'P1', 'point_mm': [3., 1.5, 1.], 'direction': [1., 0., 0.]}]
    if v2: s['schema'] = 'workai.hyperelastic.solid.v2'
    return s


def excited(s):
    """Two sources (end face + Gaussian patch on top), a symmetry plane and an absorbing layer."""
    s = copy.deepcopy(s); s['schema'] = 'workai.hyperelastic.solid.v2'; s['geometry'].update(length_mm=12.)
    s['source'].update(signal={'type': 'tone_burst', 'window': 'tukey', 'tukey_alpha': .5})
    s['sources'] = [{'name': 'top', 'face': 'zmax', 'center_mm': [5., 1.5, 2.], 'radius_mm': 0., 'direction': [0., 0., -1.], 'amplitude_mpa': .7, 'frequency_mhz': .25,
                     'cycles': 2., 'delay_us': 1.5, 'phase_deg': 30., 'signal': {'type': 'tone_burst', 'window': 'hann'}, 'profile': {'type': 'gaussian', 'sigma_mm': 1.2}}]
    s['boundaries'] = [{'type': 'symmetry', 'face': 'ymax'}, {'type': 'absorbing', 'face': 'xmax', 'length_mm': 4., 'ramp': 'cubic', 'attenuation_db': 60.}]
    s['acquisition']['duration_us'] = 12.
    return s


class ModelTests(unittest.TestCase):
    def test_weights_reproduce_the_training_material_and_other_materials_are_refused(self):
        for degree in (3, 4, 5):
            model = load(degree); self.assertTrue(model['material_check']['pass'], model['material_check'])
        model = load(3); other = dict(model['material'], l_pa=model['material']['l_pa'] * 1.01)
        self.assertFalse(material_check(model['weights'], other)['pass'])
        s = plate(); s['material']['l_pa'] *= 1.01
        with self.assertRaises(ValueError) as caught: network_scene(s, 3, model['material'])
        self.assertTrue(any(word in str(caught.exception) for word in ('训练材料', '多晶铝')), str(caught.exception))
        for bad, message in [(lambda x: x['numerics'].update(cfl=.33), 'CFL'), (lambda x: x['geometry'].update(kind='cad'), '')]:
            s = plate(); bad(s)
            with self.assertRaises(ValueError): network_scene(s, 3, model['material'])
        with self.assertRaises(ValueError): network_scene(plate(), 2, model['material'])


class InputLayerTests(unittest.TestCase):
    def test_version1_is_frozen_and_the_same_scene_written_as_v2_is_bit_identical(self):
        from graphwave.predict import predict as frozen
        model = load(3); s1 = network_scene(plate(), 3, model['material']); mesh = make_mesh(s1, 3, 1.6)
        a = predict(s1, mesh, model); reference, _ = frozen(s1, mesh, model['weights'], cfl=.3)
        self.assertEqual(a['route'], 'frozen_v1')
        b = predict(network_scene(plate(v2=True), 3, model['material']), mesh, model); self.assertEqual(b['route'], 'v2_input_layer')
        for i, law in enumerate(('linear', 'stvk', 'murnaghan')):
            self.assertTrue(np.array_equal(a['fields'][law], reference[i])); self.assertTrue(np.array_equal(b['fields'][law], reference[i]), law)
        self.assertTrue(all(d['energy_balance_within_1_percent'] for d in b['diagnostics']))

    def test_multi_source_symmetry_and_absorbing_layer_match_the_spectral_solver(self):
        from spectral3d.solver import solve
        from spectral3d.loads import prepare_excitation
        model = load(3); weights = model['weights']; scene = network_scene(excited(plate()), 3, model['material']); mesh = make_mesh(scene, 3, 1.6)
        inputs = build(scene, mesh); self.assertEqual(inputs['loads'].shape[0], 2); self.assertIsNotNone(inputs['alpha'])
        self.assertTrue(np.all(inputs['constrained'][:, 1] == (np.abs(mesh['points'][:, 1] - mesh['points'][:, 1].max()) < 1e-12)))
        dt, sub, frames, steps = layers(scene, mesh, .3); signals = signal_table(inputs, np.arange(steps + 1) * dt * 1e6); inverse = inverse_mapping(mesh)
        def net(state_weights, law):
            return forward(mesh['elements'], weights['derivative_3'], inverse, mesh['volume_weight'], mesh['mass'], inputs['loads'], signals, inputs['free'], inputs['alpha'], True,
                           weights['quadratic_tensor'], weights['cubic_tensor'], state_weights, law, dt, sub, frames)
        m = scene['material']; speed = math.sqrt((m['lambda_pa'] + 2 * m['mu_pa']) / m['rho_kg_m3'])
        for name, law in (('linear', 0), ('murnaghan', 2)):
            s = copy.deepcopy(scene); s['constitutive'] = name
            reference = solve(s, mesh, cfl=.3 * speed * 1.05 / 10000, keep_fields=True, excitation=prepare_excitation(mesh, s))
            self.assertEqual(reference['diagnostics']['internal_steps'], steps); peak = np.max(np.abs(reference['fields_m']))
            exact, energy, _, _ = net(np.array([1., 1.]), law); learned, learned_energy, _, _ = net(weights['state_weights'], law)
            self.assertLess(np.max(np.abs(exact - reference['fields_m'])) / peak, 1e-11)
            # The learned state weights are 1 − 1.4e-10 and 1 − 4.1e-11: a tiny per-layer damping.
            self.assertLess(np.max(np.abs(learned - reference['fields_m'])) / peak, 2e-10 * steps)
            self.assertAlmostEqual(energy[-1, 4], reference['diagnostics']['dissipated_energy_j'], delta=1e-9 * abs(energy[-1, 3]))
            stored = learned_energy[:, 1] + learned_energy[:, 2] + learned_energy[:, 4]
            self.assertLess(np.max(np.abs(stored - learned_energy[:, 3])) / np.max(np.abs(learned_energy[:, 3])), 1e-3)
            self.assertGreater(learned_energy[-1, 4], .1 * learned_energy[-1, 3])

    def test_long_record_matches_frozen_and_resumes_bit_identically(self):
        from graphnet.rollout import rollout, read_record
        from graphwave.long_rollout import rollout as frozen
        model = load(3, long_record=True); s1 = network_scene(plate(duration=15.), 3, model['material']); mesh = make_mesh(s1, 3, 1.6)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); identity = {'test': 'graphnet'}
            frozen(s1, mesh, model['weights'], model['temporal'], root / 'frozen', identity)
            v2 = network_scene(plate(duration=15., v2=True), 3, model['material']); manifest = rollout(v2, mesh, model, root / 'v2', identity)
            self.assertEqual(manifest['phase'], 'completed'); self.assertGreaterEqual(len(manifest['chunks']), 2); record = read_record(root / 'v2')
            chunks = sorted((root / 'frozen' / 'fields').glob('*.npz'))
            for law in ('linear', 'stvk', 'murnaghan'):
                reference = np.concatenate([np.load(c)[law] for c in chunks]); self.assertTrue(np.array_equal(record[law], reference), law)
            scene = network_scene(dict(excited(plate()), acquisition={'duration_us': 15., 'sample_step_us': .05}), 3, model['material']); mesh3 = make_mesh(scene, 3, 1.6)
            partial = rollout(scene, mesh3, model, root / 'resumed', identity, stop_after_chunks=1); self.assertEqual(partial['phase'], 'running')
            rollout(scene, mesh3, model, root / 'resumed', identity); rollout(scene, mesh3, model, root / 'whole', identity)
            a, b = read_record(root / 'resumed'), read_record(root / 'whole')
            for key in ('linear', 'stvk', 'murnaghan', 'probes', 'energies'): self.assertTrue(np.array_equal(a[key], b[key]), key)
            e = b['energies'][2]; stored = e[:, 1] + e[:, 2] + e[:, 4]; self.assertLess(np.max(np.abs(stored - e[:, 3])) / np.max(np.abs(e[:, 3])), 1e-3)
            with self.assertRaises(ValueError): rollout(dict(scene, name='changed'), mesh3, model, root / 'whole', identity)


class RunTests(unittest.TestCase):
    def test_command_line_run_writes_the_sweep_layout_without_solver_modules(self):
        s = excited(plate()); s['receiver_lines'] = [{'name': 'mid', 'from_mm': [2., 1.5, 1.], 'to_mm': [7., 1.5, 1.], 'count': 3, 'direction': [1., 0., 0.]}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'scene.json').write_text(json.dumps(s))
            done = subprocess.run([sys.executable, '-m', 'graphnet.run', '--scene', str(root / 'scene.json'), '--output', str(root / 'out'), '--degree', '3'], cwd=REPO, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            record = json.loads((root / 'out' / 'result.json').read_text()); self.assertEqual(record['route'], 'v2_input_layer'); self.assertEqual(record['dimension'], '3d')
            self.assertTrue(record['neural_model']['neural_forward_only']); self.assertEqual(sorted(k for k in record['receivers'][0] if k.startswith('u_')), ['u_x_m', 'u_y_m', 'u_z_m'])
            self.assertEqual([c['law'] for c in record['controls']], ['linear', 'stvk', 'murnaghan'])
            controls = np.load(root / 'out' / 'controls.npz'); lines = np.load(root / 'out' / 'lines.npz')
            self.assertEqual(controls['murnaghan_probes_m'].shape[1], 4); self.assertEqual(lines['line_mid_projected_m'].shape[1], 3)
            self.assertTrue(np.allclose(controls['murnaghan_probes_m'][:, 0, 0], record['receivers'][0]['u_x_m']))
            self.assertEqual(set(record['files']), {'lines.npz', 'controls.npz'})


if __name__ == '__main__': unittest.main()
