import copy
import os
import json
import unittest
from pathlib import Path
import numpy as np
from physics.scene import validate_scene
from spectral3d.mesh import make_mesh
from graphwave.inference import graph_message, inverse_mapping
from graphnet.model import load
from coarsegraph.operator import AxisParameterization, Parameterization, apply, gll
from coarsegraph.simulate import Coarse
from coarsegraph.evaluate import compare

REPO = Path(__file__).resolve().parents[1]


def scene(length=12., duration=12.):
    s = json.loads((REPO / 'coarsegraph' / 'specs' / 'labels-a-top-gaussian.json').read_text(encoding='utf-8'))['base_scene']
    s['geometry'].update(length_mm=length); s['source']['center_mm'][0] = length / 2; s['acquisition']['duration_us'] = duration
    s['source']['cycles'] = 2.; s['boundaries'] = [{'type': 'symmetry', 'face': 'ymax'}]
    s['receivers'][0]['point_mm'][0] = length - 1; s['receiver_lines'] = []
    s['numerics']['h_mm'] = .4; v = validate_scene(s); v['numerics']['h_mm'] = 3.
    return v


class OperatorTests(unittest.TestCase):
    def test_constraints_hold_for_any_parameters_and_zero_is_gll(self):
        for degree in (3, 4, 5):
            P = Parameterization(degree); D, q, m = P.unpack(np.zeros(P.size)); x, w, D0 = gll(degree)
            self.assertTrue(np.allclose(D, D0, atol=1e-12) and np.array_equal(q, w) and np.array_equal(m, w))
            for seed in range(5):
                c = P.check(np.random.default_rng(seed).normal(size=P.size) * .05)
                for key, value in c.items():
                    if key != 'positive': self.assertLess(value, 1e-12, (degree, key))

    def test_learned_operator_keeps_symmetry_rigid_motions_patch_test_and_energy(self):
        weights = load(3)['weights']; s = scene(); mesh = make_mesh(s, 3, 3.); P = Parameterization(3)
        D0, vol0, mass0 = apply(mesh, s['material']['rho_kg_m3'], *P.unpack(np.zeros(P.size)))
        self.assertTrue(np.array_equal(vol0, mesh['volume_weight']) and np.array_equal(mass0, mesh['mass']))
        theta = np.random.default_rng(3).normal(size=P.size) * .04; D, volume, mass = apply(mesh, s['material']['rho_kg_m3'], *P.unpack(theta))
        inverse = inverse_mapping(mesh); K = lambda u: graph_message(u, mesh['elements'], D, inverse, volume, weights['quadratic_tensor'], weights['cubic_tensor'], 0)[0]
        rng = np.random.default_rng(0); u, v = rng.normal(size=(2, len(mass), 3)) * 1e-9
        self.assertAlmostEqual(float(np.sum(v * K(u))), float(np.sum(u * K(v))), delta=1e-12 * abs(float(np.sum(v * K(u)))))
        points = mesh['points']; scale = np.max(np.abs(K(u)))
        self.assertLess(np.max(np.abs(K(np.ones_like(points) * 1e-9))), 1e-10 * scale)
        W = np.array([[0., -1., 2.], [1., 0., -.5], [-2., .5, 0.]]) * 1e-9
        self.assertLess(np.max(np.abs(K(points @ W.T))), 1e-6 * scale)
        E = np.array([[1., .3, 0.], [.3, -.4, .2], [0., .2, .5]]) * 1e-9 / 1e-3
        interior = np.all([(points[:, a] > points[:, a].min() + 1e-9) & (points[:, a] < points[:, a].max() - 1e-9) for a in range(3)], axis=0)
        force = K(points @ E.T); self.assertTrue(interior.any())
        self.assertLess(np.max(np.abs(force[interior])), 1e-9 * np.max(np.abs(force)))
        c = Coarse(s, 3, 3.); self.assertLess(c.stability(D, volume, mass, weights), 1.5)
        projected, energy = c.run(D, volume, mass, 'murnaghan', weights)
        stored = energy[:, 1] + energy[:, 2] + energy[:, 4]; self.assertLess(np.max(np.abs(stored - energy[:, 3])) / np.max(np.abs(energy[:, 3])), 1e-2)
        self.assertTrue(np.all(np.isfinite(projected)))

    def test_zero_parameters_reproduce_the_frozen_network_on_the_same_grid(self):
        # the GLL baseline of any coarse grid is the frozen network's own computation at that h
        from graphnet.predict import predict
        network = load(3); s = scene(); P = Parameterization(3); c = Coarse(s, 3, 3.); D, volume, mass = c.operators(*P.unpack(np.zeros(P.size)))
        fields = predict(copy.deepcopy(c.scene), c.mesh, network, laws=(0, 2))['fields']
        for law in ('linear', 'murnaghan'):
            probes = np.stack([np.einsum('n,tnc->tc', w, fields[law][:, ids]) for ids, w in c.taps], 1); frozen = np.einsum('trc,rc->tr', probes, c.directions)
            coarse, _ = c.run(D, volume, mass, law, network['weights'])
            self.assertLess(np.max(np.abs(frozen - coarse)), 1e-6 * np.max(np.abs(coarse)), law)

    def test_per_axis_kernel_matches_the_frozen_kernel_and_keeps_constraints(self):
        from coarsegraph.kernel import graph_message as axis_message, forward as axis_forward
        from coarsegraph.operator import AxisParameterization
        from graphnet.recurrence import forward
        weights = load(3)['weights']; s = scene(); mesh = make_mesh(s, 3, 3.); inverse = inverse_mapping(mesh); D = np.ascontiguousarray(mesh['derivative'])
        u = np.random.default_rng(1).normal(size=(len(mesh['mass']), 3)) * 1e-9
        for law in (0, 1, 2):
            a = graph_message(u, mesh['elements'], D, inverse, mesh['volume_weight'], weights['quadratic_tensor'], weights['cubic_tensor'], law)
            b = axis_message(u, mesh['elements'], D, D, D, inverse, mesh['volume_weight'], weights['quadratic_tensor'], weights['cubic_tensor'], law)
            self.assertTrue(np.array_equal(a[0], b[0]) and a[1] == b[1])
        c = Coarse(s, 3, 3.); args = (inverse, mesh['volume_weight'], mesh['mass'], c.inputs['loads'], c.signals, c.inputs['free'], c.alpha, c.damped,
                                      weights['quadratic_tensor'], weights['cubic_tensor'], weights['state_weights'], 2, c.dt, c.subdivisions, c.frames)
        self.assertTrue(np.array_equal(forward(mesh['elements'], D, *args)[0], axis_forward(mesh['elements'], D, D, D, *args)[0]))
        P = AxisParameterization(3); theta = np.random.default_rng(2).normal(size=P.size) * .04; check = P.check(theta)
        self.assertTrue(check['positive']); self.assertTrue(all(v < 1e-12 for k, v in check.items() if k != 'positive'))
        Ds, qs, ms = P.unpack(theta); self.assertEqual(Ds.shape, (3, 4, 4)); self.assertFalse(np.allclose(Ds[0], Ds[2]))
        shared = Parameterization(3); t = np.random.default_rng(4).normal(size=shared.size) * .04; Dt = P.unpack(P.from_shared(t))[0]
        self.assertTrue(all(np.allclose(Dt[a], shared.unpack(t)[0]) for a in range(3)))
        Dv, volume, mass = apply(mesh, s['material']['rho_kg_m3'], Ds, qs, ms); projected, energy = c.run(Dv, volume, mass, 'murnaghan', weights)
        stored = energy[:, 1] + energy[:, 2] + energy[:, 4]; self.assertLess(np.max(np.abs(stored - energy[:, 3])) / np.max(np.abs(energy[:, 3])), 1e-2)

    def test_relative_floor_metric(self):
        t = np.arange(400) * .05; f = .5; base = np.sin(2 * np.pi * f * t) * np.hanning(400)
        fine = np.stack([base * 1e-9 + 1e-11 * np.sin(4 * np.pi * f * t), base * 1e-13], 1)
        coarse = fine.copy(); coarse[:, 1] *= 3
        out = compare(t, coarse, fine, f); self.assertTrue(out['pass']); self.assertEqual(out['waveform']['receivers'], 1)
        coarse = fine * 1.1; out = compare(t, coarse, fine, f); self.assertFalse(out['pass']); self.assertAlmostEqual(out['a2']['worst'], .1, places=6)


class TrainingAndRunTests(unittest.TestCase):
    def test_harmonic_targets_match_the_gate_metric(self):
        from coarsegraph.train import harmonic_targets
        from dispersion.analysis import harmonic_coefficients
        t = np.arange(601) * .05; f = .3; x = np.stack([1e-9 * np.sin(2 * np.pi * f * t) + 3e-12 * np.sin(4 * np.pi * f * t + .4), 1e-13 * np.sin(4 * np.pi * f * t)], 1)
        h = harmonic_targets({'times': t, 'frequency_mhz': f, 'fine_N': x, 'fine_I': x})
        for i in range(2): self.assertAlmostEqual(abs(h['2f_record']['fine'][1, i]), harmonic_coefficients(t, x[:, i], f, t[0], t[-1])['harmonics'][1]['amplitude'], delta=1e-25)
        scale = h['2f_record']['scale'][0]; self.assertAlmostEqual(scale[1], .1 * scale[0])

    def test_run_entry_reproduces_the_simulator_and_refuses_tampered_models(self):
        import subprocess, sys, tempfile
        P = Parameterization(3); D, q, m = P.unpack(np.zeros(P.size))
        model = {'schema': 'workai.coarsegraph.model.v1', 'degree': 3, 'h_mm': 3., 'cfl': .3, 'axes': 'shared', 'theta': [0.] * P.size,
                 'operators': {'derivative': D.tolist(), 'quadrature_weights': q.tolist(), 'mass_weights': m.tolist()}, 'summary': {}}
        raw = json.loads((REPO / 'coarsegraph' / 'specs' / 'labels-a-top-gaussian.json').read_text(encoding='utf-8'))['base_scene']
        raw['geometry'].update(length_mm=12.); raw['source']['center_mm'][0] = 6.; raw['source']['cycles'] = 2.; raw['acquisition']['duration_us'] = 12.
        raw['boundaries'] = [{'type': 'symmetry', 'face': 'ymax'}]; raw['receivers'][0]['point_mm'][0] = 11.; raw['receiver_lines'] = []; raw['numerics']['h_mm'] = .4
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'model.json').write_text(json.dumps(model)); (root / 'scene.json').write_text(json.dumps(raw))
            done = subprocess.run([sys.executable, '-m', 'coarsegraph.run', '--model', str(root / 'model.json'), '--scene', str(root / 'scene.json'), '--output', str(root / 'out'), '--laws', 'scene'],
                                  cwd=REPO, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr[-2000:]); record = json.loads((root / 'out' / 'result.json').read_text())
            self.assertEqual(record['engine'], 'workai_coarse_graph_learned_operator'); self.assertTrue(record['diagnostics']['energy_balance_within_1_percent'])
            c = Coarse(validate_scene(dict(raw, numerics=dict(raw['numerics'], h_mm=.4 / 3))), 3, 3.); weights = load(3)['weights']
            Dv, volume, mass = c.operators(D, q, m); projected, _ = c.run(Dv, volume, mass, 'murnaghan', weights)
            self.assertTrue(np.array_equal(np.asarray(record['receivers'][0]['projected_m']), projected[:, 0]))
            model['operators']['derivative'][1][2] += .01; (root / 'model.json').write_text(json.dumps(model))
            bad = subprocess.run([sys.executable, '-m', 'coarsegraph.run', '--model', str(root / 'model.json'), '--scene', str(root / 'scene.json'), '--output', str(root / 'out2')], cwd=REPO, capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0); self.assertIn('约束', bad.stderr)

    def test_parameters_are_local_coordinates_and_the_stored_operators_define_the_model(self):
        # A null space has no preferred basis; another BLAS kernel returns another one (simulated by a rotation).
        for P in (Parameterization(4), AxisParameterization(3)):
            theta = np.random.default_rng(5).normal(size=P.size) * .04; ops = P.unpack(theta)
            np.testing.assert_allclose(P.theta_from_operators(*ops), theta, atol=1e-10)
            other = copy.deepcopy(P); base = other.base if hasattr(other, 'base') else other
            rotation, _ = np.linalg.qr(np.random.default_rng(6).normal(size=(base.ND.shape[1],) * 2)); base.ND = base.ND @ rotation
            moved = other.theta_from_operators(*ops); self.assertGreater(np.abs(moved - theta).max(), 1e-3)  # different coordinates ...
            for a, b in zip(other.unpack(moved), ops): np.testing.assert_allclose(a, b, atol=1e-10)  # ... same operators
            with self.assertRaises(ValueError): P.theta_from_operators(ops[0] + .01, *ops[1:])

    def test_shipped_models_load_whatever_the_blas_kernel(self):
        import subprocess, sys
        names = sorted(p.parent.name for p in (REPO / 'coarsegraph' / 'models').glob('*/model.json'))
        code = ('import sys, numpy as np\nfrom coarsegraph.run import load_model, local_theta\nfrom coarsegraph.operator import parameterization\n'
                'for name in sys.argv[1:]:\n    model, ops = load_model(f"coarsegraph/models/{name}/model.json")\n'
                '    P = parameterization(model["degree"], model.get("axes", "shared"))\n'
                '    assert all(np.allclose(a, b, rtol=0, atol=1e-10) for a, b in zip(P.unpack(local_theta(model)), ops)), name\nprint("ok")')
        for kernel in ('', 'Prescott', 'Haswell'):  # OpenBLAS kernels that pick different SVD null-space bases (ignored by other BLAS builds)
            env = dict(os.environ, **({'OPENBLAS_CORETYPE': kernel} if kernel else {}))
            done = subprocess.run([sys.executable, '-c', code, *names], cwd=REPO, env=env, capture_output=True, text=True)
            self.assertEqual(done.stdout.strip(), 'ok', (kernel, done.stderr[-1500:]))


if __name__ == '__main__': unittest.main()
