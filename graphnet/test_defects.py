"""Defects through the frozen network: mesh consistency, network ≡ GLL solver on defect meshes, contact physics, a worker job."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from solid.schema import default_scene
from spectral3d.feature_mesh import make_mesh
from graphnet.model import load
from graphnet.inputs import network_scene, build, signal_table
from graphnet.predict import layers
from graphnet.recurrence import forward
from graphwave.inference import inverse_mapping

REPO = Path(__file__).resolve().parents[1]


def plate(L=12., W=2., T=1.6, duration=8.):
    s = default_scene('plate'); s['schema'] = 'workai.hyperelastic.solid.v2'; s['geometry'].update(length_mm=L, width_mm=W, thickness_mm=T)
    s['source'].update(center_mm=[0., W / 2, T / 2], frequency_mhz=.3, cycles=2., delay_us=.2)
    s['acquisition'] = {'duration_us': duration, 'sample_step_us': .05}; s['numerics']['h_mm'] = 1.
    s['receivers'] = [{'name': 'P', 'point_mm': [L - 2., W / 2, T / 2], 'direction': [1., 0., 0.]}]
    return s


def bar(sign, contact):
    """1 × 1 mm bar, unipolar 3 μs Hann traction on the x = 0 end (sign +1 compressive), optional closed interface at x = 12 mm."""
    s = plate(L=24., W=1., T=1., duration=8.); step = .05; t = np.arange(0, 3. + 1e-9, step)
    s['source'] = {'face': 'xmin', 'center_mm': [0., .5, .5], 'radius_mm': 0., 'direction': [float(sign), 0., 0.], 'amplitude_mpa': 1., 'frequency_mhz': .3, 'cycles': 1.,
                   'delay_us': .2, 'phase_deg': 0., 'signal': {'type': 'samples', 'sample_step_us': step, 'values': (.5 * (1 - np.cos(2 * np.pi * t / 3.))).round(9).tolist(), 'normalize': False}}
    s['receivers'] = [{'name': 'before', 'point_mm': [6., .5, .5], 'direction': [1., 0., 0.]}, {'name': 'after', 'point_mm': [18., .5, .5], 'direction': [1., 0., 0.]}]
    if contact: s['defects'] = [{'type': 'contact', 'orientation': 'vertical', 'x_mm': 12.}]
    return s


class DefectMesh(unittest.TestCase):
    def test_volume_patch_test_splits_and_contact_area(self):
        from spectral3d.solver import internal_force as cartesian
        from spectral3d.curved_solver import internal_force as curved
        constants = np.array([58.1e9, 26.1e9, 0., 0., 0.]); A = np.array([[1, .3, .2], [.1, -.5, .4], [.2, .1, .7]]) * 1e-5
        eps = .5 * (A + A.T); density = .5 * 58.1e9 * np.trace(eps) ** 2 + 26.1e9 * np.sum(eps * eps)
        cases = [([{'type': 'crack', 'x_mm': 6., 'depth_mm': .5}], 12 * 2 * 1.6, 36, 0),
                 ([{'type': 'crack', 'x_mm': 6., 'depth_mm': .6, 'from': 'inside', 'z_mm': .8, 'closed': True}], 12 * 2 * 1.6, 27, 27),
                 ([{'type': 'contact', 'orientation': 'vertical', 'x_mm': 6.}], 12 * 2 * 1.6, 81, 81),
                 ([{'type': 'notch', 'x_mm': 6., 'width_mm': .8, 'depth_mm': .5}], 12 * 2 * 1.6 - .8 * .5 * 2, 0, 0),
                 ([{'type': 'hole', 'x_mm': 6., 'y_mm': 1., 'radius_mm': .5}], 12 * 2 * 1.6 - np.pi * .25 * 1.6, 0, 0),
                 ([{'type': 'hole', 'axis': 'y', 'x_mm': 6., 'z_mm': .8, 'radius_mm': .4}], 12 * 2 * 1.6 - np.pi * .16 * 2, 0, 0)]
        for defects, volume, split, pairs in cases:
            s = plate(); s['defects'] = defects; m = make_mesh(s, 4, 1.)
            self.assertAlmostEqual(m['mass'].sum() / 2700 * 1e9, volume, places=3, msg=defects)
            self.assertEqual(int(sum(m['split_nodes'])), split); self.assertEqual(len(m['contact_a']), pairs)
            u = m['points'] @ A.T
            f, energy, *_ = (curved(u, m['elements'], m['derivative'], m['inverse_jacobian'], m['volume_weight'], constants, 0) if 'inverse_jacobian' in m
                             else cartesian(u, m['elements'], m['derivative'], m['size'], m['volume_weight'], constants, 0))
            self.assertAlmostEqual(energy / (density * volume * 1e-9), 1., places=4, msg=defects)   # homogeneous strain is represented exactly
        s = plate(); s['defects'] = [{'type': 'contact', 'orientation': 'vertical', 'x_mm': 6.}]; m = make_mesh(s, 4, 1.)
        self.assertAlmostEqual(m['contact_area'].sum(), 2e-3 * 1.6e-3, places=15)   # the whole cross-section, GLL-weighted
        self.assertTrue(np.allclose(m['points'][m['contact_a']], m['points'][m['contact_b']])); self.assertTrue(np.all(m['contact_normal'] == [1., 0., 0.]))
        self.assertIsNone(make_mesh(plate(), 4, 1.).get('contact_a'))   # no defects: the frozen mesh unchanged
        close = plate(); close['defects'] = [{'type': 'crack', 'x_mm': 6., 'depth_mm': .5}, {'type': 'delamination', 'x_start_mm': 6.05, 'x_end_mm': 10., 'depth_mm': 1.2}]
        with self.assertRaises(ValueError) as caught: make_mesh(close, 4, 1.)
        self.assertIn('极薄', str(caught.exception))

    def test_network_equals_the_gll_solver_on_cracked_notched_and_holed_plates(self):
        from spectral3d.solver import solve
        from spectral3d.loads import prepare_excitation
        model = load(3); weights = model['weights']; m = model['material']; speed = np.sqrt((m['lambda_pa'] + 2 * m['mu_pa']) / m['rho_kg_m3'])
        for defects in ([{'type': 'crack', 'x_mm': 5., 'depth_mm': .5}, {'type': 'notch', 'x_mm': 8., 'width_mm': .6, 'depth_mm': .4, 'from': 'bottom'}],
                        [{'type': 'hole', 'x_mm': 6., 'y_mm': 1., 'radius_mm': .5}]):
            raw = plate(duration=8.); raw['defects'] = defects
            scene = network_scene(raw, 3, model['material']); mesh = make_mesh(scene, 3, 1.)
            inputs = build(scene, mesh); dt, sub, frames, steps = layers(scene, mesh, .3); signals = signal_table(inputs, np.arange(steps + 1) * dt * 1e6)
            alpha = np.zeros(len(mesh['mass']))
            for name, law in (('linear', 0), ('murnaghan', 2)):
                s = copy.deepcopy(scene); s['constitutive'] = name
                reference = solve(s, mesh, cfl=.3 * speed * 1.05 / 10000, keep_fields=True, excitation=prepare_excitation(mesh, s))
                self.assertEqual(reference['diagnostics']['internal_steps'], steps); peak = np.max(np.abs(reference['fields_m']))
                net, energy, _, _ = forward(mesh['elements'], weights['derivative_3'], inverse_mapping(mesh), mesh['volume_weight'], mesh['mass'], inputs['loads'], signals, inputs['free'], alpha, False,
                                            weights['quadratic_tensor'], weights['cubic_tensor'], np.array([1., 1.]), law, dt, sub, frames)
                self.assertLess(np.max(np.abs(net - reference['fields_m'])) / peak, 1e-10, (defects[0]['type'], name))


class Contact(unittest.TestCase):
    def test_closed_interface_passes_compression_and_opens_in_tension(self):
        from graphnet.run import simulate
        out = {}
        for sign in (1, -1):
            for contact in (False, True):
                run = simulate(bar(sign, contact), degree=3, laws='scene'); d = run['computed']['diagnostics']
                self.assertTrue(d['energy_balance_within_1_percent'], d['external_work_balance_relative'])
                out[sign, contact] = run['probes']['murnaghan'][:, :, 0]
                if contact:
                    self.assertEqual(d['contact']['pairs'], 16); self.assertLessEqual(d['contact']['omega_dt_max'], .6 + 1e-12)
                    self.assertLess(d['contact']['compression_reflection']['source_frequency'], .05)
        ref, closed = out[1, False], out[1, True]
        self.assertAlmostEqual(np.max(np.abs(closed[:, 1])) / np.max(np.abs(ref[:, 1])), 1., delta=.03)          # compression passes
        self.assertLess(np.max(np.abs(closed[:, 0] - ref[:, 0])) / np.max(np.abs(ref[:, 0])), .05)                 # with a small penalty echo
        ref, opened = out[-1, False], out[-1, True]
        self.assertLess(np.max(np.abs(opened[:, 1])) / np.max(np.abs(ref[:, 1])), 1e-3)                          # tension opens: nothing passes
        self.assertGreater(np.max(np.abs(opened[:, 0] - ref[:, 0])) / np.max(np.abs(ref[:, 0])), .5)               # and is reflected

    def test_a_job_with_a_closed_crack_runs_with_the_automatic_mesh(self):
        from offline.domain import validate_request
        raw = plate(L=16., duration=10.); raw['boundaries'] = [{'type': 'symmetry', 'face': 'ymax'}]
        raw['defects'] = [{'type': 'crack', 'x_mm': 8., 'depth_mm': .5, 'closed': True}]
        request = validate_request({'scene': raw, 'mesh': 'auto'})
        self.assertEqual(request['mesh'], 'auto'); report = request['mesh_report']; self.assertEqual(report['contact_pairs'], report['split_nodes'])
        self.assertGreaterEqual(report['nodes_per_2f_shear_wavelength'], 10 - 1e-9)
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / 'local-defect'; job.mkdir()
            (job / 'request.json').write_text(json.dumps(request), encoding='utf-8')
            (job / 'status.json').write_text(json.dumps({'job_id': job.name, 'phase': 'starting'}), encoding='utf-8')
            done = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'offline.worker', '--job-dir', str(job)], cwd=REPO, capture_output=True, text=True)
            status = json.loads((job / 'status.json').read_text(encoding='utf-8'))
            self.assertEqual(done.returncode, 0, done.stderr[-2000:]); self.assertEqual(status['phase'], 'completed', status)
            manifest = json.loads((job / 'record' / 'recording.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['contact']['pairs'], report['contact_pairs'])
            full = json.loads((job / 'full.json').read_text(encoding='utf-8'))
            self.assertEqual(full['scene']['defects'][0]['type'], 'crack'); self.assertTrue(all(c['energy_balance_within_1_percent'] for c in full['controls']))


if __name__ == '__main__': unittest.main()
