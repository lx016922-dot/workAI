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


def plate(L=12.0, W=2.0, T=1.6, duration=8.0):
    s = default_scene('plate')
    s['schema'] = 'workai.hyperelastic.solid.v2'
    s['geometry'].update(length_mm=L, width_mm=W, thickness_mm=T)
    s['source'].update(center_mm=[0.0, W / 2, T / 2], frequency_mhz=0.3, cycles=2.0, delay_us=0.2)
    s['acquisition'] = {'duration_us': duration, 'sample_step_us': 0.05}
    s['numerics']['h_mm'] = 1.0
    s['receivers'] = [{'name': 'P', 'point_mm': [L - 2.0, W / 2, T / 2], 'direction': [1.0, 0.0, 0.0]}]
    return s


def quarter(scene):
    """The same problem with every length and time divided by four (a power of two, so the scaling is exact in binary)."""
    s = copy.deepcopy(scene)
    g = s['geometry']
    for key in ('length_mm', 'width_mm', 'thickness_mm'):
        g[key] /= 4
    s['source'].update(
        center_mm=[v / 4 for v in s['source']['center_mm']],
        frequency_mhz=s['source']['frequency_mhz'] * 4,
        delay_us=s['source']['delay_us'] / 4,
    )
    s['acquisition'] = {k: v / 4 for k, v in s['acquisition'].items()}
    s['numerics']['h_mm'] /= 4
    for r in s['receivers']:
        r['point_mm'] = [v / 4 for v in r['point_mm']]
    s['defects'] = [{k: v / 4 if type(v) is float else v for k, v in d.items()} for d in s.get('defects', [])]
    return s


def bar(sign, contact):
    """1 × 1 mm bar, unipolar 3 μs Hann traction on the x = 0 end (sign +1 compressive), optional closed interface at x = 12 mm."""
    s = plate(L=24.0, W=1.0, T=1.0, duration=8.0)
    step = 0.05
    t = np.arange(0, 3.0 + 1e-9, step)
    s['source'] = {
        'face': 'xmin',
        'center_mm': [0.0, 0.5, 0.5],
        'radius_mm': 0.0,
        'direction': [float(sign), 0.0, 0.0],
        'amplitude_mpa': 1.0,
        'frequency_mhz': 0.3,
        'cycles': 1.0,
        'delay_us': 0.2,
        'phase_deg': 0.0,
        'signal': {
            'type': 'samples',
            'sample_step_us': step,
            'values': (0.5 * (1 - np.cos(2 * np.pi * t / 3.0))).round(9).tolist(),
            'normalize': False,
        },
    }
    s['receivers'] = [
        {'name': 'before', 'point_mm': [6.0, 0.5, 0.5], 'direction': [1.0, 0.0, 0.0]},
        {'name': 'after', 'point_mm': [18.0, 0.5, 0.5], 'direction': [1.0, 0.0, 0.0]},
    ]
    if contact:
        s['defects'] = [{'type': 'contact', 'orientation': 'vertical', 'x_mm': 12.0}]
    return s


class DefectMesh(unittest.TestCase):
    def test_volume_patch_test_splits_and_contact_area(self):
        from spectral3d.solver import internal_force as cartesian
        from spectral3d.curved_solver import internal_force as curved

        constants = np.array([58.1e9, 26.1e9, 0.0, 0.0, 0.0])
        A = np.array([[1, 0.3, 0.2], [0.1, -0.5, 0.4], [0.2, 0.1, 0.7]]) * 1e-5
        eps = 0.5 * (A + A.T)
        density = 0.5 * 58.1e9 * np.trace(eps) ** 2 + 26.1e9 * np.sum(eps * eps)
        # Split counts include the front refinement: the crack face 1.1–1.6 mm gets a grid line at 1.35 (tip + h/4), so two
        # elements of 4 GLL intervals → 8 nodes split per y line (the tip stays shared) × 9 y lines.
        cases = [
            ([{'type': 'crack', 'x_mm': 6.0, 'depth_mm': 0.5}], 12 * 2 * 1.6, 72, 0),
            ([{'type': 'crack', 'x_mm': 6.0, 'depth_mm': 0.6, 'from': 'inside', 'z_mm': 0.8, 'closed': True}], 12 * 2 * 1.6, 63, 63),
            ([{'type': 'delamination', 'x_start_mm': 4.0, 'x_end_mm': 8.0, 'depth_mm': 0.5}], 12 * 2 * 1.6, 207, 0),
            (
                [{'type': 'crack', 'x_mm': 6.0, 'depth_mm': 0.5}, {'type': 'hole', 'x_mm': 7.0, 'y_mm': 1.0, 'radius_mm': 0.3}],
                12 * 2 * 1.6 - np.pi * 0.09 * 1.6,
                72,
                0,
            ),
            ([{'type': 'contact', 'orientation': 'vertical', 'x_mm': 6.0}], 12 * 2 * 1.6, 81, 81),
            ([{'type': 'notch', 'x_mm': 6.0, 'width_mm': 0.8, 'depth_mm': 0.5}], 12 * 2 * 1.6 - 0.8 * 0.5 * 2, 0, 0),
            ([{'type': 'hole', 'x_mm': 6.0, 'y_mm': 1.0, 'radius_mm': 0.5}], 12 * 2 * 1.6 - np.pi * 0.25 * 1.6, 0, 0),
            ([{'type': 'hole', 'axis': 'y', 'x_mm': 6.0, 'z_mm': 0.8, 'radius_mm': 0.4}], 12 * 2 * 1.6 - np.pi * 0.16 * 2, 0, 0),
        ]
        for defects, volume, split, pairs in cases:
            s = plate()
            s['defects'] = defects
            m = make_mesh(s, 4, 1.0)
            self.assertAlmostEqual(m['mass'].sum() / 2700 * 1e9, volume, places=3, msg=defects)
            self.assertEqual(int(sum(m['split_nodes'])), split)
            self.assertEqual(len(m['contact_a']), pairs)
            u = m['points'] @ A.T
            f, energy, *_ = (
                curved(u, m['elements'], m['derivative'], m['inverse_jacobian'], m['volume_weight'], constants, 0)
                if 'inverse_jacobian' in m
                else cartesian(u, m['elements'], m['derivative'], m['size'], m['volume_weight'], constants, 0)
            )
            self.assertAlmostEqual(
                energy / (density * volume * 1e-9), 1.0, places=4, msg=defects
            )  # homogeneous strain is represented exactly
        s = plate()
        s['defects'] = [{'type': 'contact', 'orientation': 'vertical', 'x_mm': 6.0}]
        m = make_mesh(s, 4, 1.0)
        self.assertAlmostEqual(m['contact_area'].sum(), 2e-3 * 1.6e-3, places=15)  # the whole cross-section, GLL-weighted
        self.assertTrue(np.allclose(m['points'][m['contact_a']], m['points'][m['contact_b']]))
        self.assertTrue(np.all(m['contact_normal'] == [1.0, 0.0, 0.0]))
        self.assertIsNone(make_mesh(plate(), 4, 1.0).get('contact_a'))  # no defects: the frozen mesh unchanged
        close = plate()
        close['defects'] = [
            {'type': 'crack', 'x_mm': 6.0, 'depth_mm': 0.5},
            {'type': 'delamination', 'x_start_mm': 6.05, 'x_end_mm': 10.0, 'depth_mm': 1.2},
        ]
        with self.assertRaises(ValueError) as caught:
            make_mesh(close, 4, 1.0)
        self.assertIn('极薄', str(caught.exception))

    def test_fronts_inside_the_material_get_refinement_lines(self):
        from spectral3d.feature_mesh import _front_cuts

        size = (12.0, 2.0, 1.6)
        crack = {'axis': 0, 'position_mm': 6.0, 'ranges_mm': [[6.0, 6.0], [0.0, 2.0], [1.1, 1.6]]}  # from the top: tip at z = 1.1
        cuts = [{0.0: 'f', 6.0: 'c', 12.0: 'f'}, {0.0: 'f', 2.0: 'f'}, {0.0: 'f', 1.1: 'c', 1.6: 'f'}]
        _front_cuts([crack], cuts, size, 0.25)
        self.assertEqual(sorted(cuts[0]), [0.0, 5.75, 6.0, 6.25, 12.0])  # across the crack plane
        self.assertEqual(sorted(cuts[2]), [0.0, 0.85, 1.1, 1.35, 1.6])  # across the tip, in the plane
        self.assertEqual(sorted(cuts[1]), [0.0, 2.0])  # full width: the y edges open, no front
        delamination = {'axis': 2, 'position_mm': 1.5, 'ranges_mm': [[4.0, 8.0], [0.0, 2.0], [1.5, 1.5]]}  # 0.1 mm under the top
        cuts = [{0.0: 'f', 4.0: 'd', 8.0: 'd', 12.0: 'f'}, {0.0: 'f', 2.0: 'f'}, {0.0: 'f', 1.5: 'd', 1.6: 'f'}]
        _front_cuts([delamination], cuts, size, 0.25)
        self.assertEqual(sorted(cuts[0]), [0.0, 3.75, 4.0, 4.25, 7.75, 8.0, 8.25, 12.0])
        self.assertEqual(sorted(cuts[2]), [0.0, 1.25, 1.5, 1.6])  # 1.75 lies outside the plate; no sliver lines

    def test_a_defect_mesh_and_its_record_are_the_same_at_a_quarter_of_the_size(self):
        # Elasticity without dissipation has no length of its own: with the same traction, the displacements of the quarter-size
        # problem are a quarter. The software must not bring a length in (the gap rule between grid lines once did: 0.08 mm).
        from spectral3d.feature_mesh import cut_gap
        from graphnet.rollout import read_record, rollout

        self.assertEqual([cut_gap(h) for h in (2.5, 0.8, 0.25)], [0.08, 0.08, 0.025])
        raw = plate(L=8.0, duration=4.0)
        raw['source'].update(frequency_mhz=0.6)
        raw['defects'] = [{'type': 'crack', 'x_mm': 4.0, 'depth_mm': 0.8, 'closed': True}]
        model = load(4, long_record=True)
        runs = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, case in enumerate((raw, quarter(raw))):
                scene = network_scene(case, 4, model['material'])
                mesh = make_mesh(scene, 4, scene['numerics']['h_mm'])
                rollout(scene, mesh, model, Path(tmp) / str(i), {'t': 1}, time_plan='auto', controls=('linear', 'murnaghan'))
                runs.append((mesh, read_record(Path(tmp) / str(i))))
        (a, ra), (b, rb) = runs
        self.assertEqual(
            len(np.unique(a['points'][:, 0])),
            len(np.unique(make_mesh(network_scene(plate(L=8.0), 4, model['material']), 4, 1.0)['points'][:, 0])) + 8,
        )  # two front lines beside the crack plane, at both sizes
        self.assertTrue(np.array_equal(a['elements'], b['elements']))
        self.assertTrue(np.array_equal(a['contact_a'], b['contact_a']))
        self.assertTrue(np.allclose(a['points'], 4 * b['points'], rtol=1e-12, atol=0))
        self.assertTrue(np.allclose(a['mass'], 64 * b['mass'], rtol=1e-12, atol=0))
        for key in ('linear', 'murnaghan'):
            self.assertGreater(np.abs(ra[key]).max(), 1e-9)
            self.assertLess(np.abs(ra[key] - 4 * rb[key]).max(), 1e-9 * np.abs(ra[key]).max(), key)
        increment = ra['murnaghan'] - ra['linear']
        self.assertLess(np.abs(increment - 4 * (rb['murnaghan'] - rb['linear'])).max(), 1e-6 * np.abs(increment).max())

    def test_network_equals_the_gll_solver_on_cracked_notched_and_holed_plates(self):
        from spectral3d.solver import solve
        from spectral3d.loads import prepare_excitation

        model = load(3)
        weights = model['weights']
        m = model['material']
        speed = np.sqrt((m['lambda_pa'] + 2 * m['mu_pa']) / m['rho_kg_m3'])
        for defects in (
            [
                {'type': 'crack', 'x_mm': 5.0, 'depth_mm': 0.5},
                {'type': 'notch', 'x_mm': 8.0, 'width_mm': 0.6, 'depth_mm': 0.4, 'from': 'bottom'},
            ],
            [{'type': 'hole', 'x_mm': 6.0, 'y_mm': 1.0, 'radius_mm': 0.5}],
        ):
            raw = plate(duration=8.0)
            raw['defects'] = defects
            scene = network_scene(raw, 3, model['material'])
            mesh = make_mesh(scene, 3, 1.0)
            inputs = build(scene, mesh)
            dt, sub, frames, steps = layers(scene, mesh, 0.3)
            signals = signal_table(inputs, np.arange(steps + 1) * dt * 1e6)
            alpha = np.zeros(len(mesh['mass']))
            for name, law in (('linear', 0), ('murnaghan', 2)):
                s = copy.deepcopy(scene)
                s['constitutive'] = name
                reference = solve(s, mesh, cfl=0.3 * speed * 1.05 / 10000, keep_fields=True, excitation=prepare_excitation(mesh, s))
                self.assertEqual(reference['diagnostics']['internal_steps'], steps)
                peak = np.max(np.abs(reference['fields_m']))
                net, energy, _, _ = forward(
                    mesh['elements'],
                    weights['derivative_3'],
                    inverse_mapping(mesh),
                    mesh['volume_weight'],
                    mesh['mass'],
                    inputs['loads'],
                    signals,
                    inputs['free'],
                    alpha,
                    False,
                    weights['quadratic_tensor'],
                    weights['cubic_tensor'],
                    np.array([1.0, 1.0]),
                    law,
                    dt,
                    sub,
                    frames,
                )
                self.assertLess(np.max(np.abs(net - reference['fields_m'])) / peak, 1e-10, (defects[0]['type'], name))


class Contact(unittest.TestCase):
    def test_closed_interface_passes_compression_and_opens_in_tension(self):
        from graphnet.run import simulate

        out = {}
        for sign in (1, -1):
            for contact in (False, True):
                run = simulate(bar(sign, contact), degree=3, laws='scene')
                d = run['computed']['diagnostics']
                self.assertTrue(d['energy_balance_within_1_percent'], d['external_work_balance_relative'])
                out[sign, contact] = run['probes']['murnaghan'][:, :, 0]
                if contact:
                    self.assertEqual(d['contact']['pairs'], 16)
                    self.assertLessEqual(d['contact']['omega_dt_max'], 0.6 + 1e-12)
                    self.assertLess(d['contact']['compression_reflection']['source_frequency'], 0.05)
        ref, closed = out[1, False], out[1, True]
        self.assertAlmostEqual(np.max(np.abs(closed[:, 1])) / np.max(np.abs(ref[:, 1])), 1.0, delta=0.03)  # compression passes
        self.assertLess(np.max(np.abs(closed[:, 0] - ref[:, 0])) / np.max(np.abs(ref[:, 0])), 0.05)  # with a small penalty echo
        ref, opened = out[-1, False], out[-1, True]
        self.assertLess(np.max(np.abs(opened[:, 1])) / np.max(np.abs(ref[:, 1])), 1e-3)  # tension opens: nothing passes
        self.assertGreater(np.max(np.abs(opened[:, 0] - ref[:, 0])) / np.max(np.abs(ref[:, 0])), 0.5)  # and is reflected

    def test_a_job_with_a_closed_crack_runs_with_the_automatic_mesh(self):
        from offline.domain import validate_request

        raw = plate(L=16.0, duration=10.0)
        raw['boundaries'] = [{'type': 'symmetry', 'face': 'ymax'}]
        raw['defects'] = [{'type': 'crack', 'x_mm': 8.0, 'depth_mm': 0.5, 'closed': True}]
        request = validate_request({'scene': raw, 'mesh': 'auto'})
        self.assertEqual(request['mesh'], 'auto')
        report = request['mesh_report']
        self.assertEqual(report['contact_pairs'], report['split_nodes'])
        self.assertGreaterEqual(report['nodes_per_2f_shear_wavelength'], 10 - 1e-9)
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / 'local-defect'
            job.mkdir()
            (job / 'request.json').write_text(json.dumps(request), encoding='utf-8')
            (job / 'status.json').write_text(json.dumps({'job_id': job.name, 'phase': 'starting'}), encoding='utf-8')
            done = subprocess.run(
                [sys.executable, '-X', 'utf8', '-m', 'offline.worker', '--job-dir', str(job)], cwd=REPO, capture_output=True, text=True
            )
            status = json.loads((job / 'status.json').read_text(encoding='utf-8'))
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            self.assertEqual(status['phase'], 'completed', status)
            manifest = json.loads((job / 'record' / 'recording.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['contact']['pairs'], report['contact_pairs'])
            full = json.loads((job / 'full.json').read_text(encoding='utf-8'))
            self.assertEqual(full['scene']['defects'][0]['type'], 'crack')
            self.assertTrue(all(c['energy_balance_within_1_percent'] for c in full['controls']))


if __name__ == '__main__':
    unittest.main()
