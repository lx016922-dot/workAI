import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from solid.schema import default_scene
from physics.runs import write_run_outputs
from sweep.spec import validate_spec, build_scene
from sweep.runner import Campaign
from sweep.convergence import settings, variants, compare_runs, selected_cases
from sweep.summarize import summarize


def plate(kind='plate', h=1.5):
    s = default_scene(kind); s['schema'] = 'workai.hyperelastic.solid.v2'; s['geometry'].update(length_mm=12., width_mm=2., thickness_mm=2.)
    s['source'].update(center_mm=[0., 1., 1.], frequency_mhz=.2, cycles=2.); s['numerics']['h_mm'] = h
    s['receivers'] = [{'name': 'P1', 'point_mm': [4., 1., 1.], 'direction': [1., 0., 0.]}]
    s['receiver_lines'] = [{'name': 'mid', 'from_mm': [3., 1., 1.], 'to_mm': [9., 1., 1.], 'count': 3, 'direction': [1., 0., 0.]}]
    return s


def spec(engine='3d_spectral', **extra):
    base = plate(h=1.5 if engine == '3d_spectral' else .9)
    return dict({'schema': 'workai.sweep.v1', 'name': f'{engine} convergence', 'base_scene': base, 'engine': engine, 'grid': {'source.amplitude_mpa': [1.]},
                 'analysis': {'window': {'mode': 'fixed'}}, 'budget': {'seconds_per_case': 600, 'workers': 1}}, **extra)


def fake_run(directory, probes, times):
    receivers = [{'name': 'P1', 'point_mm': [4., 1., 1.], 'direction': [1., 0., 0.], 'group': None, 'index': None, 'distance_mm': None}] + \
                [{'name': f'mid[{i}]', 'point_mm': [3. + 3 * i, 1., 1.], 'direction': [1., 0., 0.], 'group': 'mid', 'index': i, 'distance_mm': 3. * i} for i in range(3)]
    scene = {'schema': 'workai.hyperelastic.solid.v1'}; computed = {'probes_m': probes, 'fields_m': None, 'times_us': times, 'diagnostics': {'nodes': 10, 'internal_steps': 1, 'internal_step_us': .1, 'external_work_balance_relative': 0., 'elapsed_s': 0.}}
    write_run_outputs({'scene': scene, 'mesh': {'points': np.zeros((10, 3))}, 'receivers': receivers, 'computed': computed}, directory, 'test', 'test', '3d')


class ConvergenceUnitTests(unittest.TestCase):
    def test_variants_follow_each_engine(self):
        conv = settings({}); s = validate_spec(spec()); scene = build_scene(s, {})
        rows = {r['level']: r for r in variants(s, scene, 2, 4, conv)}
        self.assertAlmostEqual(rows['space']['scene']['numerics']['h_mm'], 1.5 / np.sqrt(2)); self.assertEqual(rows['degree']['settings']['degree'], 3)
        self.assertAlmostEqual(rows['time']['scene']['numerics']['cfl'], .15); self.assertEqual(rows['time']['settings'], {'degree': 2, 'substeps': 4})
        hole = copy.deepcopy(scene); hole['geometry']['kind'] = 'hole'; self.assertEqual({r['level']: r for r in variants(s, hole, 2, 4, conv)}['time']['settings']['substeps'], 8)
        self.assertIn('maximum', {r['level']: r for r in variants(s, scene, 6, 4, conv)}['degree']['skipped'])
        slow = copy.deepcopy(scene); slow['numerics']['cfl'] = .05; self.assertIn('minimum', {r['level']: r for r in variants(s, slow, 2, 4, conv)}['time']['skipped'])
        t = validate_spec(spec('3d_tetra')); self.assertIn('linear tetrahedra', {r['level']: r for r in variants(t, build_scene(t, {}), 3, 4, conv)}['degree']['skipped'])

    def test_settings_merge_and_reject(self):
        s = settings({'convergence': {'tolerance': {'harmonic_relative': .1}, 'levels': ['space'], 'cases': 2}}, {'time_factor': .25})
        self.assertEqual(s['tolerance'], {'waveform_relative_l2': .05, 'harmonic_relative': .1}); self.assertEqual(s['levels'], ['space']); self.assertEqual(s['time_factor'], .25)
        for bad, message in [({'levels': ['mesh']}, '收敛层级'), ({'levels': []}, '收敛层级'), ({'space_factor': 1.}, '细化系数'), ({'tolerance': {'x': .1}}, '容差'), ({'window_us': [5, 1]}, 'window_us')]:
            with self.assertRaises(ValueError) as caught: settings({'convergence': bad})
            self.assertIn(message, str(caught.exception))
        with self.assertRaises(ValueError) as caught: validate_spec(spec(convergence={'levels': ['space'], 'tolerance': {'waveform_relative_l2': 2}}))
        self.assertIn('容差', str(caught.exception))
        done = [{'case_id': 'a', 'parameters': {'constitutive': 'linear', 'x': 1}}, {'case_id': 'b', 'parameters': {'constitutive': 'murnaghan', 'x': 1}}, {'case_id': 'c', 'parameters': {'constitutive': 'murnaghan', 'x': 2}}]
        self.assertEqual(selected_cases({}, done), ['a', 'b', 'c']); self.assertEqual(selected_cases({}, done, '1'), ['a']); self.assertEqual(selected_cases({'convergence': {'cases': ['c']}}, done), ['c'])
        self.assertEqual(selected_cases({}, done, 'b'), ['a', 'b']); self.assertEqual(selected_cases({}, done, 'b', include_controls=False), ['b'])
        self.assertEqual(selected_cases({}, done, '{"constitutive": "murnaghan", "x": 1}'), ['a', 'b']); self.assertEqual(selected_cases({}, done, {'x': [2]}), ['c'])
        for bad in ('a,z', {'x': 3}, {}): 
            with self.assertRaises(ValueError): selected_cases({}, done, bad)

    def test_comparison_metrics_and_floor(self):
        times = np.arange(400) * .05; f = .5; wave = np.sin(2 * np.pi * f * times) * np.hanning(400)
        probes = np.zeros((400, 4, 3)); probes[:, :, 0] = 1e-8 * wave[:, None] + 1e-10 * np.sin(4 * np.pi * f * times)[:, None]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fake_run(root / 'a', probes, times); fake_run(root / 'same', probes.copy(), times); fake_run(root / 'off', probes * 1.1, times)
            tiny = probes * 1e-6; fake_run(root / 'tiny', tiny, times); fake_run(root / 'tiny2', tiny * 1.5, times)
            tol = {'waveform_relative_l2': .05, 'harmonic_relative': .05}
            same = compare_runs(root / 'a', root / 'same', f, tol); self.assertTrue(same['pass']); self.assertEqual(same['worst']['waveform_relative_l2'], 0.); self.assertEqual(len(same['receivers']), 4)
            off = compare_runs(root / 'a', root / 'off', f, tol); self.assertFalse(off['pass']); self.assertAlmostEqual(off['worst']['waveform_relative_l2'], .1 / 1.1, places=9)
            self.assertAlmostEqual(off['worst']['a2_relative'], .1 / 1.1, places=6)
            loose = compare_runs(root / 'a', root / 'off', f, {'waveform_relative_l2': .1, 'harmonic_relative': .1}); self.assertTrue(loose['pass'])
            floor = compare_runs(root / 'tiny', root / 'tiny2', f, tol); self.assertTrue(floor['pass']); self.assertFalse(floor['receivers'][0]['waveform']['above_reporting_floor'])


class SpecHashTests(unittest.TestCase):
    def test_hash_is_stable_for_existing_plane_strain_campaigns(self):
        import hashlib
        from sweep.runner import _spec_sha
        two = json.loads((Path(__file__).resolve().parents[1] / 'docs' / 'examples' / 'plate-2d-s0-line.json').read_text(encoding='utf-8'))
        validated = validate_spec({'schema': 'workai.sweep.v1', 'name': 'hash', 'base_scene': two, 'grid': {'source.amplitude_mpa': [1., 2.]}})
        old = {k: v for k, v in validated.items() if k not in ('engine', 'substeps')}
        self.assertEqual(_spec_sha(validated), hashlib.sha256(json.dumps(old, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
        self.assertEqual(_spec_sha(validated), _spec_sha(validate_spec(dict(validated, convergence={'levels': ['space']}))))
        three = validate_spec(spec()); self.assertNotEqual(_spec_sha(three), _spec_sha(validate_spec(spec('3d_tetra', base_scene=three['base_scene']))))


class RunEntryTests(unittest.TestCase):
    def test_spectral_run_entry_matches_the_web_worker_solver_call(self):
        from spectral3d.run import simulate
        from spectral3d.mesh import make_mesh
        from spectral3d.solver import solve
        from physics.scene import validate_scene
        v1 = plate(); v1['schema'] = 'workai.hyperelastic.solid.v1'; v1.pop('receiver_lines'); v1['acquisition'] = {'duration_us': 11., 'sample_step_us': .1}
        run = simulate(copy.deepcopy(v1), 2)
        raw = copy.deepcopy(v1); raw['numerics']['h_mm'] = .75; scene = validate_scene(raw); scene['numerics']['h_mm'] = 1.5
        reference = solve(scene, make_mesh(scene, 2, 1.5), cfl=scene['numerics']['cfl'])
        self.assertTrue(np.array_equal(run['computed']['probes_m'], reference['probes_m']))
        lined = plate(); lined['acquisition'] = v1['acquisition']; lined['receiver_lines'][0]['to_mm'] = [13., 1., 1.]
        with self.assertRaises(ValueError): simulate(lined, 2)


class ConvergenceCampaignTests(unittest.TestCase):
    def test_three_dimensional_campaigns_converge_and_summarize(self):
        for engine, levels, grid in [('3d_spectral', ['space', 'degree', 'time'], {'constitutive': ['linear', 'murnaghan']}), ('3d_tetra', ['degree', 'time'], {'source.amplitude_mpa': [1.]})]:
            with self.subTest(engine=engine), tempfile.TemporaryDirectory() as tmp:
                s = validate_spec(spec(engine, degree=2, grid=grid, convergence={'levels': levels, 'cases': {'constitutive': 'murnaghan'}} if 'constitutive' in grid else {'levels': levels}))
                root = Path(tmp) / 'campaign'; campaign = Campaign.create(s, root); result = campaign.run(s)
                self.assertEqual(result['completed'], len(next(iter(grid.values()))), [c.get('error') for c in campaign.manifest['cases']]); self.assertEqual(campaign.manifest['sweep_engine'], engine)
                cid = campaign.manifest['cases'][-1]['case_id']; record = json.loads((root / 'cases' / cid / 'out' / 'result.json').read_text())
                self.assertEqual(record['dimension'], '3d'); self.assertEqual(sorted(k for k in record['receivers'][0] if k.startswith('u_')), ['u_x_m', 'u_y_m', 'u_z_m'])
                checked = campaign.converge(s); self.assertEqual(checked['cases'], len(next(iter(grid.values()))))
                report = json.loads((root / 'cases' / cid / 'convergence' / 'report.json').read_text()); rows = {r['level']: r for r in report['levels']}
                if engine == '3d_tetra': self.assertIn('skipped', report['summary']['levels']['degree'])
                for level in [l for l in levels if not rows[l].get('skipped')]:
                    comparison = rows[level]['comparison']; self.assertIsNone(rows[level].get('error')); self.assertEqual(len(comparison['receivers']), 4)
                    key = 'internal_step_us' if level == 'time' else 'nodes'
                    self.assertNotEqual(comparison['refined_diagnostics'][key], comparison['base_diagnostics'][key])
                entry = json.loads((root / 'manifest.json').read_text())['cases'][-1]['convergence']; self.assertEqual(entry['pass'], report['summary']['pass']); self.assertEqual(len(entry['report_sha256']), 64)
                if 'constitutive' in grid:
                    self.assertEqual(checked['increment_pairs'], 1); increment = json.loads((root / 'cases' / cid / 'convergence' / 'increment.json').read_text())
                    self.assertEqual(increment['linear_control'], campaign.manifest['cases'][0]['case_id']); self.assertEqual(entry['increment']['pass'], increment['summary']['pass'])
                    for row in increment['levels']:
                        a2 = row['comparison']['receivers'][0]['harmonics']['a2']; self.assertTrue(a2['available']); self.assertGreater(a2['base_m'], 0); self.assertEqual(row['comparison']['verdict_terms'], ['a2'])
                else: self.assertEqual(checked['increment_pairs'], 0)
                summary = summarize(root); self.assertEqual(summary['rows'][-1]['convergence']['pass'], report['summary']['pass'])
                header = (root / 'summary.csv').read_text().splitlines()[0]; self.assertIn('convergence_pass', header)


if __name__ == '__main__': unittest.main()
