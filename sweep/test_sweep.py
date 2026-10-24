"""Sweep specification, planning, resumable execution, summaries and datasets on tiny plane-strain cases."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from solid.schema import default_scene
from physics.scene import V2
from .spec import validate_spec, expand_cases, build_scene, plan, assign_splits, set_path, get_path
from .runner import Campaign
from .summarize import summarize
from .dataset import build_dataset, load_dataset, select


def base_scene():
    s = default_scene('plate'); s['schema'] = V2; s['dimension'] = '2d_plane_strain'; s['constitutive'] = 'murnaghan'
    s['geometry'].update(length_mm=24., width_mm=1., thickness_mm=2.); s['numerics']['h_mm'] = 1.5
    s['source'].update(center_mm=[0., 0., 1.], frequency_mhz=.25, cycles=3., amplitude_mpa=1.)
    s['acquisition'].update(duration_us=20., sample_step_us=.05)
    s['receivers'] = [{'name': 'P', 'point_mm': [8., 0., 1.], 'direction': [1., 0., 0.]}]
    s['receiver_lines'] = [{'name': 'top', 'from_mm': [6., 0., 2.], 'to_mm': [18., 0., 2.], 'count': 7, 'direction': [1., 0., 0.]}]
    return s


def spec(**changes):
    s = {'schema': 'workai.sweep.v1', 'name': 'tiny', 'base_scene': base_scene(), 'degree': 3, 'grid': {'source.amplitude_mpa': [1., 2.], 'constitutive': ['linear', 'murnaghan']},
         'analysis': {'line': 'top', 'window': {'mode': 'moving'}}, 'split': {'train': .5, 'validation': .25, 'test': .25, 'group_by': ['source.amplitude_mpa'], 'seed': 3}, 'budget': {'seconds_per_case': 600, 'workers': 2}}
    s.update(changes); return s


class SweepTests(unittest.TestCase):
    def test_paths_expansion_links_and_ids(self):
        s = validate_spec(spec(grid={'source.amplitude_mpa': [1., 2.], 'source.frequency_mhz': [.25, .3]}, links={'source.cycles': {'path': 'source.frequency_mhz', 'scale': 12., 'offset': 0.}}, split={'train': .5, 'validation': .25, 'test': .25, 'group_by': ['source.frequency_mhz'], 'seed': 3}))
        rows = expand_cases(s); self.assertEqual(len(rows), 4)
        scene = build_scene(s, {'source.frequency_mhz': .3, 'source.amplitude_mpa': 2.}); self.assertAlmostEqual(scene['source']['cycles'], 3.6); self.assertEqual(scene['source']['amplitude_mpa'], 2.)
        planned = plan(s); self.assertEqual(len({r['case_id'] for r in planned}), 4); self.assertEqual(planned, plan(s))
        splits = [r['split'] for r in planned]; groups = {}
        for r in planned: groups.setdefault(r['parameters']['source.frequency_mhz'], set()).add(r['split'])
        self.assertTrue(all(len(v) == 1 for v in groups.values()), 'cases of one group must share a split'); self.assertEqual(len(set(splits)), 2)
        obj = {'a': [1, {'b': [0, 2]}]}; set_path(obj, 'a.1.b.1', 7); self.assertEqual(get_path(obj, 'a.1.b.1'), 7)
        explicit = validate_spec(spec(grid={}, cases=[{'geometry.length_mm': 20.}, {'geometry.length_mm': 28.}])); self.assertEqual(expand_cases(explicit), [{'geometry.length_mm': 20.}, {'geometry.length_mm': 28.}])

    def test_sampling_is_reproducible_and_bounded(self):
        s = validate_spec(spec(grid={}, sampling={'method': 'latin_hypercube', 'count': 12, 'seed': 5, 'ranges': {'geometry.thickness_mm': [1.5, 2.5], 'source.frequency_mhz': [.2, .3]}},
                              links={'source.center_mm.2': {'path': 'geometry.thickness_mm', 'scale': .5, 'offset': 0.}, 'receivers.0.point_mm.2': {'path': 'geometry.thickness_mm', 'scale': .5, 'offset': 0.},
                                     'receiver_lines.0.from_mm.2': {'path': 'geometry.thickness_mm', 'scale': 1., 'offset': 0.}, 'receiver_lines.0.to_mm.2': {'path': 'geometry.thickness_mm', 'scale': 1., 'offset': 0.}},
                              split={'train': .5, 'validation': .25, 'test': .25, 'group_by': ['geometry.thickness_mm'], 'seed': 3}))
        rows = expand_cases(s); again = expand_cases(s); self.assertEqual(rows, again); self.assertEqual(len(rows), 12)
        thickness = sorted(r['geometry.thickness_mm'] for r in rows); self.assertTrue(all(1.5 <= v <= 2.5 for v in thickness))
        # Latin hypercube: exactly one sample per stratum.
        strata = sorted(int((v - 1.5) / (1. / 12) - 1e-12) for v in thickness); self.assertEqual(strata, list(range(12)))
        planned = plan(s); self.assertEqual(len(planned), 12); self.assertTrue(all(r['scene']['source']['center_mm'][2] == r['scene']['geometry']['thickness_mm'] / 2 for r in planned))
        self.assertEqual(sum(1 for r in planned if r['split'] == 'train'), 6)

    def test_specification_rejections(self):
        for mutate, message in [(lambda s: s.update(schema='x'), 'schema'), (lambda s: s.update(degree=9), '阶次'), (lambda s: s.update(grid={'a': 'not-a-list'}), '网格'),
                                (lambda s: s.update(sampling={'method': 'sobol', 'count': 2, 'seed': 1, 'ranges': {'a': [0, 1]}}), '采样方法'), (lambda s: s.update(links={'x': {'path': 'y'}}), '联动'),
                                (lambda s: s.update(split={'train': .5, 'validation': .5, 'test': .5, 'group_by': [], 'seed': 1}), '划分比例'), (lambda s: s['base_scene'].update(dimension='3d'), '2d_plane_strain'),
                                (lambda s: s.update(unknown=1), '只接受字段')]:
            bad = spec(); mutate(bad)
            with self.assertRaises(ValueError) as caught: validate_spec(bad)
            self.assertIn(message, str(caught.exception))
        with self.assertRaises(ValueError): build_scene(validate_spec(spec()), {'source.frequency_mhz': 5.})

    def test_campaign_runs_resumes_summarizes_and_builds_a_dataset(self):
        s = validate_spec(spec())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'campaign'; campaign = Campaign.create(s, root); self.assertEqual(len(campaign.manifest['cases']), 4)
            events = []; result = campaign.run(s, workers=2, progress=events.append)
            self.assertEqual(result, {'cases': 4, 'completed': 4, 'failed': 0, 'skipped_already_complete': 0}, (result, [c.get('error') for c in campaign.manifest['cases']]))
            manifest = json.loads((root / 'manifest.json').read_text()); self.assertTrue(all(c['phase'] == 'completed' and len(c['result_sha256']) == 64 for c in manifest['cases']))
            self.assertTrue((root / 'cases' / manifest['cases'][0]['case_id'] / 'out' / 'lines.npz').is_file()); self.assertIn('spectral2d/solver.py', manifest['source_sha256'])
            # Resume: nothing re-runs, a tampered result re-runs.
            first_times = {c['case_id']: c['completed_at_utc'] for c in manifest['cases']}
            tampered = manifest['cases'][1]['case_id']; (root / 'cases' / tampered / 'out' / 'result.json').write_text('{}')
            resumed = Campaign.create(s, root, resume=True); again = resumed.run(s, workers=1)
            self.assertEqual(again['skipped_already_complete'], 3); self.assertEqual(again['completed'], 4)
            for c in resumed.manifest['cases']:
                if c['case_id'] != tampered: self.assertEqual(c['completed_at_utc'], first_times[c['case_id']])
            with self.assertRaises(ValueError): Campaign.create(validate_spec(spec(name='other')), root, resume=True)
            with self.assertRaises(FileExistsError): Campaign.create(s, root)
            summary = summarize(root); self.assertEqual(summary['completed_cases'], 4); self.assertEqual(len(summary['control_pairs']), 2)
            # Raw 2f of a 3-cycle burst is the burst's own linear content: it scales by 2 and must be flagged.
            raw = [p for p in summary['amplitude_scaling_pairs'] if p['basis'] == 'raw']; self.assertEqual(len(raw), 2)
            murnaghan = next(p for p in raw if 'increment' in p); self.assertFalse(murnaghan['all_pass']); self.assertTrue(all(abs(c['a2_ratio'] - 2) < .15 for c in murnaghan['checks']), murnaghan['checks'])
            # Subtracting the linear control on the same mesh leaves the nonlinear increment, which scales by 4.
            self.assertTrue(murnaghan['increment']['all_pass'], murnaghan['increment']); self.assertTrue(all(abs(q - 4) < .4 for q in murnaghan['increment']['a2_ratios']))
            nonlinear_rows = [r for r in summary['rows'] if r['parameters']['constitutive'] == 'murnaghan']; self.assertTrue(all('nonlinear_increment' in r['line'] for r in nonlinear_rows))
            row = summary['rows'][0]; self.assertIn('phase_velocity_m_s', row['line']); self.assertGreater(row['theory_phase_velocity_m_s'], 1000); self.assertLessEqual(row['energy_balance_relative'], .01)
            self.assertEqual(len((root / 'summary.csv').read_text().strip().split('\n')), 5)
            meta = build_dataset(root, root / 'dataset.npz'); self.assertEqual(meta['cases'], 4); self.assertEqual(meta['shapes']['line_projected_m'], [4, 401, 7]); self.assertEqual(meta['shapes']['receiver_u_m'], [4, 1, 2, 401])
            data = load_dataset(root / 'dataset.npz'); self.assertEqual(list(data['parameter_names']), ['source.amplitude_mpa']); self.assertEqual(data['manifest']['engine'], 'workai_spectral_plane_strain_2d')
            train = select(data, 'train'); self.assertEqual(len(train['case_ids']), int(np.sum(data['split'] == 'train'))); self.assertEqual(train['line_projected_m'].shape[1:], (401, 7)); self.assertEqual(len(train['times_us']), 401)
            # Doubling the amplitude doubles the stored traces (linear part dominates).
            ids = list(data['case_ids']); by = {(c['parameters']['source.amplitude_mpa'], c['parameters']['constitutive']): ids.index(c['case_id']) for c in manifest['cases']}
            a = data['line_projected_m'][by[(1., 'murnaghan')]]; b = data['line_projected_m'][by[(2., 'murnaghan')]]; self.assertLess(np.max(np.abs(b - 2 * a)) / np.max(np.abs(b)), 1e-2)


if __name__ == '__main__': unittest.main()
