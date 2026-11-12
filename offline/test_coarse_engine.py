"""Fast engine (learned coarse-graph operator) in the offline software: request rules, preview and a complete job."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import re

from offline.coarse_engine import DEFAULT_MODEL, EXCITATIONS, MODELS, model_summary, out_of_range
from offline.domain import validate_request
from offline.preview import preview

REPO = Path(__file__).resolve().parents[1]


def two_patch(duration=30.):
    s = json.loads((REPO / 'coarsegraph/specs/labels-c-two-patch.json').read_text(encoding='utf-8'))['base_scene']
    s['numerics']['h_mm'] = 1.5; s['acquisition']['duration_us'] = duration
    return s


class FastEngine(unittest.TestCase):
    def test_requests_previews_and_limits(self):
        s = two_patch()
        self.assertEqual(validate_request({'scene': s, 'degree': 4, 'engine': 'coarse'})['engine'], 'coarse')
        self.assertNotIn('engine', validate_request({'scene': s, 'degree': 4}))
        with self.assertRaises(ValueError): validate_request({'scene': s, 'degree': 4, 'engine': 'solver'})
        from physics.materials import presets
        cfrp = copy.deepcopy(s); cfrp['material'] = presets()['cfrp_ud']['material']
        for bad, word in [(cfrp, '铝'), (dict(s, geometry=dict(s['geometry'], kind='notch')), '平板'), (dict(s, acquisition={'duration_us': 300., 'sample_step_us': .05}), '采样点')]:
            with self.assertRaises(ValueError) as caught: validate_request({'scene': bad, 'degree': 4, 'engine': 'coarse'})
            self.assertIn(word, str(caught.exception))
        p = preview({'scene': s, 'degree': 4, 'engine': 'coarse'}, '.')
        summary = model_summary(); self.assertTrue(p['ok'], p['errors'])
        self.assertEqual(p['mesh']['degree'], summary['degree']); self.assertEqual(p['mesh']['h_mm'], summary['h_mm'])
        self.assertEqual(len(p['engine']['out_of_training_range']), 1); self.assertIn('2 个激励源', p['engine']['out_of_training_range'][0])
        self.assertLess(p['mesh']['nodes'], preview({'scene': s, 'degree': 4}, '.')['mesh']['nodes'] / 4)
        wide = copy.deepcopy(s); wide['geometry']['thickness_mm'] = 3.; wide['source']['frequency_mhz'] = .5
        self.assertEqual(len(out_of_range(wide)), 3)

    def test_model_choice(self):
        s = two_patch()
        self.assertEqual(validate_request({'scene': s, 'degree': 4, 'engine': 'coarse'})['coarse_model'], DEFAULT_MODEL)
        for name in MODELS:
            self.assertEqual(validate_request({'scene': s, 'degree': 4, 'engine': 'coarse', 'coarse_model': name})['coarse_model'], name)
            p = preview({'scene': s, 'degree': 4, 'engine': 'coarse', 'coarse_model': name}, '.'); self.assertTrue(p['ok'], p['errors'])
            self.assertEqual((p['engine']['name'], p['mesh']['h_mm']), (name, model_summary(REPO / 'coarsegraph/models' / name / 'model.json')['h_mm']))
            self.assertEqual(len(p['engine']['out_of_training_range']), 1); self.assertIn('2 个激励源', p['engine']['out_of_training_range'][0])
        for bad in [{'engine': 'coarse', 'coarse_model': 'p9-h1'}, {'coarse_model': DEFAULT_MODEL}]:
            with self.assertRaises(ValueError): validate_request(dict({'scene': s, 'degree': 4}, **bad))
        fast, accurate = (preview({'scene': s, 'degree': 4, 'engine': 'coarse', 'coarse_model': name}, '.')['mesh']['nodes'] for name in (DEFAULT_MODEL, 'p4-h2-multisource'))
        self.assertLess(fast, accurate)
        # the accurate model's recorded held-out test: every top, end and two-patch case passes; the comb is not representable on 2 mm
        forms = {r['group']: (r['passed'], r['cases']) for r in model_summary(REPO / 'coarsegraph/models/p4-h2-multisource/model.json')['test_by_excitation']}
        self.assertEqual(forms, {'labels-a-top-gaussian': (2, 2), 'labels-b-end-axial': (2, 2), 'labels-c-two-patch': (8, 8), 'labels-d-comb': (0, 4)})
        self.assertIn('精确模型', out_of_range(s)[0]); self.assertIn('其他多源组合', out_of_range(s, 'p4-h2-multisource')[0])
        listed = re.search(r'COARSE_MODELS = \{([^}]*)\}', (REPO / 'offline/web/inspector.js').read_text(encoding='utf-8')).group(1)
        self.assertEqual(re.findall(r"'([\w.-]+)':", listed), list(MODELS))  # the interface offers exactly the shipped models, default first

    def test_excitation_forms(self):
        a = json.loads((REPO / 'coarsegraph/specs/labels-a-top-gaussian.json').read_text(encoding='utf-8'))['base_scene']
        self.assertEqual(out_of_range(a), [])
        for change, word in [({'profile': {'type': 'comb', 'pitch_mm': 4., 'teeth': 3, 'tooth_width_mm': 2., 'along': 'x', 'alternating_sign': True}}, '梳状'),
                             ({'profile': {'type': 'gaussian', 'sigma_mm': .5}}, '高斯宽度'), ({'radius_mm': 3.}, '圆形'), ({'cycles': 2.}, '周期'),
                             ({'signal': {'type': 'ricker'}}, 'ricker'), ({'face': 'ymin', 'direction': [0., 1., 0.]}, '加载面')]:
            scene = copy.deepcopy(a); scene['source'].update(change); items = out_of_range(scene)
            self.assertEqual(len(items), 1, items); self.assertIn(word, items[0])
        forms = model_summary()['test_by_excitation']
        self.assertEqual([r['group'] for r in forms], list(EXCITATIONS))
        self.assertEqual(sum(r['cases'] for r in forms), model_summary()['test_cases']); self.assertEqual(sum(r['passed'] for r in forms), model_summary()['test_passed'])

    def test_a_windows_checkout_with_crlf_line_endings_still_matches_its_evaluation(self):
        # v1.0.0 on Windows: core.autocrlf rewrote model.json to CRLF and every fast-engine preview failed with
        # "evaluation-abcd.json 不是这个模型的评估。"
        for name in MODELS:
            source = REPO / 'coarsegraph/models' / name
            with tempfile.TemporaryDirectory() as tmp:
                copy_dir = Path(tmp) / name; copy_dir.mkdir()
                for file in ('model.json', 'evaluation-abcd.json'):
                    (copy_dir / file).write_bytes((source / file).read_bytes().replace(b'\n', b'\r\n'))
                self.assertIn(b'\r\n', (copy_dir / 'model.json').read_bytes())
                crlf = model_summary(copy_dir / 'model.json')
                self.assertEqual(crlf['model_sha256'], model_summary(source / 'model.json')['model_sha256'])
                self.assertEqual([r['group'] for r in crlf['test_by_excitation']], list(EXCITATIONS))
                tampered = json.loads((source / 'model.json').read_text(encoding='utf-8')); tampered['h_mm'] += 1
                (copy_dir / 'model.json').write_text(json.dumps(tampered), encoding='utf-8')
                with self.assertRaises(ValueError) as caught: model_summary(copy_dir / 'model.json')
                self.assertIn('不是这个模型的评估', str(caught.exception))

    def test_a_job_runs_and_exports_the_five_responses(self):
        for name, duration in [(DEFAULT_MODEL, 20.), ('p4-h2-multisource', 20.)]:
            with self.subTest(model=name): self._job(name, duration)

    def _job(self, name, duration):
        request = validate_request({'scene': two_patch(duration), 'degree': 4, 'engine': 'coarse', 'coarse_model': name})
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / 'local-fast'; job.mkdir()
            (job / 'request.json').write_text(json.dumps(request), encoding='utf-8')
            (job / 'status.json').write_text(json.dumps({'job_id': job.name, 'phase': 'starting'}), encoding='utf-8')
            done = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'offline.worker', '--job-dir', str(job)], cwd=REPO, capture_output=True, text=True)
            status = json.loads((job / 'status.json').read_text(encoding='utf-8'))
            self.assertEqual(done.returncode, 0, done.stderr[-2000:]); self.assertEqual(status['phase'], 'completed', status); self.assertEqual((status['engine'], status['coarse_model']), ('coarse', name))
            self.assertEqual(set(json.loads((job / 'exports.json').read_text(encoding='utf-8'))), {'full', 'linear', 'stvk', 'geometric', 'material'})
            full = json.loads((job / 'full.json').read_text(encoding='utf-8'))
            self.assertEqual(full['engine'], 'workai_coarse_graph_offline'); self.assertEqual(full['coarse_model']['name'], name)
            self.assertEqual(full['controls'][0]['h_mm'], model_summary(REPO / 'coarsegraph/models' / name / 'model.json')['h_mm'])
            self.assertEqual(len(full['times_us']), round(duration / request['scene']['acquisition']['sample_step_us']) + 1); self.assertTrue(all(c['energy_balance_within_1_percent'] for c in full['controls']))
            self.assertTrue(full['loaded_faces'])


if __name__ == '__main__': unittest.main()
