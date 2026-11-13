"""The machine limits the model, not the display: resource estimates, record modes, large meshes and display results."""
import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
from offline import resources, worker
from offline.domain import validate_request
from offline.preview import preview
from offline.recording import traces
from offline.test_network_v2 import v2_plate


def strip(length=100., width=20.):
    """The workbench template (top Gaussian source, absorbing ends, half model) on a larger plate."""
    s = v2_plate(); s['geometry'].update(length_mm=length, width_mm=width, thickness_mm=1.6); s['sources'] = []
    s['source'] = {'name': '顶面高斯', 'face': 'zmax', 'center_mm': [16., width / 2, 1.6], 'radius_mm': 0., 'direction': [0., 0., -1.], 'amplitude_mpa': 1., 'frequency_mhz': .4, 'cycles': 5.,
                   'delay_us': 0., 'phase_deg': 0., 'signal': {'type': 'tone_burst', 'window': 'hann'}, 'profile': {'type': 'gaussian', 'sigma_mm': 2.}}
    s['boundary'] = {'fixed_face': 'none'}
    s['boundaries'] = [{'type': 'symmetry', 'face': 'ymax'}, {'type': 'absorbing', 'face': 'xmin', 'length_mm': 8., 'ramp': 'cubic', 'attenuation_db': 60.},
                       {'type': 'absorbing', 'face': 'xmax', 'length_mm': 8., 'ramp': 'cubic', 'attenuation_db': 60.}]
    s['acquisition'] = {'duration_us': 30., 'sample_step_us': .05}
    s['receivers'] = [{'name': f'top_{i}', 'point_mm': [x, width / 2, 1.6], 'direction': [0., 0., 1.]} for i, x in enumerate((22., 28., 34.))]
    return s


def run_job(directory, request):
    directory.mkdir(parents=True)
    (directory / 'request.json').write_text(json.dumps(request), encoding='utf-8'); (directory / 'status.json').write_text(json.dumps({'job_id': directory.name, 'phase': 'starting'}), encoding='utf-8')
    worker.run(directory); return json.loads((directory / 'status.json').read_text(encoding='utf-8'))


class Requests(unittest.TestCase):
    def test_record_and_time_layer_settings(self):
        self.assertNotIn('record', validate_request({'scene': v2_plate(), 'degree': 3}))
        result = validate_request({'scene': v2_plate(), 'degree': 3, 'record': 'compact', 'time_stepping': 'frozen'})
        self.assertEqual((result['record'], result['time_stepping']), ('compact', 'frozen')); self.assertEqual(validate_request(copy.deepcopy(result)), result)
        for bad in ({'record': 'tiny'}, {'time_stepping': 'fast'}, {'record': 'full', 'engine': 'coarse'}, {'other': 1}):
            with self.assertRaises(ValueError): validate_request({'scene': v2_plate(), 'degree': 3, **bad})
        v1 = json.loads((Path(__file__).parent / 'example-request.json').read_text(encoding='utf-8'))
        with self.assertRaises(ValueError): validate_request({'scene': v1['scene'], 'degree': v1['degree'], 'record': 'compact'})

    def test_record_mode_and_memory_limit(self):
        self.assertEqual(resources.record_mode('auto', 601, 13689), 'full'); self.assertEqual(resources.record_mode('auto', 601, 229293), 'compact')
        self.assertEqual(resources.record_mode('full', 601, 229293), 'full'); self.assertEqual(resources.record_mode('compact', 601, 100), 'compact')
        self.assertGreater(resources.physical_memory() or 1, 0)
        with mock.patch.object(resources, 'physical_memory', return_value=4 * 1024 ** 3):
            self.assertIsNone(resources.memory_problem(1_000_000)); self.assertIn('GB 内存', resources.memory_problem(3_000_000))
        with mock.patch.object(resources, 'physical_memory', return_value=None): self.assertIsNone(resources.memory_problem(10 ** 8))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(resources.throughput(tmp), (resources.DEFAULT_RATE, False))
            resources.save_throughput(tmp, 1000, 100, 1.); self.assertEqual(resources.throughput(tmp)[1], False)   # too small to measure a rate
            resources.save_throughput(tmp, 200000, 6000, 400.); self.assertEqual(resources.throughput(tmp), (3e6, True))


class LargeModels(unittest.TestCase):
    def test_a_plate_beyond_the_old_display_budget_keeps_its_accuracy_mesh(self):
        request = validate_request({'scene': strip(), 'mesh': 'auto'}); report = request['mesh_report']
        self.assertGreater(report['nodes'], 200000); self.assertEqual(report['limited_by'], 'accuracy'); self.assertNotIn('warning', report)
        shown = preview({'scene': strip(), 'mesh': 'auto'}, '.'); self.assertTrue(shown['ok'], shown['errors'])
        cost = shown['estimate']; self.assertEqual((cost['record'], shown['record']['mode']), ('compact', 'compact')); self.assertEqual(cost['scheme'], 'verlet2')
        self.assertLess(cost['evaluations_per_control'], cost['frozen_evaluations_per_control'] / 2); self.assertLess(cost['disk_bytes'], 601 * report['nodes'] * 72 / 4)
        self.assertGreater(cost['seconds'], 0); self.assertLess(cost['memory_bytes'], 2 * 1024 ** 3); self.assertEqual(shown['record']['uncompressed_bytes'], cost['disk_bytes'])
        small = preview({'scene': v2_plate(), 'degree': 3}, '.'); self.assertEqual(small['estimate']['record'], 'full'); self.assertEqual(small['record']['mode'], 'full')
        self.assertEqual(preview({'scene': v2_plate(), 'degree': 3, 'time_stepping': 'frozen'}, '.')['estimate']['scheme'], 'composition4')

    def test_the_machine_memory_coarsens_or_refuses_with_advice(self):
        nodes = validate_request({'scene': strip(), 'mesh': 'auto'})['mesh_report']['nodes']
        with mock.patch.object(resources, 'physical_memory', return_value=int((resources.BASE_BYTES + .8 * nodes * resources.NODE_BYTES) / resources.MEMORY_FRACTION)):
            report = validate_request({'scene': strip(), 'mesh': 'auto'})['mesh_report']
            self.assertEqual(report['limited_by'], 'budget'); self.assertLess(report['nodes'], nodes); self.assertIn('warning', report)
        with mock.patch.object(resources, 'physical_memory', return_value=600 * 1024 ** 2):
            with self.assertRaises(ValueError) as refused: validate_request({'scene': strip(), 'mesh': 'auto'})
            self.assertIn('模型太大', str(refused.exception)); self.assertIn('GB 内存', str(refused.exception)); self.assertIn('对称', str(refused.exception))
            shown = preview({'scene': strip(), 'mesh': 'auto'}, '.'); self.assertFalse(shown['ok']); self.assertTrue(any('模型太大' in e for e in shown['errors']))


class Jobs(unittest.TestCase):
    def test_compact_job_exports_display_results_and_receiver_waveforms(self):
        scene = v2_plate()
        with tempfile.TemporaryDirectory() as tmp:
            status = run_job(Path(tmp) / 'compact', validate_request({'scene': scene, 'degree': 3, 'record': 'compact'}))
            self.assertEqual((status['phase'], status['record_mode'], status['resumable']), ('completed', 'compact', True)); self.assertEqual(status['time_plan']['scheme'], 'verlet2')
            full_status = run_job(Path(tmp) / 'full', validate_request({'scene': scene, 'degree': 3})); self.assertEqual(full_status['record_mode'], 'full')
            a = json.loads((Path(tmp) / 'compact' / 'full.json').read_text(encoding='utf-8')); b = json.loads((Path(tmp) / 'full' / 'full.json').read_text(encoding='utf-8'))
            self.assertNotIn('display', b); self.assertEqual(b['field']['dtype'], 'little_endian_float64'); self.assertEqual(b['recording']['analysis_source'], 'complete_float64_chunks')
            display = a['display']; self.assertEqual(a['field']['dtype'], 'little_endian_float32'); self.assertEqual(a['field']['shape'], [len(a['times_us']), display['nodes'], 3])
            self.assertEqual((len(a['nodes_mm']), len(a['surface'])), (display['nodes'], display['triangles'])); self.assertEqual(display['computation_nodes'], len(b['nodes_mm']))
            self.assertEqual(a['spectral']['node_count'], len(b['nodes_mm'])); self.assertNotIn('elements', a['spectral']); self.assertEqual(len(a['surface']), len(b['surface']))
            self.assertEqual(a['recording']['record_mode'], 'compact'); self.assertEqual(a['recording']['analysis_source'], 'receiver_float64_probes')
            self.assertEqual([r['name'] for r in a['recording']['receivers']], [r['name'] for r in scene['receivers']]); self.assertTrue(a['loaded_faces'])
            # the animation is the full field at the display nodes (same frames), as float32
            ids = {tuple(np.round(p, 9)): i for i, p in enumerate(b['nodes_mm'])}; nodes = [ids[tuple(np.round(p, 9))] for p in a['nodes_mm']]
            self.assertEqual(a['times_us'], b['times_us'])
            for part in ('full', 'material'):
                shown = np.frombuffer(gzip.decompress((Path(tmp) / 'compact' / f'{part}.bin.gz').read_bytes()), dtype='<f4').reshape(a['field']['shape'])
                whole = np.frombuffer(gzip.decompress((Path(tmp) / 'full' / f'{part}.bin.gz').read_bytes()), dtype='<f8').reshape(b['field']['shape'])[:, nodes]
                self.assertLess(np.max(np.abs(shown - whole)) / np.max(np.abs(whole)), 1e-6, part)
            # waveforms: every sample at the receivers, equal to the full record; other points are refused with the reason
            points = [r['point_mm'] for r in scene['receivers']]
            scale = max(np.max(np.abs(p['xyz_m'])) for p in traces(Path(tmp) / 'full' / 'record', 'full', points)['points'])   # increments differ by rounding of the full field
            for part in ('full', 'linear', 'geometric', 'material'):
                x = traces(Path(tmp) / 'compact' / 'record', part, points); y = traces(Path(tmp) / 'full' / 'record', part, points)
                self.assertEqual(x['times_us'], y['times_us'])
                for p, q in zip(x['points'], y['points']):
                    self.assertLess(np.max(np.abs(np.array(p['xyz_m']) - np.array(q['xyz_m']))), 1e-12 * scale, part)
            with self.assertRaises(ValueError) as refused: traces(Path(tmp) / 'compact' / 'record', 'full', [[5., 1., 1.]])
            self.assertIn('精简记录', str(refused.exception)); self.assertIn(scene['receivers'][0]['name'], str(refused.exception))
            self.assertLess(sum(f.stat().st_size for f in (Path(tmp) / 'compact').rglob('*.npz')), sum(f.stat().st_size for f in (Path(tmp) / 'full').rglob('*.npz')) / 2)

    def test_a_full_record_too_large_for_the_browser_is_shown_on_the_display_mesh(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(resources, 'BROWSER_NODES', 100):
            status = run_job(Path(tmp) / 'job', validate_request({'scene': v2_plate(), 'degree': 3, 'time_stepping': 'frozen'}))
            self.assertEqual((status['phase'], status['record_mode']), ('completed', 'full')); self.assertEqual(status['time_plan']['scheme'], 'composition4')
            full = json.loads((Path(tmp) / 'job' / 'full.json').read_text(encoding='utf-8'))
            self.assertIn('display', full); self.assertEqual(full['recording']['record_mode'], 'full'); self.assertEqual(full['recording']['analysis_source'], 'complete_float64_chunks')
            self.assertEqual(len(traces(Path(tmp) / 'job' / 'record', 'full', [[5., 1., 1.]])['points']), 1)   # a full record still serves any point


class Finishing(unittest.TestCase):
    def test_a_worker_that_published_its_final_status_does_not_block_the_next_job(self):
        from offline.app import Application, save

        class Exiting:   # a worker process that ends when it is waited for
            def __init__(self): self.done = False; self.waited = None
            def poll(self): return 0 if self.done else None
            def wait(self, timeout=None): self.waited = timeout; self.done = True; return 0

        class Stuck(Exiting):
            def wait(self, timeout=None):
                import subprocess
                raise subprocess.TimeoutExpired('worker', timeout)

        with tempfile.TemporaryDirectory() as tmp:
            app = Application(Path(tmp)); self.assertTrue(app.idle())
            job = 'local-' + '0' * 32; (app.jobs / job).mkdir(); app.current = job
            status = lambda phase: save(app.jobs / job / 'status.json', {'job_id': job, 'phase': phase, 'started_at_utc': '2026-11-13T00:00:00+00:00'})
            status('running'); app.process = Exiting(); self.assertFalse(app.idle()); self.assertIsNone(app.process.waited)   # really running: refuse at once
            for phase in ('completed', 'failed', 'paused'):
                status(phase); app.process = Exiting(); self.assertTrue(app.idle()); self.assertIsNotNone(app.process.waited)
            status('completed'); app.process = Stuck(); self.assertFalse(app.idle(wait_s=.01))
            with self.assertRaises(ValueError): app.submit({'scene': v2_plate(), 'degree': 3})


if __name__ == '__main__': unittest.main()
