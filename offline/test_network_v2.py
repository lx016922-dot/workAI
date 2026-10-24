"""Version-2 scenes in the offline software: validation and one complete local job."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from .domain import templates, validate_request

REPO = Path(__file__).resolve().parents[1]


def v2_plate():
    s = templates()['plate']; s['schema'] = 'workai.hyperelastic.solid.v2'
    s['source'].update(cycles=3., signal={'type': 'tone_burst', 'window': 'tukey', 'tukey_alpha': .5})
    s['sources'] = [{'name': 'top patch', 'face': 'zmax', 'center_mm': [6., 1.75, 1.8], 'radius_mm': 0., 'direction': [0., 0., -1.], 'amplitude_mpa': .8, 'frequency_mhz': .2,
                     'cycles': 2., 'delay_us': 2., 'phase_deg': 0., 'signal': {'type': 'tone_burst', 'window': 'hann'}, 'profile': {'type': 'gaussian', 'sigma_mm': 1.}}]
    s['boundaries'] = [{'type': 'symmetry', 'face': 'ymax'}, {'type': 'absorbing', 'face': 'xmax', 'length_mm': 3., 'ramp': 'cubic', 'attenuation_db': 60.}]
    s['acquisition'] = {'duration_us': 20., 'sample_step_us': .05}
    return s


class NetworkV2Requests(unittest.TestCase):
    def test_validation_accepts_v2_scenes_and_keeps_bounds(self):
        s = v2_plate(); s['signal'] = {'kind': 'hann', 'end_frequency_mhz': .28}
        result = validate_request({'scene': s, 'degree': 3})
        self.assertNotIn('signal', result['scene']); self.assertEqual(len(result['scene']['sources']), 1)
        self.assertEqual(validate_request(copy.deepcopy(result))['scene'], result['scene'])
        for mutate in (lambda x: x['acquisition'].update(sample_step_us=.5), lambda x: x['acquisition'].update(duration_us=25000.),
                       lambda x: x['numerics'].update(h_mm=3.), lambda x: x['material'].update(l_pa=-1e11), lambda x: x['numerics'].update(cfl=.34)):
            bad = v2_plate(); mutate(bad)
            with self.assertRaises(ValueError): validate_request({'scene': bad, 'degree': 3})
        with self.assertRaises(ValueError): validate_request({'scene': v2_plate(), 'degree': 2})

    def test_local_job_runs_through_the_v2_input_layer(self):
        request = validate_request({'scene': v2_plate(), 'degree': 3})
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / 'local-test'; job.mkdir()
            (job / 'request.json').write_text(json.dumps(request), encoding='utf-8')
            (job / 'status.json').write_text(json.dumps({'job_id': job.name, 'phase': 'starting'}), encoding='utf-8')
            done = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'offline.worker', '--job-dir', str(job)], cwd=REPO, capture_output=True, text=True)
            status = json.loads((job / 'status.json').read_text(encoding='utf-8'))
            self.assertEqual(done.returncode, 0, done.stderr[-2000:]); self.assertEqual(status['phase'], 'completed', status)
            recording = json.loads((job / 'record' / 'recording.json').read_text(encoding='utf-8'))
            self.assertEqual(recording['schema'], 'workai.graphnet.long_record.v1'); self.assertEqual(recording['input_layer'], 'v2_multi_source')
            self.assertEqual([r['name'] for r in recording['excitation']['sources']], ['primary', 'top patch'])
            self.assertGreater(recording['dissipated_energy_j'][2], 0)
            exports = json.loads((job / 'exports.json').read_text(encoding='utf-8')); self.assertEqual(set(exports), {'full', 'linear', 'stvk', 'geometric', 'material'})
            full = json.loads((job / 'full.json').read_text(encoding='utf-8'))
            self.assertEqual(full['input_layer'], 'v2_multi_source'); self.assertTrue(full['neural_model']['neural_forward_only']); self.assertFalse(full['neural_model']['numerical_correction'])
            self.assertTrue(full['loaded_faces']); self.assertEqual(len(full['excitation']['sources']), 2)
            self.assertTrue(all(d['energy_balance_within_1_percent'] for d in full['controls']))


if __name__ == '__main__': unittest.main()
