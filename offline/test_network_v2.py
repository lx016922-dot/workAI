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
    s = templates()['plate']
    s['schema'] = 'workai.hyperelastic.solid.v2'
    s['source'].update(cycles=3.0, signal={'type': 'tone_burst', 'window': 'tukey', 'tukey_alpha': 0.5})
    s['sources'] = [
        {
            'name': 'top patch',
            'face': 'zmax',
            'center_mm': [6.0, 1.75, 1.8],
            'radius_mm': 0.0,
            'direction': [0.0, 0.0, -1.0],
            'amplitude_mpa': 0.8,
            'frequency_mhz': 0.2,
            'cycles': 2.0,
            'delay_us': 2.0,
            'phase_deg': 0.0,
            'signal': {'type': 'tone_burst', 'window': 'hann'},
            'profile': {'type': 'gaussian', 'sigma_mm': 1.0},
        }
    ]
    s['boundaries'] = [
        {'type': 'symmetry', 'face': 'ymax'},
        {'type': 'absorbing', 'face': 'xmax', 'length_mm': 3.0, 'ramp': 'cubic', 'attenuation_db': 60.0},
    ]
    s['acquisition'] = {'duration_us': 20.0, 'sample_step_us': 0.05}
    return s


class NetworkV2Requests(unittest.TestCase):
    def test_validation_accepts_v2_scenes_and_keeps_bounds(self):
        s = v2_plate()
        s['signal'] = {'kind': 'hann', 'end_frequency_mhz': 0.28}
        result = validate_request({'scene': s, 'degree': 3})
        self.assertNotIn('signal', result['scene'])
        self.assertEqual(len(result['scene']['sources']), 1)
        self.assertEqual(validate_request(copy.deepcopy(result))['scene'], result['scene'])
        for mutate in (
            lambda x: x['acquisition'].update(sample_step_us=0.5),
            lambda x: x['acquisition'].update(duration_us=25000.0),
            lambda x: x['numerics'].update(h_mm=3.0),
            lambda x: x['material'].update(l_pa=-1e11),
            lambda x: x['numerics'].update(cfl=0.34),
        ):
            bad = v2_plate()
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_request({'scene': bad, 'degree': 3})
        with self.assertRaises(ValueError):
            validate_request({'scene': v2_plate(), 'degree': 2})
        outside = v2_plate()
        outside['receivers'][0]['point_mm'] = [12.0, 1.75, 0.9]  # the plate is 10 mm long: found before the computation, not during it
        with self.assertRaises(ValueError) as refused:
            validate_request({'scene': outside, 'degree': 3})
        self.assertIn('在板外', str(refused.exception))
        self.assertIn(outside['receivers'][0]['name'], str(refused.exception))

    def test_long_v2_records_are_accepted_up_to_20_ms(self):
        for duration, step in ((5000.0, 0.2), (20000.0, 0.1), (1200.0, 0.05)):
            s = v2_plate()
            s['acquisition'] = {'duration_us': duration, 'sample_step_us': step}
            result = validate_request({'scene': s, 'degree': 4})
            self.assertEqual(result['scene']['acquisition'], {'duration_us': duration, 'sample_step_us': step})
        s = v2_plate()
        s['acquisition'] = {'duration_us': 5000.0, 'sample_step_us': 0.2}
        s['source']['frequency_mhz'] = 0.4
        with self.assertRaises(ValueError):
            validate_request({'scene': s, 'degree': 4})  # 2f needs 8 samples per period
        s = v2_plate()
        s['acquisition'] = {'duration_us': 5000.0, 'sample_step_us': 0.2}
        s['numerics']['h_mm'] = 2.5
        s['source']['frequency_mhz'] = 0.25
        s['sources'][0]['frequency_mhz'] = 0.25
        with self.assertRaises(ValueError):
            validate_request({'scene': s, 'degree': 3})  # mesh screen still applies on long records

    def test_local_job_runs_through_the_v2_input_layer(self):
        request = validate_request({'scene': v2_plate(), 'degree': 3})
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / 'local-test'
            job.mkdir()
            (job / 'request.json').write_text(json.dumps(request), encoding='utf-8')
            (job / 'status.json').write_text(json.dumps({'job_id': job.name, 'phase': 'starting'}), encoding='utf-8')
            done = subprocess.run(
                [sys.executable, '-X', 'utf8', '-m', 'offline.worker', '--job-dir', str(job)], cwd=REPO, capture_output=True, text=True
            )
            status = json.loads((job / 'status.json').read_text(encoding='utf-8'))
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            self.assertEqual(status['phase'], 'completed', status)
            recording = json.loads((job / 'record' / 'recording.json').read_text(encoding='utf-8'))
            self.assertEqual(recording['schema'], 'workai.graphnet.long_record.v1')
            self.assertEqual(recording['input_layer'], 'v2_multi_source')
            self.assertEqual([r['name'] for r in recording['excitation']['sources']], ['primary', 'top patch'])
            self.assertGreater(recording['dissipated_energy_j'][2], 0)
            exports = json.loads((job / 'exports.json').read_text(encoding='utf-8'))
            self.assertEqual(set(exports), {'full', 'linear', 'stvk', 'geometric', 'material'})
            full = json.loads((job / 'full.json').read_text(encoding='utf-8'))
            self.assertEqual(full['input_layer'], 'v2_multi_source')
            self.assertTrue(full['neural_model']['neural_forward_only'])
            self.assertFalse(full['neural_model']['numerical_correction'])
            self.assertTrue(full['loaded_faces'])
            self.assertEqual(len(full['excitation']['sources']), 2)
            self.assertTrue(all(d['energy_balance_within_1_percent'] for d in full['controls']))


def megahertz_plate():
    """A 1 mm plate driven at 1.5 MHz: ten nodes per shear wavelength at 2f need elements of about 0.24 mm."""
    s = v2_plate()
    s['geometry'].update(thickness_mm=1.0)
    s['sources'] = []
    s['source'].update(frequency_mhz=1.5, cycles=5.0, center_mm=[0.0, 1.75, 0.5])
    s['acquisition'] = {'duration_us': 5.0, 'sample_step_us': 0.01}
    return s


class ElementSizes(unittest.TestCase):
    def test_elements_down_to_a_tenth_of_a_millimetre_with_the_automatic_time_layer(self):
        report = validate_request({'scene': megahertz_plate(), 'mesh': 'auto'})['mesh_report']
        self.assertLess(report['h_mm'], 0.3)
        self.assertEqual(report['limited_by'], 'accuracy')
        self.assertGreaterEqual(report['nodes_per_2f_shear_wavelength'], 10.0)
        s = megahertz_plate()
        s['numerics']['h_mm'] = 0.3
        self.assertEqual(validate_request({'scene': copy.deepcopy(s), 'degree': 4})['scene']['numerics']['h_mm'], 0.3)
        for h in (0.05, 2.6):
            bad = megahertz_plate()
            bad['numerics']['h_mm'] = h
            with self.assertRaises(ValueError) as refused:
                validate_request({'scene': bad, 'degree': 4})
            self.assertIn('0.1–2.5 mm', str(refused.exception))

    def test_the_trained_time_layer_and_the_fast_engine_keep_their_validated_sizes(self):
        s = megahertz_plate()
        s['numerics']['h_mm'] = 0.3
        for extra in ({'time_stepping': 'frozen'}, {'engine': 'coarse'}):
            with self.assertRaises(ValueError) as refused:
                validate_request({'scene': copy.deepcopy(s), 'degree': 4, **extra})
            self.assertIn('0.8–2.5 mm', str(refused.exception))
            self.assertIn('自动时间层', str(refused.exception))
        # the automatic mesh stops at 0.8 mm for the trained layer, and the refusal says what to change
        with self.assertRaises(ValueError) as refused:
            validate_request({'scene': megahertz_plate(), 'mesh': 'auto', 'time_stepping': 'frozen'})
        self.assertIn('0.8–2.5 mm', str(refused.exception))
        self.assertIn('时间层改为“自动”', str(refused.exception))
        # a scene inside the old range chooses the same mesh with either layer
        sizes = [
            validate_request({'scene': v2_plate(), 'mesh': 'auto', **extra})['mesh_report']['h_mm']
            for extra in ({}, {'time_stepping': 'frozen'})
        ]
        self.assertEqual(sizes[0], sizes[1])
        self.assertGreaterEqual(sizes[0], 0.8)

    def test_a_model_within_five_per_cent_of_its_target_keeps_the_mesh_of_earlier_versions(self):
        def report(frequency, **extra):
            s = v2_plate()
            s['sources'] = []
            s['source'].update(frequency_mhz=frequency, cycles=5.0, signal={'type': 'tone_burst', 'window': 'hann'})
            s['acquisition'] = {'duration_us': 20.0, 'sample_step_us': 0.025}
            return validate_request({'scene': s, 'mesh': 'auto', **extra})['mesh_report']

        near, old = report(0.6), report(0.6, time_stepping='frozen')  # 9.6 nodes per 2f wavelength at 0.8 mm for a target of 10
        self.assertEqual((near['h_mm'], near['limited_by'], near['nodes']), (0.8, 'near_target', old['nodes']))
        self.assertEqual(old['limited_by'], 'h_min')
        self.assertGreaterEqual(near['nodes_per_2f_shear_wavelength'], 0.95 * near['target'])
        self.assertIn('不再加密', near['reason'])
        far, old = report(0.8), report(0.8, time_stepping='frozen')  # 7.2 at 0.8 mm: refined to the target
        self.assertEqual(far['limited_by'], 'accuracy')
        self.assertLess(far['h_mm'], 0.6)
        self.assertGreaterEqual(far['nodes_per_2f_shear_wavelength'], 10.0)
        self.assertEqual(old['h_mm'], 0.8)
        self.assertLess(old['nodes_per_2f_shear_wavelength'], 8.0)


if __name__ == '__main__':
    unittest.main()
