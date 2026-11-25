"""Post-processing for nonlinear guided waves (offline/nonlinear.py) and the convergence check (offline/scan.py), on a
computed plane-strain plate with an S0 tone burst: the measured wavenumbers and group velocity against the theory, the
growth of A2/A1², the f–k spectrum, and the same model on a finer mesh."""

import base64
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np
from . import nonlinear
from .domain import templates
from .scan import Busy, case_request, convergence_definition


def s0_plate():
    """A 1 mm aluminium plate in plane strain (symmetry faces in y), 0.2 MHz S0 burst from the x = 0 face, absorbing far end."""
    s = templates()['plate']
    s['schema'] = 'workai.hyperelastic.solid.v2'
    s.pop('signal', None)
    s['name'] = 'S0 板'
    s['geometry'].update(length_mm=90.0, width_mm=2.5, thickness_mm=1.0)
    s['source'].update(
        face='xmin',
        center_mm=[0.0, 1.25, 0.5],
        radius_mm=0.0,
        direction=[1.0, 0.0, 0.0],
        amplitude_mpa=5.0,
        frequency_mhz=0.2,
        cycles=3.0,
        delay_us=0.0,
        phase_deg=0.0,
        signal={'type': 'tone_burst', 'window': 'hann'},
    )
    s['sources'] = []
    s['boundary'] = {'fixed_face': 'none'}
    s['boundaries'] = [
        {'type': 'symmetry', 'face': 'ymin'},
        {'type': 'symmetry', 'face': 'ymax'},
        {'type': 'absorbing', 'face': 'xmax', 'length_mm': 25.0, 'ramp': 'cubic', 'attenuation_db': 60.0},
    ]
    s['acquisition'] = {'duration_us': 32.0, 'sample_step_us': 0.05}
    s['receivers'] = [{'name': f'P{i + 1}', 'point_mm': [10.0 + 10.0 * i, 1.25, 1.0], 'direction': [1.0, 0.0, 0.0]} for i in range(5)]
    return s


class Checks(unittest.TestCase):
    def test_contact_is_said_to_be_missing_from_the_increment(self):
        closed = {'defects': [{'type': 'crack', 'x_mm': 10.0, 'depth_mm': 0.5, 'closed': True}]}
        self.assertIn('不含接触产生的谐波', nonlinear.contact_note(closed, 'nonlinear'))
        self.assertIsNone(nonlinear.contact_note(closed, 'full'))  # the complete response contains them
        for scene in (
            {},
            {'defects': []},
            {'defects': [{'type': 'crack', 'x_mm': 10.0, 'depth_mm': 0.5, 'closed': False}, {'type': 'hole'}]},
        ):
            self.assertIsNone(nonlinear.contact_note(scene, 'nonlinear'))
        for defect in ({'type': 'contact', 'orientation': 'vertical'}, {'type': 'delamination', 'closed': True}):
            self.assertIsNotNone(nonlinear.contact_note({'defects': [defect]}, 'nonlinear'))

    def test_a_convergence_check_raises_the_degree_first_then_refines(self):
        request = {'scene': s0_plate(), 'mesh': 'auto', 'mesh_report': {'degree': 4, 'h_mm': 2.303}, 'controls': 'full_linear'}
        scan = convergence_definition(request, 'S0 板', 'local-' + '0' * 32, 2)
        self.assertEqual([(l['degree'], l['h_mm']) for l in scan['levels']], [(4, 2.303), (5, 2.303), (5, 1.7272)])
        self.assertEqual(scan['axes'][0]['values'], [0.57575, 0.4606, 0.34544])
        self.assertNotIn('mesh', scan['request'])
        self.assertEqual((scan['request']['degree'], scan['request']['controls'], scan['kind']), (4, 'full_linear', 'convergence'))
        finer = case_request(scan, [0.34544])
        self.assertEqual((finer['degree'], finer['scene']['numerics']['h_mm']), (5, 1.7272))
        self.assertTrue(finer['scene']['name'].endswith(' · 5 阶 · 单元 1.7272 mm'))
        self.assertEqual((request['mesh'], request['mesh_report']['h_mm']), ('auto', 2.303))  # the job's own request is untouched
        explicit = convergence_definition({'scene': s0_plate(), 'degree': 5}, 'x', 'local-' + '0' * 32, 1)
        self.assertEqual([(l['degree'], round(l['h_mm'], 4)) for l in explicit['levels']], [(5, 1.4), (5, 1.05)])
        for body, text in (
            ({'scene': s0_plate(), 'mesh': 'auto', 'mesh_report': {'degree': 4, 'h_mm': 2.0}, 'engine': 'coarse'}, '快速引擎'),
            ({'scene': s0_plate(), 'mesh': 'auto'}, '报告'),
            ({'scene': dict(s0_plate(), schema='workai.hyperelastic.solid.v1'), 'degree': 4}, 'v2'),
        ):
            with self.assertRaises(ValueError) as refused:
                convergence_definition(body, 'x', 'local-' + '0' * 32, 1)
            self.assertIn(text, str(refused.exception))
        with self.assertRaises(ValueError):
            convergence_definition(request, 'x', 'local-' + '0' * 32, 3)

    def test_fits_arrivals_and_wrapped_phases(self):
        fit = nonlinear.linear_fit([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0])
        self.assertAlmostEqual(fit['slope'], 2.0)
        self.assertAlmostEqual(fit['intercept'], 1.0)
        self.assertAlmostEqual(fit['r_squared'], 1.0)
        self.assertIsNone(nonlinear.linear_fit([0.0, 1.0], [1.0, 2.0])['r_squared'])
        self.assertIsNone(nonlinear.linear_fit([0.0], [1.0]))
        # two packets: the first one, half as large as a later overlap, is the arrival
        t = np.arange(0.0, 100.0, 0.05)
        burst = lambda centre, size: size * np.exp(-(((t - centre) / 4.0) ** 2)) * np.cos(2 * np.pi * 0.5 * (t - centre))
        traces = np.stack([burst(20.0, 1.0) + burst(70.0, 1.8), burst(30.0, 1.0) + burst(60.0, 1.8)], axis=1)
        np.testing.assert_allclose(nonlinear.arrivals(t, nonlinear.envelope(traces), 2.0), [20.0, 30.0], atol=0.3)
        # points 20 mm apart sample a 27 mm wave less than twice per wavelength: without a plate only the range that is determined can be said
        k = 231.4
        x = np.arange(5) * 0.02
        phase = np.unwrap(np.angle(np.exp(-1j * k * x)))
        notes = []
        found = nonlinear.wavenumber(x, phase, 0.2e6, None, 'f', notes)
        self.assertAlmostEqual(found['unambiguous_below_rad_mm'], np.pi / 20.0)
        self.assertGreater(0.2314, found['unambiguous_below_rad_mm'])
        self.assertIsNone(found['theory'])
        self.assertIsNone(found['phase_velocity_m_s'])
        self.assertAlmostEqual(found['wavenumber_rad_mm'], 0.2314 - 2 * np.pi / 20.0, places=4)  # the principal value: one turn short
        from .guided import plate_of

        notes = []
        found = nonlinear.wavenumber(
            x, phase, 0.2e6, plate_of(s0_plate()), 'f', notes
        )  # with the plate, the one multiple of 2π/Δx that lands on a mode
        self.assertEqual(
            (found['alias_turns'], found['theory']['mode'], found['theory']['matched'], found['reliable']), (1, 'S0', True, True)
        )
        self.assertAlmostEqual(found['wavenumber_rad_mm'], 0.2314, places=3)
        self.assertIn('S0', notes[0])
        # a second harmonic that follows the fundamental (wavenumber 2k) and is on no symmetric mode: it is set against 2k(f), and SH or antisymmetric modes are not candidates
        thick = s0_plate()
        thick['geometry']['thickness_mm'] = 2.0
        thick['source']['frequency_mhz'] = 0.5
        plate = plate_of(thick)
        x = np.arange(60) * 0.002
        forced = 2 * 588.7
        notes = []
        found = nonlinear.wavenumber(x, -forced * x, 1e6, plate, '2f', notes, secondary=True, forced=forced)
        self.assertEqual((found['theory']['mode'], found['theory']['matched'], notes), ('S0', False, []))
        self.assertAlmostEqual(found['forced']['difference'], 0.0, places=9)
        self.assertAlmostEqual(found['theory']['difference'], -0.113, places=2)


class Computed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from .app import Application

        cls.tmp = tempfile.TemporaryDirectory()
        cls.app = Application(Path(cls.tmp.name) / 'data')
        cls.job = cls.app.submit({'scene': s0_plate(), 'mesh': 'auto', 'record': 'full'})['job_id']
        while cls.app.busy():
            time.sleep(0.5)
        assert cls.app.status(cls.job)['phase'] == 'completed', cls.app.status(cls.job)
        cls.path = cls.app.path(cls.job)

    @classmethod
    def tearDownClass(cls):
        cls.app.close()
        cls.tmp.cleanup()

    def test_the_second_harmonic_grows_along_the_plate_and_the_waves_are_the_theoretical_s0(self):
        a = nonlinear.accumulation(
            self.path, {'points': {'mode': 'line', 'start_mm': [8.0, 1.25, 1.0], 'end_mm': [50.0, 1.25, 1.0]}, 'direction': [1.0, 0.0, 0.0]}
        )
        self.assertGreater(len(a['rows']), 40)
        self.assertTrue(all(r['usable'] for r in a['rows']))
        self.assertEqual((a['a2_basis'], a['window']['mode'], a['window']['speed_source']), ('nonlinear', 'follow', 'measured'))
        fit = a['fit']
        self.assertEqual((fit['f']['theory']['mode'], fit['2f']['theory']['mode']), ('S0', 'S0'))
        self.assertLess(abs(fit['f']['theory']['difference']), 2e-3)
        self.assertLess(abs(fit['2f']['theory']['difference']), 1.5e-2)
        self.assertEqual(fit['f']['alias_turns'], 0)
        self.assertTrue(fit['f']['theory']['matched'])
        self.assertLess(abs(fit['2f']['forced']['difference']), 1.5e-2)
        self.assertEqual((fit['pair']['primary'], fit['pair']['secondary']), ('S0', 'S0'))
        self.assertLess(abs(fit['pair']['mismatch']), 5e-3)
        self.assertGreater(fit['pair']['coherence_length_mm'], 1000.0)
        self.assertAlmostEqual(a['group']['velocity_m_s'] / fit['f']['theory']['group_velocity_m_s'], 1.0, delta=0.02)
        self.assertGreater(fit['ratio']['slope_per_m2'], 0)
        self.assertGreater(fit['ratio']['r_squared'], 0.99)
        self.assertGreater(fit['a2']['r_squared'], 0.99)
        self.assertLess(fit['a1_variation'], 0.05)
        ratios = [r['ratio_per_m'] for r in a['rows']]
        self.assertGreater(ratios[-1], 3 * ratios[0])
        self.assertTrue(all(r['a2_geometric_m'] > 0 and r['a2_material_m'] > 0 and r['a2_m'] > 0 for r in a['rows']))
        self.assertEqual(a['notes'], [])
        # the receivers of the computation give the same line, 10 mm apart; a fit range keeps the points inside it
        r = nonlinear.accumulation(self.path, {'points': {'mode': 'receivers'}, 'fit_mm': [0.0, 30.0]})
        self.assertEqual(
            ([row['name'] for row in r['rows']], r['fit']['points'], r['fit']['range_mm']), (['P1', 'P2', 'P3', 'P4', 'P5'], 4, [0.0, 30.0])
        )
        self.assertAlmostEqual(r['fit']['ratio']['slope_per_m2'] / fit['ratio']['slope_per_m2'], 1.0, delta=0.05)
        self.assertEqual(r['fit']['f']['theory']['mode'], 'S0')
        # the same window everywhere, and what is refused
        fixed = nonlinear.accumulation(
            self.path, {'points': {'mode': 'receivers'}, 'window': {'mode': 'fixed', 'start_us': 0.0, 'end_us': 32.0}}
        )
        self.assertEqual(fixed['window'], {'mode': 'fixed', 'start_us': 0.0, 'end_us': 32.0})
        self.assertTrue(all(row['window_us'] == [0.0, 32.0] for row in fixed['rows']))
        late = nonlinear.accumulation(self.path, {'points': {'mode': 'receivers'}, 'window': {'mode': 'follow', 'length_us': 40.0}})
        self.assertEqual([row['usable'] for row in late['rows']], [True, True, False, False, False])
        self.assertIn('末尾', late['notes'][-1])
        for body, text in (
            ({'points': {'mode': 'line', 'start_mm': [8.0, 1.25, 1.0], 'end_mm': [8.0, 1.25, 1.0]}}, '重合'),
            ({'points': {'mode': 'line', 'start_mm': [8.0, 1.25, 1.0], 'end_mm': [50.0, 1.25, 3.0]}}, '不在材料内'),
            ({'points': {'mode': 'line', 'start_mm': [8.0, 1.25, 1.0], 'end_mm': [50.0, 1.25, 1.0], 'count': 1}}, '采样点数'),
            ({'points': {'mode': 'grid'}}, '点列'),
            ({'direction': [0.0, 0.0, 0.0]}, '零向量'),
            ({'window': {'mode': 'fixed', 'start_us': 5.0, 'end_us': 5.0}}, '时间窗'),
            ({'fit_mm': [40.0, 10.0]}, '拟合范围'),
            ({'window': {'mode': 'follow', 'speed_m_s': 1.0}}, '跟随速度'),
            ({'extra': 1}, '只接受'),
        ):
            with self.assertRaises(ValueError) as refused:
                nonlinear.accumulation(self.path, body)
            self.assertIn(text, str(refused.exception), body)

    def test_the_fk_spectrum_peaks_on_the_s0_branch(self):
        line = {'mode': 'line', 'start_mm': [5.0, 1.25, 1.0], 'end_mm': [60.0, 1.25, 1.0], 'count': 64}
        k = nonlinear.fk(self.path, {'points': line, 'direction': [1.0, 0.0, 0.0]})
        image = np.frombuffer(base64.b64decode(k['image']), dtype=np.uint8).reshape(k['frequency']['count'], k['wavenumber']['count'])
        i, j = np.unravel_index(int(image.argmax()), image.shape)
        f = k['frequency']['first_mhz'] + i * k['frequency']['step_mhz']
        wavenumber = k['wavenumber']['first_rad_mm'] + j * k['wavenumber']['step_rad_mm']
        self.assertEqual(int(image.max()), 255)
        self.assertAlmostEqual(f, 0.2, delta=0.03)
        self.assertAlmostEqual(2 * np.pi * f / wavenumber, 5.43, delta=0.6)  # km/s: the S0 speed, to the resolution of a 55 mm line
        self.assertEqual({c['name'] for c in k['curves']}, {'S0', 'A0', 'SH0'})
        self.assertTrue(k['validity']['plane_strain'])
        self.assertEqual((k['points'], k['part'], k['samples']), (64, 'full', 641))
        self.assertAlmostEqual(k['wavenumber']['nyquist_rad_mm'], np.pi / k['spacing_mm'])
        self.assertLess(abs(k['wavenumber']['first_rad_mm'] + (k['wavenumber']['count'] - 1) / 2 * k['wavenumber']['step_rad_mm']), 1e-9)
        part = nonlinear.fk(self.path, {'points': line, 'part': 'nonlinear', 'range_us': [5.0, 30.0], 'frequency_max_mhz': 0.5})
        self.assertEqual((part['part'], part['range_us'], part['samples']), ('nonlinear', [5.0, 30.0], 501))
        for body, text in (
            ({'points': {'mode': 'receivers'}}, '8 个'),
            ({'points': line, 'part': 'none'}, '没有这种响应'),
            ({'points': line, 'range_us': [3.0, 3.5]}, '32 个'),
            ({'points': line, 'frequency_max_mhz': 50.0}, 'Nyquist'),
        ):
            with self.assertRaises(ValueError) as refused:
                nonlinear.fk(self.path, body)
            self.assertIn(text, str(refused.exception), body)
        d = nonlinear.dispersion(self.path)
        self.assertEqual([m['name'] for m in d['modes']['f']], ['A0', 'SH0', 'S0'])
        with self.assertRaises(ValueError):
            nonlinear.dispersion(self.path, 720.0)

    def test_the_same_model_on_a_finer_mesh_gives_the_same_harmonics(self):
        app, job = self.app, self.job
        plan = app.scans.converge_plan(job)
        self.assertTrue(plan['ok'])
        self.assertEqual((plan['base']['degree'], plan['cases'][0]['degree'], plan['existing']), (4, 5, []))
        self.assertGreater(plan['cases'][0]['nodes'], plan['base']['nodes'])
        self.assertGreater(plan['seconds_total'], 0)
        self.assertGreater(plan['disk_bytes_total'], 0)
        started = app.scans.converge(job)
        with self.assertRaises(Busy):
            app.scans.converge(job)
        with self.assertRaises(Busy):
            app.submit({'scene': s0_plate(), 'mesh': 'auto'})
        early = app.scans.convergence(started['scan_id'])
        self.assertEqual((len(early['rows']), early['verdict'], [p['degree'] for p in early['pending']]), (1, None, [5]))
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            scan = app.scans.detail(started['scan_id'])
            if scan['state'] != 'running' and not app.busy():
                break
            time.sleep(0.5)
        self.assertEqual(
            (scan['state'], scan['kind'], scan['base_job'], [c['phase'] for c in scan['cases']], scan['cases'][0]['job_id']),
            ('completed', 'convergence', job, ['completed', 'completed'], job),
        )
        self.assertEqual([c['label'] for c in scan['cases']], [level['label'] for level in scan['levels']])
        self.assertEqual(scan['cases'][0]['attempts'], 0)
        refined = app.status(scan['cases'][1]['job_id'])
        self.assertTrue(refined['name'].endswith('· 5 阶 · 单元 ' + f'{scan["levels"][1]["h_mm"]:g} mm'))
        self.assertEqual(refined['scan']['index'], 1)
        c = app.scans.convergence(started['scan_id'])
        base, fine = c['rows']
        self.assertEqual((base['degree'], fine['degree'], fine['is_reference'], c['pending'], c['a2_basis']), (4, 5, True, [], 'nonlinear'))
        self.assertGreater(fine['nodes'], base['nodes'])
        self.assertTrue(c['verdict']['converged'])
        self.assertIn('已收敛', c['verdict']['text'])
        self.assertEqual(len(base['receivers']), 5)
        self.assertLess(base['worst']['a1_change'], 1e-3)
        self.assertLess(base['worst']['a2_change'], 1e-2)
        self.assertLess(base['worst']['waveform_change'], 1e-2)
        self.assertGreater(base['worst']['waveform_change'], 0)
        self.assertEqual(fine['worst']['a1_change'], 0.0)
        self.assertEqual(app.scans.convergence(started['scan_id']), c)  # the second time from the stored comparison
        windowed = app.scans.convergence(started['scan_id'], [5.0, 25.0])
        self.assertEqual(windowed['window_us'], [5.0, 25.0])
        self.assertNotEqual(windowed['rows'][0]['receivers'][0]['a1_m'], base['receivers'][0]['a1_m'])
        self.assertEqual([s['scan_id'] for s in app.scans.checks_of(job)], [started['scan_id']])
        self.assertEqual(app.scans.checks_of(scan['cases'][1]['job_id']), [])
        table = app.scans.summary(started['scan_id'])
        self.assertEqual(
            [row['label'] for row in table['rows']], [level['label'] for level in scan['levels']]
        )  # it is also an ordinary scan
        with self.assertRaises(ValueError):
            app.scans.convergence(started['scan_id'], [5.0, 5.0])


if __name__ == '__main__':
    unittest.main()
