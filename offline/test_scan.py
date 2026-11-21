"""Parameter scans of the workbench: the grid, the checks before a scan, the queue (failures, retry, pause, restart) and the summary table."""

import copy
import csv
import io
import json
import math
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np
from .scan import MAX_CASES, Busy, case_request, combinations, harmonics, label, plan, validate_scan
from .test_network_v2 import v2_plate


def model():
    """The workbench template with one source, as the interface would send it."""
    s = v2_plate()
    s['sources'] = []
    s['name'] = '扫描测试'
    return {'scene': s, 'mesh': 'auto'}


def wait(app, scan_id, done=lambda scan: scan['state'] != 'running', limit_s=900):
    deadline = time.monotonic() + limit_s
    while time.monotonic() < deadline:
        scan = app.scans.detail(scan_id)
        if done(scan) and not app.busy():
            return scan
        time.sleep(0.5)
    raise AssertionError(app.scans.detail(scan_id))


class Definition(unittest.TestCase):
    def test_the_grid_is_the_product_of_the_axes_in_order(self):
        scan = validate_scan(
            {
                'request': model(),
                'axes': [{'path': 'source.frequency_mhz', 'values': [0.2, 0.25]}, {'path': 'source.amplitude_mpa', 'values': [1, 2, 4]}],
            }
        )
        rows = combinations(scan)
        self.assertEqual(rows, [[0.2, 1.0], [0.2, 2.0], [0.2, 4.0], [0.25, 1.0], [0.25, 2.0], [0.25, 4.0]])
        request = case_request(scan, rows[4])
        self.assertEqual((request['scene']['source']['frequency_mhz'], request['scene']['source']['amplitude_mpa']), (0.25, 2.0))
        self.assertEqual(label(scan['axes'], rows[4]), 'frequency_mhz=0.25, amplitude_mpa=2')
        self.assertTrue(request['scene']['name'].endswith(' · frequency_mhz=0.25, amplitude_mpa=2'))
        self.assertEqual(
            scan['request']['scene']['source']['amplitude_mpa'], model()['scene']['source']['amplitude_mpa']
        )  # the scan's own request is untouched
        self.assertEqual(request['mesh'], 'auto')
        self.assertLessEqual(len(request['scene']['name']), 100)
        named = validate_scan(
            {
                'request': model(),
                'axes': [
                    {'path': 'source.frequency_mhz', 'values': [0.2, 0.25], 'label': '频率', 'unit': 'MHz'},
                    {'path': 'source.cycles', 'values': [3.0], 'label': '周期数'},
                ],
            }
        )
        self.assertEqual(label(named['axes'], [0.25, 3.0]), '频率 0.25 MHz，周期数 3')
        self.assertTrue(case_request(named, [0.25, 3.0])['scene']['name'].endswith(' · 频率 0.25 MHz，周期数 3'))
        from .scan import heading

        self.assertEqual([heading(a) for a in named['axes']], ['频率 (MHz)', '周期数'])
        self.assertEqual(heading(scan['axes'][0]), 'source.frequency_mhz')
        with self.assertRaises(ValueError):
            validate_scan({'request': model(), 'axes': [{'path': 'source.cycles', 'values': [3.0], 'label': 'x' * 61}]})

    def test_what_is_not_a_scan_is_refused_with_the_reason(self):
        ok = [{'path': 'source.amplitude_mpa', 'values': [1.0, 2.0]}]
        for body, text in (
            ({'request': model()}, '1–3 个参数'),
            ({'request': model(), 'axes': []}, '1–3 个参数'),
            ({'request': model(), 'axes': ok * 2}, '重复'),
            ({'request': model(), 'axes': [{'path': 'source.signal.type', 'values': [1.0]}]}, '不是数值'),
            ({'request': model(), 'axes': [{'path': 'source.missing_mpa', 'values': [1.0]}]}, '没有参数'),
            ({'request': model(), 'axes': [{'path': 'name', 'values': [1.0]}]}, 'path'),
            ({'request': model(), 'axes': [{'path': 'numerics.cfl', 'values': [0.2]}]}, 'path'),
            ({'request': model(), 'axes': [{'path': 'source.amplitude_mpa', 'values': [1.0, 1.0]}]}, '重复'),
            ({'request': model(), 'axes': [{'path': 'source.amplitude_mpa', 'values': [1.0, float('nan')]}]}, '有限数值'),
            (
                {
                    'request': model(),
                    'axes': [
                        {'path': 'source.amplitude_mpa', 'values': list(range(1, 16))},
                        {'path': 'source.frequency_mhz', 'values': [0.2 + 0.001 * i for i in range(14)]},
                    ],
                },
                f'超过上限 {MAX_CASES}',
            ),
            ({'request': {'scene': {'schema': 'workai.hyperelastic.solid.v1'}}, 'axes': ok}, 'v2'),
            ({'request': model(), 'axes': ok, 'extra': 1}, 'request'),
        ):
            with self.assertRaises(ValueError) as refused:
                validate_scan(body)
            self.assertIn(text, str(refused.exception), body.get('axes'))

    def test_what_sits_on_a_moving_face_moves_with_it(self):
        body = model()
        s = body['scene']
        W, T = s['geometry']['width_mm'], s['geometry']['thickness_mm']
        s['source'].update(face='zmax', center_mm=[4.0, W / 2, T], direction=[0.0, 0.0, -1.0])
        s['receivers'] = [
            {'name': 'top', 'point_mm': [7.0, W / 2, T], 'direction': [0.0, 0.0, 1.0]},
            {'name': 'mid', 'point_mm': [7.0, 1.0, T / 2], 'direction': [1.0, 0.0, 0.0]},
            {'name': 'inside', 'point_mm': [7.0, 1.0, 0.4], 'direction': [1.0, 0.0, 0.0]},
        ]
        s['defects'] = [{'type': 'crack', 'x_mm': 5.0, 'depth_mm': 0.5, 'y_range_mm': [0.0, W]}]
        scan = validate_scan(
            {'request': body, 'axes': [{'path': 'geometry.thickness_mm', 'values': [1.2]}, {'path': 'geometry.width_mm', 'values': [5.0]}]}
        )
        moved = case_request(scan, [1.2, 5.0])['scene']
        self.assertEqual(moved['source']['center_mm'], [4.0, 2.5, 1.2])
        self.assertEqual([r['point_mm'] for r in moved['receivers']], [[7.0, 2.5, 1.2], [7.0, 1.0, 0.6], [7.0, 1.0, 0.4]])
        self.assertEqual(moved['defects'][0]['y_range_mm'], [0.0, 5.0])
        self.assertEqual(moved['defects'][0]['depth_mm'], 0.5)
        self.assertEqual(scan['request']['scene']['source']['center_mm'], [4.0, W / 2, T])

    def test_the_plan_checks_every_combination_and_adds_up_the_cost(self):
        planned = plan(
            {
                'request': model(),
                'axes': [{'path': 'source.frequency_mhz', 'values': [0.2, 0.3]}, {'path': 'source.amplitude_mpa', 'values': [1.0, 2.0]}],
            }
        )
        self.assertEqual((planned['count'], planned['valid']), (4, 4))
        self.assertTrue(all(c['ok'] and c['nodes'] > 0 and c['seconds'] > 0 for c in planned['cases']))
        self.assertAlmostEqual(planned['seconds_total'], sum(c['seconds'] for c in planned['cases']))
        self.assertEqual(planned['memory_bytes_max'], max(c['memory_bytes'] for c in planned['cases']))
        self.assertEqual(planned['cases'][1]['label'], 'frequency_mhz=0.2, amplitude_mpa=2')
        # the disk estimate is the record and the result files (the animation of five responses), and it is compared with the free space
        case = planned['cases'][0]
        self.assertEqual((case['record'], planned['full_records'], planned['free_bytes'], planned['disk_short']), ('full', 4, None, False))
        frames = 401
        self.assertGreater(case['disk_bytes'], frames * case['nodes'] * 3 * 8 * 3)
        self.assertLess(case['disk_bytes'], 2.2 * frames * case['nodes'] * 3 * 8 * 3)
        with tempfile.TemporaryDirectory() as tmp:
            here = plan({'request': model(), 'axes': [{'path': 'source.frequency_mhz', 'values': [0.2]}]}, tmp)
            self.assertGreater(here['free_bytes'], 0)
            self.assertEqual(here['disk_short'], here['disk_bytes_total'] > here['free_bytes'])
        # 3 MHz needs more output samples per period than the record has: that combination is marked, the others stay valid
        planned = plan({'request': model(), 'axes': [{'path': 'source.frequency_mhz', 'values': [0.2, 3.0]}]})
        self.assertEqual((planned['count'], planned['valid']), (2, 1))
        self.assertFalse(planned['cases'][1]['ok'])
        self.assertTrue(planned['cases'][1]['errors'])
        self.assertEqual(planned['seconds_total'], planned['cases'][0]['seconds'])

    def test_harmonics_are_the_windowed_coefficients_at_f_and_2f(self):
        t = np.arange(4001) * 0.01
        f = 0.5
        v = 3e-9 * np.sin(2 * math.pi * f * t + 0.4) + 2e-11 * np.cos(2 * math.pi * 2 * f * t - 1.0) + 5e-9
        first, second = harmonics(t, v, f)
        self.assertAlmostEqual(first['amplitude_m'] / 3e-9, 1.0, places=6)
        self.assertAlmostEqual(second['amplitude_m'] / 2e-11, 1.0, places=4)
        self.assertAlmostEqual(first['phase_deg'], math.degrees(0.4) - 90.0, places=4)
        self.assertAlmostEqual(second['phase_deg'], math.degrees(-1.0), places=2)
        self.assertEqual(harmonics(t, v, 60.0), [None, None])
        self.assertIsNone(harmonics(t, v, 30.0)[1])
        self.assertIsNotNone(harmonics(t, v, 30.0)[0])  # Nyquist is 50 MHz
        windowed = harmonics(t, v, f, [10.0, 30.0])
        self.assertAlmostEqual(windowed[0]['amplitude_m'] / 3e-9, 1.0, places=5)
        with self.assertRaises(ValueError):
            harmonics(t, v, f, [1.0, 1.05])


class Queue(unittest.TestCase):
    def test_a_scan_runs_its_combinations_keeps_going_after_a_failure_and_reruns_it(self):
        from .app import Application

        with tempfile.TemporaryDirectory() as tmp:
            app = Application(Path(tmp) / 'data')
            try:
                with self.assertRaises(ValueError) as refused:
                    app.scans.create({'request': model(), 'axes': [{'path': 'source.amplitude_mpa', 'values': [1.0, 50.0]}]})
                self.assertIn('第 2 个组合（amplitude_mpa=50）不能计算', str(refused.exception))
                self.assertEqual(app.scans.list(), [])  # checked before anything starts
                # 1 and 2 MPa with a hole between the receivers, and the hole moved onto the second receiver: that combination passes the checks and fails when it is computed
                body = model()
                body['scene']['defects'] = [{'type': 'hole', 'x_mm': 5.5, 'y_mm': 1.75, 'radius_mm': 0.4}]
                body = {
                    'request': body,
                    'axes': [{'path': 'defects.0.x_mm', 'values': [5.5, 8.0]}, {'path': 'source.amplitude_mpa', 'values': [1.0, 2.0]}],
                    'name': '孔位置与幅值',
                }
                started = app.scans.create(body)
                self.assertEqual(started['count'], 4)
                with self.assertRaises(Busy):
                    app.scans.create(body)  # one scan at a time
                with self.assertRaises(Busy):
                    app.submit(model())  # and no single run in between its combinations
                scan = wait(app, started['scan_id'])
                self.assertEqual(
                    (scan['state'], [c['phase'] for c in scan['cases']]), ('completed', ['completed', 'completed', 'failed', 'failed'])
                )
                self.assertIn('2 个组合未完成', scan['message'])
                self.assertTrue(scan['cases'][2]['message'])
                self.assertEqual(scan['counts'], {'queued': 0, 'running': 0, 'completed': 2, 'failed': 2, 'paused': 0})
                jobs = [app.status(c['job_id']) for c in scan['cases']]
                self.assertEqual([j['scan'] for j in jobs], [{'id': scan['scan_id'], 'index': i, 'count': 4} for i in range(4)])
                self.assertTrue(jobs[1]['name'].endswith('x_mm=5.5, amplitude_mpa=2'))
                table = app.scans.summary(scan['scan_id'])
                a, b, bad = table['rows'][:3]
                self.assertEqual((len(a['receivers']), a['receivers'][0]['a2_basis'], bad['phase']), (2, 'nonlinear', 'failed'))
                self.assertNotIn('receivers', bad)
                for ra, rb in zip(a['receivers'], b['receivers']):
                    self.assertAlmostEqual(rb['a1_m'] / ra['a1_m'], 2.0, places=3)  # the linear part doubles
                    self.assertAlmostEqual(rb['a2_nonlinear_m'] / ra['a2_nonlinear_m'], 4.0, places=2)  # the nonlinear increment quadruples
                    self.assertAlmostEqual(rb['a2_over_a1_squared_per_m'] / ra['a2_over_a1_squared_per_m'], 1.0, places=2)
                    self.assertGreater(ra['a2_geometric_m'], 0)
                    self.assertGreater(ra['a2_material_m'], 0)
                    self.assertLess(ra['a2_nonlinear_m'], ra['a1_m'])
                self.assertEqual(app.scans.summary(scan['scan_id']), table)  # the second time from the stored rows
                windowed = app.scans.summary(scan['scan_id'], [2.0, 18.0])
                self.assertEqual(windowed['window_us'], [2.0, 18.0])
                self.assertNotEqual(windowed['rows'][0]['receivers'][0]['a1_m'], a['receivers'][0]['a1_m'])
                rows = list(csv.reader(io.StringIO(app.scans.summary_csv(scan['scan_id']).decode('utf-8-sig'))))
                self.assertEqual(len(rows), 1 + 2 + 2 + 1 + 1)
                self.assertEqual(rows[0][:4], ['组合', 'defects.0.x_mm', 'source.amplitude_mpa', '状态'])
                self.assertEqual(len({len(r) for r in rows}), 1)
                self.assertEqual(float(rows[3][rows[0].index('A1 完整 (m)')]), b['receivers'][0]['a1_m'])
                self.assertEqual((rows[1][3], rows[5][3]), ('已完成', '未完成'))
                # retry: the failed combinations run again as new jobs (and fail again, for the same reason)
                old = scan['cases'][2]['job_id']
                app.scans.control(scan['scan_id'], 'retry')
                scan = wait(app, scan['scan_id'])
                self.assertEqual(
                    (scan['cases'][2]['phase'], scan['cases'][2]['attempts'], scan['cases'][2]['previous_jobs']), ('failed', 2, [old])
                )
                self.assertNotEqual(scan['cases'][2]['job_id'], old)
                self.assertEqual([c['attempts'] for c in scan['cases']], [1, 1, 2, 2])
                with self.assertRaises(ValueError):
                    app.scans.control(scan['scan_id'], 'resume')  # nothing is paused
                self.assertEqual([s['scan_id'] for s in app.scans.list()], [scan['scan_id']])
                self.assertNotIn('cases', app.scans.list()[0])
            finally:
                app.close()

    def test_a_paused_scan_waits_and_a_scan_left_running_is_paused_at_the_next_start(self):
        from .app import Application

        with tempfile.TemporaryDirectory() as tmp:
            app = Application(Path(tmp) / 'data')
            try:
                started = app.scans.create({'request': model(), 'axes': [{'path': 'source.cycles', 'values': [2.0, 3.0]}]})
                app.scans.control(started['scan_id'], 'pause')
                with self.assertRaises(ValueError):
                    app.scans.control(started['scan_id'], 'pause')
                scan = wait(app, started['scan_id'], done=lambda s: s['cases'][0]['phase'] in ('completed', 'paused'))
                self.assertEqual((scan['state'], scan['cases'][1]['phase'], scan['cases'][1]['job_id']), ('paused', 'queued', None))
                self.assertFalse(app.scans.active())
                single = app.submit(model())['job_id']  # a paused scan leaves the queue free
                while app.busy():
                    time.sleep(0.3)
                self.assertEqual(app.status(single)['phase'], 'completed')
                self.assertEqual(app.scans.detail(started['scan_id'])['state'], 'paused')
                # as if the application had been closed during the scan
                stored = app.scans.load(started['scan_id'])
                stored['state'] = 'running'
                app.scans.save(stored)
            finally:
                app.close()
            app = Application(Path(tmp) / 'data')
            try:
                scan = app.scans.detail(started['scan_id'])
                self.assertEqual(scan['state'], 'paused')
                self.assertIn('上次退出', scan['message'])
                app.scans.control(started['scan_id'], 'resume')
                scan = wait(app, started['scan_id'])
                self.assertEqual(
                    (scan['state'], [c['phase'] for c in scan['cases']], scan['message']),
                    ('completed', ['completed', 'completed'], '全部组合已完成。'),
                )
            finally:
                app.close()


if __name__ == '__main__':
    unittest.main()
