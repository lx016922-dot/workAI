"""Wave-field record: the request, the size before the computation, the record against the complete snapshots, the export."""

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from . import wavefield
from .domain import validate_request
from .preview import preview
from .test_network_v2 import v2_plate


def plate():
    s = v2_plate()
    s['acquisition'] = {'duration_us': 20.0, 'sample_step_us': 0.1}
    return s


class Requests(unittest.TestCase):
    def test_a_request_carries_the_setting_and_refuses_what_cannot_be(self):
        self.assertEqual(wavefield.validate({}), {'region': 'top', 'components': 'z', 'every': 1})
        self.assertIsNone(wavefield.validate(None))
        checked = validate_request({'scene': plate(), 'degree': 3, 'wavefield': {'region': 'bottom', 'components': 'xyz', 'every': 4}})
        self.assertEqual(checked['wavefield'], {'region': 'bottom', 'components': 'xyz', 'every': 4})
        self.assertEqual(validate_request(json.loads(json.dumps(checked)))['wavefield'], checked['wavefield'])
        self.assertNotIn('wavefield', validate_request({'scene': plate(), 'degree': 3, 'wavefield': None}))
        for bad in ({'region': 'side'}, {'components': 'x'}, {'every': 0}, {'every': 1.5}, {'every': 2000}, {'pitch': 1}, 'top'):
            with self.assertRaises(ValueError):
                validate_request({'scene': plate(), 'degree': 3, 'wavefield': bad})
        with self.assertRaisesRegex(ValueError, '完整记录已经保存'):
            validate_request({'scene': plate(), 'degree': 3, 'record': 'full', 'wavefield': {}})
        with self.assertRaisesRegex(ValueError, '冻结网络'):
            validate_request({'scene': plate(), 'mesh': 'auto', 'engine': 'coarse', 'wavefield': {}})
        pipe = plate()
        pipe['geometry']['kind'] = 'pipe'
        pipe['source'].update(face='xmin', center_mm=[0.0, 0.0, 0.0], direction=[1.0, 0.0, 0.0])
        pipe['receivers'] = [{'name': 'P', 'point_mm': [3.0, pipe['geometry']['outer_radius_mm'] - 0.2, 0.0], 'direction': [1.0, 0.0, 0.0]}]
        try:
            validate_request({'scene': pipe, 'degree': 3, 'wavefield': {'region': 'all'}})
        except ValueError as error:  # the pipe of this fixture may be refused for its own reasons, never for the wave field
            self.assertNotIn('波场', str(error))
        else:
            with self.assertRaisesRegex(ValueError, '圆管没有上、下表面'):
                validate_request({'scene': pipe, 'degree': 3, 'wavefield': {'region': 'top'}})

    def test_the_nodes_of_a_region_and_the_size(self):
        x, y, z = np.meshgrid(np.linspace(0, 0.01, 5), np.linspace(0, 0.004, 3), np.linspace(0, 0.002, 4), indexing='ij')
        points = np.stack([x.ravel(), y.ravel(), z.ravel()], 1)
        top = wavefield.node_ids(points, 'top')
        self.assertEqual(len(top), 15)
        self.assertTrue(np.allclose(points[top, 2], 0.002))
        self.assertTrue(np.allclose(points[wavefield.node_ids(points, 'bottom'), 2], 0.0))
        self.assertEqual(len(wavefield.node_ids(points, 'all')), 60)
        self.assertTrue((np.diff(top) > 0).all())
        spec = wavefield.validate({'every': 4})
        self.assertEqual(wavefield.frames_kept(5001, 4), 1251)
        self.assertEqual(wavefield.size_bytes(5001, 1000, spec, 2), 1251 * 1000 * 4 * 2)
        self.assertEqual(wavefield.size_bytes(5001, 1000, wavefield.validate({'components': 'xyz'}), 1), 5001 * 1000 * 3 * 4)
        gx, gy, index = wavefield.grid(points[top] * 1e3)
        self.assertEqual((len(gx), len(gy), index.shape), (5, 3, (3, 5)))
        self.assertTrue(np.allclose(points[top][index][:, :, 0] * 1e3, gx[None]))
        self.assertIsNone(wavefield.grid(points * 1e3))
        self.assertIsNone(wavefield.grid(np.r_[points[top], points[top][:1]] * 1e3))  # a node split by a defect: not a full grid

    def test_the_preview_gives_the_size(self):
        out = preview({'scene': plate(), 'degree': 3, 'wavefield': {'region': 'top', 'components': 'z', 'every': 2}})
        self.assertTrue(out['ok'], out['errors'])
        wave = out['estimate']['wavefield']
        self.assertEqual(out['estimate']['record'], 'compact')
        self.assertEqual(wave['bytes'], wavefield.frames_kept(201, 2) * wave['nodes'] * 4 * wave['parts'])
        self.assertIn('波场记录：上表面', wave['text'])
        self.assertNotIn('wavefield', preview({'scene': plate(), 'degree': 3})['estimate'])


class Recorded(unittest.TestCase):
    def test_the_record_equals_the_complete_snapshots_and_exports(self):
        from scipy.io import loadmat
        from . import worker
        from .recording import chunks, manifest, traces

        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / ('local-' + 'a' * 32)
            job.mkdir()
            request = validate_request(
                {'scene': plate(), 'degree': 3, 'controls': 'full_linear', 'wavefield': {'region': 'top', 'components': 'xyz', 'every': 2}}
            )
            (job / 'request.json').write_text(json.dumps(request), encoding='utf-8')
            (job / 'status.json').write_text(json.dumps({'job_id': job.name, 'phase': 'starting'}), encoding='utf-8')
            worker.run(job)
            status = json.loads((job / 'status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['phase'], 'completed', status.get('message'))
            record = job / 'record'
            m = manifest(record)
            self.assertEqual(m['record_mode'], 'compact')
            entry = wavefield.info(record)
            self.assertEqual(
                (entry['components'], entry['every'], entry['parts'], entry['region']), ([0, 1, 2], 2, ['linear', 'nonlinear'], 'top')
            )
            self.assertEqual(entry['frames'], 101)
            self.assertTrue(np.allclose(entry['points_mm'][:, 2], plate()['geometry']['thickness_mm']))
            times, linear = wavefield.read(record, 'linear')
            _, increment = wavefield.read(record, 'nonlinear')
            self.assertEqual(linear.shape, (101, entry['nodes'], 3))
            self.assertEqual(linear.dtype, np.float32)
            self.assertTrue(np.allclose(times, np.arange(101) * 0.2))
            self.assertGreater(np.abs(linear).max(), 0)
            # the complete float64 fields the compact record keeps at a few samples: the wave field is those values on its nodes
            controls = m['controls']
            checked = 0
            for _, data in chunks(record, m):
                for at, frame in enumerate(data['snapshot_frames']):
                    if frame % 2:
                        continue
                    whole = dict(zip(controls, data['snapshots'][:, at]))
                    k = frame // 2
                    self.assertTrue(np.array_equal(linear[k], whole['linear'][entry['node_ids']].astype(np.float32)))
                    self.assertTrue(
                        np.array_equal(increment[k], (whole['murnaghan'] - whole['linear'])[entry['node_ids']].astype(np.float32))
                    )
                    checked += 1
            self.assertGreater(checked, 0)
            with self.assertRaisesRegex(ValueError, '没有“geometric”'):
                wavefield.read(record, 'geometric')
            # the complete response is the sum of what is stored
            self.assertEqual(wavefield.responses(entry['parts']), ['full', 'linear', 'nonlinear'])
            self.assertTrue(np.array_equal(wavefield.read(record, 'full')[1], linear + increment))
            self.assertEqual(
                wavefield.responses(['linear', 'geometric', 'material']), ['full', 'linear', 'stvk', 'nonlinear', 'geometric', 'material']
            )
            self.assertEqual(wavefield.responses(['full']), ['full'])
            # a point of the face that is no receiver: the nearest recorded node answers
            node = entry['points_mm'][len(entry['points_mm']) // 3]
            got = traces(record, 'linear', [[float(node[0]) + 1e-4, float(node[1]), float(node[2])]])
            self.assertEqual(got['points'][0]['location']['wavefield_node'], len(entry['points_mm']) // 3)
            self.assertTrue(np.allclose(np.array(got['points'][0]['xyz_m']), linear[:, len(entry['points_mm']) // 3]))
            self.assertEqual(len(got['times_us']), 101)
            with self.assertRaisesRegex(ValueError, '精简记录'):
                traces(record, 'linear', [[float(node[0]), float(node[1]), 0.3]])  # inside the plate: not recorded
            # exports
            for kind in ('npz', 'mat'):
                wavefield.export(record, 'linear', kind, job / f'field.{kind}')
            with np.load(job / 'field.npz') as z:
                self.assertTrue(np.array_equal(z['u_m'], linear))
                self.assertEqual(str(z['components']), 'xyz')
                self.assertTrue(np.array_equal(z['u_m'][:, z['grid_index']].shape[1:3], (len(z['y_mm']), len(z['x_mm']))))
                self.assertTrue(np.allclose(z['points_mm'][z['grid_index']][:, :, 0], z['x_mm'][None]))
            mat = loadmat(job / 'field.mat')
            self.assertTrue(np.array_equal(mat['u_m'], linear))
            self.assertEqual(mat['grid_index'].min(), 1)


if __name__ == '__main__':
    unittest.main()
