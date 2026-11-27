"""Defects in version-2 plate scenes: validation, normalization, placement and mesh primitives."""

import copy
import unittest
from solid.schema import default_scene
from physics.scene import validate_scene
from physics.defects import validate_defects, default_defect, footprint, primitives


def plate(L=40.0, W=2.0, T=1.6):
    s = default_scene('plate')
    s['schema'] = 'workai.hyperelastic.solid.v2'
    s['geometry'].update(length_mm=L, width_mm=W, thickness_mm=T)
    s['source'].update(center_mm=[0.0, W / 2, T / 2])
    s['receivers'] = [{'name': 'P', 'point_mm': [30.0, W / 2, T / 2], 'direction': [1.0, 0.0, 0.0]}]
    return s


class Defects(unittest.TestCase):
    def test_every_type_validates_normalizes_and_revalidates_unchanged(self):
        s = plate()
        s['defects'] = [
            {'type': 'crack', 'x_mm': 5.0, 'depth_mm': 0.5},
            {'type': 'crack', 'x_mm': 8.0, 'depth_mm': 0.6, 'from': 'inside', 'z_mm': 0.8, 'closed': True},
            {'type': 'delamination', 'x_start_mm': 10.0, 'x_end_mm': 15.0, 'depth_mm': 0.6},
            {'type': 'contact', 'orientation': 'vertical', 'x_mm': 18.0},
            {'type': 'contact', 'orientation': 'horizontal', 'x_start_mm': 20.0, 'x_end_mm': 24.0, 'depth_mm': 1.0},
            {'type': 'notch', 'x_mm': 27.0, 'width_mm': 0.8, 'depth_mm': 0.4},
            {'type': 'hole', 'x_mm': 31.0, 'y_mm': 1.0, 'radius_mm': 0.5},
            {'type': 'hole', 'axis': 'y', 'x_mm': 36.0, 'z_mm': 0.8, 'radius_mm': 0.4},
        ]
        v = validate_scene(s)
        d = v['defects']
        self.assertEqual([x['type'] for x in d], ['crack', 'crack', 'delamination', 'contact', 'contact', 'notch', 'hole', 'hole'])
        self.assertEqual(d[0]['from'], 'top')
        self.assertFalse(d[0]['closed'])
        self.assertEqual(d[0]['y_range_mm'], [0.0, 2.0])
        self.assertEqual(d[0]['name'], '裂纹 1')
        self.assertEqual(validate_scene(copy.deepcopy(v)), v)
        p = primitives(d, v)
        self.assertEqual([s['axis'] for s in p['splits']], [0, 0, 2, 0, 2])
        self.assertEqual([s['law'] for s in p['splits']], ['open', 'contact', 'open', 'contact', 'contact'])
        self.assertEqual(p['splits'][0]['ranges_mm'][2], [1.1, 1.6])
        self.assertEqual(len(p['voids']), 1)
        self.assertEqual([h['axis'] for h in p['holes']], [2, 1])
        self.assertNotIn('defects', validate_scene(plate()))

    def test_bad_defects_are_refused_with_a_reason(self):
        cases = [
            ({'type': 'crack', 'x_mm': 10.0, 'depth_mm': 1.6}, '深度'),
            ({'type': 'hole', 'x_mm': 10.0, 'y_mm': 1.0, 'radius_mm': 1.0}, '板宽'),
            ({'type': 'crack', 'x_mm': 50.0, 'depth_mm': 0.5}, '位置'),
            ({'type': 'void', 'x_mm': 1.0}, '类型'),
            ({'type': 'crack', 'x_mm': 10.0, 'depth_mm': 0.5, 'size': 1}, 'size'),
            ({'type': 'delamination', 'x_start_mm': 12.0, 'x_end_mm': 10.0, 'depth_mm': 0.5}, '终点'),
            ({'type': 'contact', 'orientation': 'diagonal'}, '方向'),
        ]
        for defect, word in cases:
            s = plate()
            s['defects'] = [defect]
            with self.assertRaises(ValueError) as caught:
                validate_scene(s)
            self.assertIn(word, str(caught.exception), defect)
        s = plate()
        s['defects'] = [{'type': 'crack', 'x_mm': 10.0, 'depth_mm': 0.5}, {'type': 'notch', 'x_mm': 10.3, 'width_mm': 0.4, 'depth_mm': 0.3}]
        with self.assertRaises(ValueError) as caught:
            validate_scene(s)
        self.assertIn('重叠', str(caught.exception))
        pipe = plate()
        pipe['geometry']['kind'] = 'pipe'
        pipe['defects'] = [{'type': 'crack', 'x_mm': 10.0, 'depth_mm': 0.5}]
        with self.assertRaises(ValueError) as caught:
            validate_defects(pipe['defects'], pipe)
        self.assertIn('平板', str(caught.exception))

    def test_a_delamination_is_a_strip_or_a_circle(self):
        from physics.defects import primitives

        s = plate(W=30.0)
        circle = {
            'type': 'delamination',
            'shape': 'circle',
            'x_mm': 20.0,
            'y_mm': 12.0,
            'diameter_mm': 8.0,
            'depth_mm': 0.5,
            'closed': True,
        }
        rows = validate_defects(
            [circle, {'type': 'delamination', 'shape': 'rectangle', 'x_start_mm': 30.0, 'x_end_mm': 34.0, 'depth_mm': 0.5}], s
        )
        self.assertEqual(rows[0], dict(circle, name='分层 1'))
        self.assertNotIn('shape', rows[1])  # the strip is written as it always was
        self.assertEqual(rows[1]['y_range_mm'], [0.0, 30.0])
        self.assertEqual(footprint(rows[0], s), [[16.0, 24.0], [8.0, 16.0], [1.1, 1.1]])
        split = primitives(rows, s)['splits'][0]
        self.assertEqual((split['axis'], split['law'], split['circle_mm']), (2, 'contact', {'centre': [20.0, 12.0], 'radius': 4.0}))
        self.assertNotIn('circle_mm', primitives(rows, s)['splits'][1])
        self.assertEqual(
            validate_defects([{k: v for k, v in circle.items() if k != 'y_mm'}], s)[0]['y_mm'], 15.0
        )  # the middle of the width
        for change, word in (
            ({'y_mm': 3.0}, '完整位于板内'),
            ({'x_mm': 3.9}, '完整位于板内'),
            ({'diameter_mm': 0.5}, '直径'),
            ({'x_start_mm': 10.0}, '不用 x_start_mm'),
            ({'y_range_mm': [0.0, 5.0]}, '不用 y_range_mm'),
            ({'shape': 'oval'}, '形状'),
        ):
            with self.assertRaises(ValueError) as caught:
                validate_defects([dict(circle, **change)], s)
            self.assertIn(word, str(caught.exception), change)
        # the front mesh of a crack, a delamination or a contact interface: standard is not written, coarse is
        self.assertNotIn('front_mesh', validate_defects([dict(circle, front_mesh='standard')], s)[0])
        coarse = validate_defects(
            [dict(circle, front_mesh='coarse'), {'type': 'crack', 'x_mm': 34.0, 'depth_mm': 0.5, 'front_mesh': 'coarse'}], s
        )
        self.assertEqual([d['front_mesh'] for d in coarse], ['coarse', 'coarse'])
        self.assertEqual([p['front_fraction'] for p in primitives(coarse, s)['splits']], [0.5, 0.5])
        self.assertNotIn('front_fraction', split)
        with self.assertRaisesRegex(ValueError, '前沿网格'):
            validate_defects([dict(circle, front_mesh='fine')], s)
        with self.assertRaisesRegex(ValueError, '不认识的字段 front_mesh'):
            validate_defects([{'type': 'hole', 'x_mm': 34.0, 'y_mm': 15.0, 'radius_mm': 1.0, 'front_mesh': 'coarse'}], s)
        with self.assertRaisesRegex(ValueError, '属于圆形分层'):
            validate_defects([{'type': 'delamination', 'x_start_mm': 30.0, 'x_end_mm': 34.0, 'depth_mm': 0.5, 'diameter_mm': 4.0}], s)
        with self.assertRaisesRegex(ValueError, '缺少 x_start_mm'):
            validate_defects([{'type': 'delamination', 'depth_mm': 0.5}], s)
        with self.assertRaisesRegex(ValueError, '重叠'):
            validate_defects([circle, {'type': 'crack', 'x_mm': 22.0, 'depth_mm': 0.6}], s)  # the crack runs across the whole width

    def test_one_click_defaults_are_valid_and_keep_clear_of_existing_defects(self):
        s = plate()
        existing = []
        for kind in ('crack', 'hole', 'delamination', 'contact', 'notch', 'crack', 'crack'):
            d = default_defect(kind, s, existing)
            existing.append(d)
            validate_defects(existing, s)  # raises on overlap
        boxes = [footprint(x, s) for x in validate_defects(existing, s)]
        self.assertEqual(len({tuple(map(tuple, b)) for b in boxes}), len(boxes))
        crowded = plate(L=4.0)
        items = []
        with self.assertRaises(ValueError) as caught:
            for _ in range(10):
                items.append(default_defect('delamination', crowded, items))
        self.assertIn('空位', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
