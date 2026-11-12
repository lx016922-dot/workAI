"""Defects in version-2 plate scenes: validation, normalization, placement and mesh primitives."""
import copy
import unittest
from solid.schema import default_scene
from physics.scene import validate_scene
from physics.defects import validate_defects, default_defect, footprint, primitives


def plate(L=40., W=2., T=1.6):
    s = default_scene('plate'); s['schema'] = 'workai.hyperelastic.solid.v2'
    s['geometry'].update(length_mm=L, width_mm=W, thickness_mm=T); s['source'].update(center_mm=[0., W / 2, T / 2])
    s['receivers'] = [{'name': 'P', 'point_mm': [30., W / 2, T / 2], 'direction': [1., 0., 0.]}]
    return s


class Defects(unittest.TestCase):
    def test_every_type_validates_normalizes_and_revalidates_unchanged(self):
        s = plate(); s['defects'] = [
            {'type': 'crack', 'x_mm': 5., 'depth_mm': .5}, {'type': 'crack', 'x_mm': 8., 'depth_mm': .6, 'from': 'inside', 'z_mm': .8, 'closed': True},
            {'type': 'delamination', 'x_start_mm': 10., 'x_end_mm': 15., 'depth_mm': .6}, {'type': 'contact', 'orientation': 'vertical', 'x_mm': 18.},
            {'type': 'contact', 'orientation': 'horizontal', 'x_start_mm': 20., 'x_end_mm': 24., 'depth_mm': 1.}, {'type': 'notch', 'x_mm': 27., 'width_mm': .8, 'depth_mm': .4},
            {'type': 'hole', 'x_mm': 31., 'y_mm': 1., 'radius_mm': .5}, {'type': 'hole', 'axis': 'y', 'x_mm': 36., 'z_mm': .8, 'radius_mm': .4}]
        v = validate_scene(s); d = v['defects']
        self.assertEqual([x['type'] for x in d], ['crack', 'crack', 'delamination', 'contact', 'contact', 'notch', 'hole', 'hole'])
        self.assertEqual(d[0]['from'], 'top'); self.assertFalse(d[0]['closed']); self.assertEqual(d[0]['y_range_mm'], [0., 2.]); self.assertEqual(d[0]['name'], '裂纹 1')
        self.assertEqual(validate_scene(copy.deepcopy(v)), v)
        p = primitives(d, v)
        self.assertEqual([s['axis'] for s in p['splits']], [0, 0, 2, 0, 2]); self.assertEqual([s['law'] for s in p['splits']], ['open', 'contact', 'open', 'contact', 'contact'])
        self.assertEqual(p['splits'][0]['ranges_mm'][2], [1.1, 1.6]); self.assertEqual(len(p['voids']), 1); self.assertEqual([h['axis'] for h in p['holes']], [2, 1])
        self.assertNotIn('defects', validate_scene(plate()))

    def test_bad_defects_are_refused_with_a_reason(self):
        cases = [({'type': 'crack', 'x_mm': 10., 'depth_mm': 1.6}, '深度'), ({'type': 'hole', 'x_mm': 10., 'y_mm': 1., 'radius_mm': 1.}, '板宽'),
                 ({'type': 'crack', 'x_mm': 50., 'depth_mm': .5}, '位置'), ({'type': 'void', 'x_mm': 1.}, '类型'), ({'type': 'crack', 'x_mm': 10., 'depth_mm': .5, 'size': 1}, 'size'),
                 ({'type': 'delamination', 'x_start_mm': 12., 'x_end_mm': 10., 'depth_mm': .5}, '终点'), ({'type': 'contact', 'orientation': 'diagonal'}, '方向')]
        for defect, word in cases:
            s = plate(); s['defects'] = [defect]
            with self.assertRaises(ValueError) as caught: validate_scene(s)
            self.assertIn(word, str(caught.exception), defect)
        s = plate(); s['defects'] = [{'type': 'crack', 'x_mm': 10., 'depth_mm': .5}, {'type': 'notch', 'x_mm': 10.3, 'width_mm': .4, 'depth_mm': .3}]
        with self.assertRaises(ValueError) as caught: validate_scene(s)
        self.assertIn('重叠', str(caught.exception))
        pipe = plate(); pipe['geometry']['kind'] = 'pipe'; pipe['defects'] = [{'type': 'crack', 'x_mm': 10., 'depth_mm': .5}]
        with self.assertRaises(ValueError) as caught: validate_defects(pipe['defects'], pipe)
        self.assertIn('平板', str(caught.exception))

    def test_one_click_defaults_are_valid_and_keep_clear_of_existing_defects(self):
        s = plate(); existing = []
        for kind in ('crack', 'hole', 'delamination', 'contact', 'notch', 'crack', 'crack'):
            d = default_defect(kind, s, existing); existing.append(d)
            validate_defects(existing, s)   # raises on overlap
        boxes = [footprint(x, s) for x in validate_defects(existing, s)]
        self.assertEqual(len({tuple(map(tuple, b)) for b in boxes}), len(boxes))
        crowded = plate(L=4.); items = []
        with self.assertRaises(ValueError) as caught:
            for _ in range(10): items.append(default_defect('delamination', crowded, items))
        self.assertIn('空位', str(caught.exception))


if __name__ == '__main__': unittest.main()
