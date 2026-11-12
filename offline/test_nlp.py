"""Text box: Chinese and English sentences → defect, size and excitation edits (offline/nlp.py)."""
import unittest
from solid.schema import default_scene
from physics.scene import validate_scene
from offline.nlp import interpret


def plate(L=200.):
    s = default_scene('plate'); s['schema'] = 'workai.hyperelastic.solid.v2'; s['geometry'].update(length_mm=L, width_mm=2., thickness_mm=1.6)
    s['source'].update(center_mm=[0., 1., .8]); s['receivers'] = [{'name': 'P', 'point_mm': [30., 1., .8], 'direction': [1., 0., 0.]}]
    return s


class Interpret(unittest.TestCase):
    def defects(self, text, scene=None):
        r = interpret(text, scene or plate()); return r, r['scene'].get('defects') or []

    def test_defects_from_chinese_sentences(self):
        r, d = self.defects('在100mm处添加裂纹'); self.assertTrue(r['changed'])
        self.assertEqual(d, [{'type': 'crack', 'x_mm': 100., 'depth_mm': .5, 'from': 'top', 'closed': False}]); self.assertIn('深度未指定', r['operations'][0]['defaults'][0])
        _, d = self.defects('在 120 mm 处加一条闭合裂纹，深 0.4 mm'); self.assertEqual((d[0]['x_mm'], d[0]['depth_mm'], d[0]['closed']), (120., .4, True))
        _, d = self.defects('在30、60 mm处各加一个直径1mm的通孔'); self.assertEqual([(x['type'], x['x_mm'], x['radius_mm']) for x in d], [('hole', 30., .5), ('hole', 60., .5)])
        _, d = self.defects('在60到80mm之间添加分层，距上表面0.8mm'); self.assertEqual((d[0]['type'], d[0]['x_start_mm'], d[0]['x_end_mm'], d[0]['depth_mm']), ('delamination', 60., 80., .8))
        _, d = self.defects('在 150mm 处添加宽 1mm 深 0.5mm 的缺口'); self.assertEqual((d[0]['type'], d[0]['width_mm'], d[0]['depth_mm']), ('notch', 1., .5))
        _, d = self.defects('在 40 mm 处加一个距上表面 0.6 mm、直径 0.6 mm 的横孔'); self.assertEqual((d[0]['axis'], d[0]['z_mm'], d[0]['radius_mm']), ('y', 1., .3))
        _, d = self.defects('在 50mm 处添加内部裂纹，高 0.6mm'); self.assertEqual((d[0]['from'], d[0]['depth_mm'], d[0]['z_mm']), ('inside', .6, .8))
        _, d = self.defects('在中间添加一个接触界面'); self.assertEqual(d, [{'type': 'contact', 'orientation': 'vertical', 'x_mm': 100.}])
        _, d = self.defects('距右端 20 mm 处添加裂纹，深度 0.3 毫米'); self.assertEqual((d[0]['x_mm'], d[0]['depth_mm']), (180., .3))
        for text in ('在100mm处添加裂纹；在 30、60 mm 处各加一个直径 1 mm 的通孔；在 120 到 140 mm 之间添加闭合分层，距上表面 0.8 mm',):
            r = interpret(text, plate()); validate_scene(r['scene']); self.assertEqual(len(r['scene']['defects']), 4)

    def test_english_sizes_settings_removal_and_refusals(self):
        _, d = self.defects('add a closed crack at x = 40 mm, depth 0.4 mm'); self.assertEqual((d[0]['x_mm'], d[0]['depth_mm'], d[0]['closed']), (40., .4, True))
        _, d = self.defects('add a hole of diameter 1 mm at 90 mm'); self.assertEqual((d[0]['x_mm'], d[0]['radius_mm']), (90., .5))
        _, d = self.defects('add a delamination from 20 to 35 mm at depth 0.5 mm'); self.assertEqual((d[0]['x_start_mm'], d[0]['x_end_mm'], d[0]['depth_mm']), (20., 35., .5))
        r = interpret('板长改为 300 mm；频率 300 kHz；记录时长 1 ms；周期数 3', plate()); s = r['scene']
        self.assertEqual((s['geometry']['length_mm'], s['source']['frequency_mhz'], s['acquisition']['duration_us'], s['source']['cycles']), (300., .3, 1000., 3.))
        base = interpret('在 50 mm 处添加裂纹；在 80 mm 处添加直径 1 mm 的通孔', plate())['scene']
        self.assertEqual([x['type'] for x in interpret('删除所有孔', base)['scene']['defects']], ['crack'])
        self.assertNotIn('defects', {k: v for k, v in interpret('删除所有缺陷', base)['scene'].items() if v})
        moved = interpret('把裂纹移到 170 mm', base)['scene']['defects']; self.assertEqual(moved[0]['x_mm'], 170.)
        r = interpret('在 50.1 mm 处添加缺口', base); self.assertFalse(r['changed']); self.assertIn('重叠', r['operations'][0]['summary']); self.assertEqual(r['scene'], base)
        r = interpret('在 300 mm 处添加裂纹', plate()); self.assertFalse(r['changed']); self.assertIn('板长改为', r['operations'][0]['summary'])
        r = interpret('今天天气不错', plate()); self.assertFalse(r['changed']); self.assertEqual(r['unparsed'], ['今天天气不错'])
        with self.assertRaises(ValueError): interpret('', plate())


if __name__ == '__main__': unittest.main()
