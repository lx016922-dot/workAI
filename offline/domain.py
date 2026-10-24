"""Bounded inputs for the recovered graph network."""
import copy
import hashlib
import json
from pathlib import Path
from solid.schema import default_scene, validate_scene

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'model'
PARTS = ('full', 'linear', 'stvk', 'geometric', 'material')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_info():
    info = json.loads((MODEL / 'model.json').read_text(encoding='utf-8'))
    if digest(MODEL / 'weights.npz') != info['weights_sha256']:
        raise ValueError('模型校验失败，请重新解压完整软件包。')
    return info


def templates():
    result = {}
    for kind in ('plate', 'pipe', 'hole', 'notch'):
        s = default_scene(kind)
        s['geometry'].update(length_mm=10., width_mm=3.5, thickness_mm=1.8,
                             outer_radius_mm=2.5, inner_radius_mm=1.5,
                             hole_x_mm=5., hole_y_mm=1.75, hole_radius_mm=.55,
                             notch_x_mm=5., notch_width_mm=.8, notch_depth_mm=.45)
        s['source'].update(center_mm=[0., 1.75, .9] if kind != 'pipe' else [0., 0., 0.],
                           delay_us=.2, cycles=1.8)
        s['numerics']['h_mm'] = 2. if kind == 'hole' else 1.4
        for i, rx in enumerate(s['receivers']):
            rx['point_mm'] = [3. if i == 0 else 8., 2. if kind == 'pipe' else 1.75, 0. if kind == 'pipe' else .9]
        result[kind] = s
    return result


def validate_request(body):
    if not isinstance(body, dict) or set(body) != {'scene', 'degree'} or type(body['degree']) is not int or body['degree'] not in (3, 4):
        raise ValueError('需要场景与 3/4 阶图设置。')
    raw = copy.deepcopy(body['scene'])
    h = raw['numerics']['h_mm']
    if type(h) not in (int, float) or not .8 <= h <= 2.:
        raise ValueError('网格尺寸需要在 0.8–2 mm 之间。')
    raw['numerics']['h_mm'] = h / body['degree']
    s = validate_scene(raw)
    s['numerics']['h_mm'] = h
    g, tx, acq = s['geometry'], s['source'], s['acquisition']
    limits = [('长度', g['length_mm'], 8, 12), ('频率', tx['frequency_mhz'], .18, .28),
              ('幅值', tx['amplitude_mpa'], .6, 1.5), ('周期', tx['cycles'], 1.5, 2.1),
              ('延迟', tx['delay_us'], .05, .5), ('相位', tx['phase_deg'], -75, 75)]
    if g['kind'] not in ('plate', 'pipe', 'hole', 'notch'):
        raise ValueError('本模型只支持参数化板、管、孔与缺口。')
    if g['kind'] == 'pipe':
        limits += [('管外半径', g['outer_radius_mm'], 2.2, 2.8),
                   ('管壁厚度', g['outer_radius_mm'] - g['inner_radius_mm'], .85, 1.1)]
    else:
        limits += [('板宽', g['width_mm'], 3, 4), ('板厚', g['thickness_mm'], 1.5, 2.1)]
    for label, value, lo, hi in limits:
        if not lo <= value <= hi:
            raise ValueError(f'{label} 超出恢复版模型范围 {lo}–{hi}。')
    if tx['face'] not in ('xmin', 'xmax') or tx['radius_mm'] != 0 or abs(tx['direction'][0]) < .95:
        raise ValueError('使用完整 X 端面、近轴向牵引。')
    if s['boundary']['fixed_face'] not in ('none', 'xmin', 'xmax'):
        raise ValueError('固定面只支持另一 X 端面。')
    if acq != {'duration_us': 14., 'sample_step_us': .05}:
        raise ValueError('当前模型采用 14 μs 时长、0.05 μs 输出间隔。')
    if g['kind'] != 'hole' and h > 1.4:
        raise ValueError('板、管和缺口的网格尺寸不大于 1.4 mm。')
    return {'scene': s, 'degree': body['degree']}
