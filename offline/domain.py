"""Bounded inputs for the original frozen graph network."""
import copy
import hashlib
import json
import math
from pathlib import Path
from solid.schema import default_scene, validate_scene
from graphwave.signals import specification

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'model'
PARTS = ('full', 'linear', 'stvk', 'geometric', 'material')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_info():
    info = json.loads((MODEL / 'model.json').read_text(encoding='utf-8'))
    if digest(MODEL / 'weights.npz') != info['weights_sha256']:
        raise ValueError('模型校验失败，请重新解压完整软件包。')
    if digest(MODEL / 'best.pt') != info['checkpoint_sha256']:
        raise ValueError('原始检查点校验失败。')
    for name, expected in info['evidence_sha256'].items():
        if digest(MODEL / name) != expected:
            raise ValueError('原始验收记录校验失败：' + name)
    for name, expected in info['runtime_source_sha256'].items():
        if digest(ROOT.parent / name) != expected:
            raise ValueError('推理代码与冻结版本不一致：' + name)
    release = json.loads((MODEL / 'release.json').read_text(encoding='utf-8'))
    base = json.loads((MODEL / 'metrics.json').read_text(encoding='utf-8'))
    refined = json.loads((MODEL / 'refined-metrics.json').read_text(encoding='utf-8'))
    if (release['weights_sha256'] != info['weights_sha256'] or
        release['metrics_sha256'] != digest(MODEL / 'metrics.json') or
        release['refined_metrics_sha256'] != digest(MODEL / 'refined-metrics.json') or
        not base['acceptance']['all_gates_passed'] or not refined['all_gates_passed'] or
        len(base['cases']) != 8 or len(refined['cases']) != 8):
        raise ValueError('原始模型与八例验收记录不一致。')
    extension = MODEL / 'extension.json'
    if extension.exists():
        extra = json.loads(extension.read_text(encoding='utf-8'))
        for name, expected in extra['files_sha256'].items():
            if digest(MODEL / name) != expected:
                raise ValueError('长记录模型文件校验失败：' + name)
        for name, expected in extra['runtime_source_sha256'].items():
            if digest(ROOT.parent / name) != expected:
                raise ValueError('长记录运行代码校验失败：' + name)
        info.update(version=extra['version'], user_notice=extra['user_notice'], long_model=extra)
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
                           delay_us=.2, cycles=5.)
        s['acquisition'] = {'duration_us': 5000., 'sample_step_us': .2}
        s['signal'] = {'kind': 'hann', 'end_frequency_mhz': .28}
        s['numerics']['h_mm'] = 2. if kind == 'hole' else 1.4
        for i, rx in enumerate(s['receivers']):
            rx['point_mm'] = [3. if i == 0 else 8., 2. if kind == 'pipe' else 1.75, 0. if kind == 'pipe' else .9]
        result[kind] = s
    return result


def validate_request(body):
    if not isinstance(body, dict) or set(body) != {'scene', 'degree'} or type(body['degree']) is not int or body['degree'] not in (3, 4, 5):
        raise ValueError('需要场景与 3/4/5 阶图设置。')
    raw = copy.deepcopy(body['scene'])
    signal = raw.pop('signal', {'kind': 'hann'})
    if not isinstance(signal, dict):
        raise ValueError('激励波形配置需要为对象。')
    acquisition = copy.deepcopy(raw['acquisition'])
    if set(acquisition) != {'duration_us', 'sample_step_us'}:
        raise ValueError('采样配置需要时长和采样间隔。')
    duration, sample = acquisition['duration_us'], acquisition['sample_step_us']
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in (duration,sample)):
        raise ValueError('时长和采样间隔需要有限数值。')
    legacy = acquisition == {'duration_us': 14., 'sample_step_us': .05} and signal.get('kind') == 'hann'
    if legacy and body['degree'] == 5:
        raise ValueError('历史 14 μs 记录重现只支持原有 3/4 阶设置；五阶请使用长记录配置。')
    if not legacy:
        if not 5000 <= duration <= 20000 or not .025 <= sample <= .2:
            raise ValueError('长记录时长为 5–20 ms（5000–20000 μs），采样间隔为 0.025–0.2 μs。')
        ratio = duration / sample
        if abs(ratio-round(ratio)) > 1e-7 or round(ratio) > 200000:
            raise ValueError('时长需要为采样间隔的整数倍，最多 200001 个采样点。')
        # Validate unchanged material/geometric rules on a bounded acquisition;
        # restore the separately checked real record length below.
        raw['acquisition'] = {'duration_us': 200., 'sample_step_us': .2}
    h = raw['numerics']['h_mm']
    if type(h) not in (int, float) or not .8 <= h <= 2.5:
        raise ValueError('网格尺寸需要在 0.8–2.5 mm 之间。')
    raw['numerics']['h_mm'] = h / body['degree']
    s = validate_scene(raw)
    s['numerics']['h_mm'] = h
    s['acquisition'] = acquisition
    s['signal'] = signal
    spec = specification(s)
    if sample * 2 * max(spec['start_frequency_mhz'], spec['end_frequency_mhz']) > 1/8 + 1e-12:
        raise ValueError('二次谐波最高频率至少需要 8 个输出采样点/周期。')
    g, tx, acq = s['geometry'], s['source'], s['acquisition']
    limits = [('长度', g['length_mm'], 8, 12), ('频率', tx['frequency_mhz'], .18, .28),
              ('幅值', tx['amplitude_mpa'], .6, 1.5), ('周期', tx['cycles'], 1, 20),
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
            raise ValueError(f'{label} 超出原模型范围 {lo}–{hi}。')
    dx, dy, dz = tx['direction']
    if (tx['face'] not in ('xmin', 'xmax') or tx['radius_mm'] != 0 or dx == 0 or
        abs(dy / dx) > .15 + 1e-10 or abs(dz / dx) > .12 + 1e-10 or
        (dx <= 0 if tx['face'] == 'xmin' else dx >= 0)):
        raise ValueError('使用完整 X 端面、近轴向牵引。')
    if s['constitutive'] != 'murnaghan':
        raise ValueError('原网络使用记录中的 Murnaghan 铝材料及三组控制。')
    if g['kind'] == 'hole' and not (.55 <= g['hole_radius_mm'] <= .8 and
        .44 <= g['hole_x_mm'] / g['length_mm'] <= .56 and .43 <= g['hole_y_mm'] / g['width_mm'] <= .57):
        raise ValueError('孔半径需为 0.55–0.8 mm，孔中心需位于原验证范围。')
    if g['kind'] == 'notch' and not (.44 <= g['notch_x_mm'] / g['length_mm'] <= .57 and
        .7 <= g['notch_width_mm'] <= 1.2 and .22 <= g['notch_depth_mm'] / g['thickness_mm'] <= .4):
        raise ValueError('缺口位置、宽度或深度超出原验证范围。')
    if s['boundary']['fixed_face'] not in ('none', 'xmin', 'xmax'):
        raise ValueError('固定面只支持另一 X 端面。')
    if legacy and g['kind'] != 'hole' and h > 1.4:
        raise ValueError('板、管和缺口的网格尺寸不大于 1.4 mm。')
    return {'scene': s, 'degree': body['degree']}
