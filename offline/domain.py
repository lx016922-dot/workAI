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
# Constitutive controls a computation runs and the responses they give. Each response is one control or the difference of two.
LAWS = ('linear', 'stvk', 'murnaghan')
CONTROL_SETS = {'all': LAWS, 'full_linear': ('linear', 'murnaghan'), 'full': ('murnaghan',)}
RECIPES = {'full': ('murnaghan', None), 'linear': ('linear', None), 'stvk': ('stvk', None), 'geometric': ('stvk', 'linear'), 'material': ('murnaghan', 'stvk'),
           'nonlinear': ('murnaghan', 'linear')}
ALL_PARTS = tuple(RECIPES)


def parts_for(controls):
    """Responses available from the computed controls. All three give the original five (the total nonlinear increment is
    then geometric + material); linear and complete give complete, linear and their difference; complete alone gives complete."""
    if tuple(controls) == LAWS: return PARTS
    return tuple(part for part, (a, b) in RECIPES.items() if a in controls and (b is None or b in controls))


def response(values, part):
    """A response from the controls' values (a mapping control name → array)."""
    a, b = RECIPES[part]
    return values[a] if b is None else values[a] - values[b]


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


V2 = 'workai.hyperelastic.solid.v2'


def validate_v2_request(body, controls=3):
    """Version-2 scenes for the v2 input layer: any planar-face sources, signals, profiles, symmetry and absorbing faces.

    Geometry, material and mesh-size bounds stay those of the offline package; every v2 job runs as a
    chunked record (bounded memory, resumable). Accuracy is governed by the mesh: check it with
    `sweep converge` rather than relying on the original 14 μs acceptance."""
    from graphnet.model import load
    from graphnet.inputs import network_scene
    from physics.scene import sources
    from physics.signals import highest_frequency_mhz
    raw = copy.deepcopy(body['scene'])
    raw.pop('signal', None)  # version-1 form field; v2 signals live in each source
    acquisition = raw.get('acquisition')
    if not isinstance(acquisition, dict) or set(acquisition) != {'duration_us', 'sample_step_us'}:
        raise ValueError('采样配置需要时长和采样间隔。')
    duration, sample = acquisition['duration_us'], acquisition['sample_step_us']
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (duration, sample)) or not 0 < duration <= 20000 or not .01 <= sample <= .2:
        raise ValueError('v2 记录时长需在 20 ms（20000 μs）以内，采样间隔为 0.01–0.2 μs。')
    ratio = duration / sample
    if abs(ratio - round(ratio)) > 1e-7 or round(ratio) > 200000:
        raise ValueError('时长需要为采样间隔的整数倍，最多 200001 个采样点。')
    report = None
    if body.get('mesh') == 'auto':
        # The software chooses the degree and element size (offline/automesh.py); any h_mm in the scene is replaced.
        from .automesh import choose
        if not isinstance(raw.get('numerics'), dict): raise ValueError('场景需要 numerics。')
        s, report = choose(raw, lambda trial, degree: _bounded_network_scene(trial, degree, load(degree, long_record=True)['material']), controls=controls)
        degree = report['degree']
    else:
        h = raw.get('numerics', {}).get('h_mm')
        if type(h) not in (int, float) or not .8 <= h <= 2.5:
            raise ValueError('网格尺寸需要在 0.8–2.5 mm 之间。')
        degree = body['degree']
        s = _bounded_network_scene(raw, degree, load(degree, long_record=True)['material'])
    for src in sources(s):
        if sample * 2 * highest_frequency_mhz(src) > 1 / 8 + 1e-12:
            raise ValueError(f'{src.get("name", "主激励")}：二次谐波最高频率至少需要 8 个输出采样点/周期。')
    if report is None: return {'scene': s, 'degree': degree}
    return {'scene': s, 'degree': degree, 'mesh': 'auto', 'mesh_report': report}


def _bounded_network_scene(raw, degree, material):
    """network_scene for any record length up to 20 ms.

    The core scene rules bound a record to 1000 μs and 2000 intervals. A longer record is checked on the
    longest admissible proxy record with the same physics (a step at which every carrier keeps 10 samples
    per period), then the real record is restored; signal coverage of the real record follows because the
    proxy is never longer than it. Sampling of the real record is checked by the caller."""
    from graphnet.inputs import network_scene
    acquisition = raw['acquisition']; duration, sample = acquisition['duration_us'], acquisition['sample_step_us']
    if duration <= 1000 and round(duration / sample) <= 2000:
        return network_scene(raw, degree, material)
    error = None
    for step in sorted({sample, .01, .02, .05, .1, .2, .25, .5}):
        proxy_duration = min(1000., 2000 * step, duration); proxy_duration = math.floor(proxy_duration / step + 1e-9) * step
        if proxy_duration / step < 20: continue
        probe = copy.deepcopy(raw); probe['acquisition'] = {'duration_us': round(proxy_duration, 9), 'sample_step_us': step}
        try:
            scene = network_scene(probe, degree, material)
        except ValueError as failure:
            error = error or failure; continue
        scene['acquisition'] = dict(acquisition)
        return scene
    raise error


ENGINES = ('network', 'coarse')
RECORDS = ('auto', 'full', 'compact')
TIME_STEPPING = ('auto', 'frozen')


def validate_request(body):
    """A job request: {'scene', 'mesh': 'auto'} (the software chooses the mesh) or {'scene', 'degree'} with the scene's h_mm;
    optional 'engine' and 'coarse_model'. A stored request's 'mesh_report' is recomputed, never trusted.

    Version-2 scenes on the network engine also take 'record' (auto: every node of every sample while that stays within
    2 GiB, otherwise the compact record; full; compact) and 'time_stepping' (auto: `graphnet.timestep` chooses the layer;
    frozen: the trained three-stage layer). Both default to auto. Version-2 scenes on either engine take 'controls':
    all (linear, StVK and complete: five responses; the default), full_linear (complete and linear: three responses) or
    full (the complete response only)."""
    if not isinstance(body, dict) or 'scene' not in body:
        raise ValueError('需要场景。')
    body = {k: v for k, v in body.items() if k != 'mesh_report'}
    if set(body) - {'scene', 'degree', 'mesh', 'engine', 'coarse_model', 'record', 'time_stepping', 'controls'}:
        raise ValueError('请求只能包含 scene、degree、mesh、engine、coarse_model、record、time_stepping、controls。')
    options = {k: body.pop(k) for k in ('record', 'time_stepping', 'controls') if k in body}
    v2 = isinstance(body['scene'], dict) and body['scene'].get('schema') == V2
    if options.get('controls', 'all') not in CONTROL_SETS:
        raise ValueError('计算的控制需为 all（线性、StVK、完整）、full_linear（完整与线性）或 full（仅完整）。')
    if 'controls' in options and not v2:
        raise ValueError('选择计算的控制需要 v2 场景。')
    if options.get('record', 'auto') not in RECORDS:
        raise ValueError('记录方式需为 auto（自动）、full（完整记录）或 compact（精简记录）。')
    if options.get('time_stepping', 'auto') not in TIME_STEPPING:
        raise ValueError('时间层需为 auto（自动）或 frozen（训练时的三级四阶时间层）。')
    if set(options) - {'controls'} and not (body.get('engine', 'network') == 'network' and v2):
        raise ValueError('记录方式和时间层设置只用于 v2 场景的冻结网络引擎。')
    return dict(_validate_request(body, len(CONTROL_SETS[options.get('controls', 'all')])), **options)


def _validate_request(body, controls=3):
    """`controls`: how many constitutive controls will run (the automatic mesh's memory limit depends on it)."""
    auto = body.get('mesh') == 'auto'
    if 'mesh' in body and not auto:
        raise ValueError('mesh 只能为 auto（软件自动选择网格）。')
    if auto:
        body.pop('degree', None)
        if not (isinstance(body['scene'], dict) and body['scene'].get('schema') == V2):
            raise ValueError('自动网格需要 v2 场景。')
    elif type(body.get('degree')) is not int or body['degree'] not in (3, 4, 5):
        raise ValueError('需要场景与 3/4/5 阶图设置（或 mesh: auto）。')
    engine = body.get('engine', 'network')
    if engine not in ENGINES:
        raise ValueError('计算引擎需为 network（冻结网络）或 coarse（粗网格算子）。')
    if 'coarse_model' in body and engine != 'coarse':
        raise ValueError('粗网格模型只用于 coarse 引擎。')
    if engine == 'coarse':
        # The scene is validated as written; the learned model then runs it on its own coarse graph.
        if not (isinstance(body['scene'], dict) and body['scene'].get('schema') == V2):
            raise ValueError('快速引擎只接受 v2 场景。')
        from .coarse_engine import DEFAULT_MODEL, check, model_path
        name = body.get('coarse_model', DEFAULT_MODEL); model_path(name)
        check(body['scene'], name)  # the engine's own limits first, so a refusal names them
        if auto:
            # The fast engine runs on its model's own coarse grid: validate the scene at the accuracy-based size, without the fine-mesh budget.
            from .automesh import DEGREE, H_MAX, H_MIN, accuracy_h
            scene = copy.deepcopy(body['scene'])
            if not isinstance(scene.get('numerics'), dict): raise ValueError('场景需要 numerics。')
            scene['numerics']['h_mm'] = round(min(H_MAX, max(H_MIN, accuracy_h(scene))), 4)
            result = dict(validate_v2_request({'scene': scene, 'degree': DEGREE}), mesh='auto')
        else: result = validate_v2_request({k: v for k, v in body.items() if k in ('scene', 'degree')})
        check(result['scene'], name)
        return dict(result, engine='coarse', coarse_model=name)
    if isinstance(body['scene'], dict) and body['scene'].get('schema') == V2:
        return validate_v2_request({k: v for k, v in body.items() if k in ('scene', 'degree', 'mesh')}, controls)
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
