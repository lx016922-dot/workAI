"""Fast engine: the learned coarse-graph operator (coarsegraph) inside the offline software.

The network is the same frozen graph network (energy layer, state update); only the element operators
(derivative filter, stiffness and mass quadrature) are the learned coarse ones, on the model's own coarse
grid (degree and element size stored in the model). It is a research model: its accuracy is the held-out
test result recorded in the model and, outside the training range, the range verification of research/coarse-range
(`RANGE_VERIFICATION`); anything outside both must be checked case by case.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from solid.schema import require

REPOSITORY = Path(__file__).resolve().parents[1]
MODELS_DIRECTORY = REPOSITORY / 'coarsegraph' / 'models'
# Shipped coarse models (name -> label); the first is the default. The h = 2 mm model has about 1.5 times the nodes and passes
# every held-out top, end and two-patch case (docs/coarse-graph-operator.md 6.2).
MODELS = {'p4-h3-harmonic': '快速 · h 3 mm', 'p4-h2-multisource': '精确 · h 2 mm'}
DEFAULT_MODEL = 'p4-h3-harmonic'
MODEL = MODELS_DIRECTORY / DEFAULT_MODEL / 'model.json'
LAWS = ('linear', 'stvk', 'murnaghan')
# Training range of the shipped model (coarsegraph/specs): 48 × 2 mm half-width aluminium plates, symmetry at y = 2 mm.
TRAINING_RANGE = [('geometry.length_mm', '板长', 48, 48, 'mm'), ('geometry.width_mm', '半宽', 2, 2, 'mm'), ('geometry.thickness_mm', '板厚', 1.6, 2.4, 'mm'),
                  ('acquisition.duration_us', '记录', 0, 30, 'μs')]
# How far outside the training range each model holds (research/coarse-range, python -m coarsegraph.verify): defect-free plates on a
# 1 mm fine reference, the gates taken per group of receivers on its arrival window. `covered`: what the cases spanned, for a model
# that passed there; `groups`/`passed`: receiver groups of the defect-free cases and how many passed every gate.
RANGE_VERIFICATION = {
    'p4-h3-harmonic': {'groups': 43, 'passed': 27, 'covered': {},
                       'text': '训练范围外的 43 组接收点（板长到 200 mm、记录到 100 μs、半宽到 8 mm）中 27 组通过全部门槛：波形差多在 5% 内，二次谐波随传播距离变差；需要二次谐波请换精确模型'},
    'p4-h2-multisource': {'groups': 43, 'passed': 42, 'covered': {'geometry.length_mm': (48, 200), 'geometry.width_mm': (2, 8), 'acquisition.duration_us': (0, 100)},
                          'text': '板长到 200 mm（传播 140 mm）、记录到 100 μs、半宽到 8 mm、全宽、端面与双贴片激励的 43 组接收点中 42 组通过全部门槛（未过的一组增量 2f 5.3%）'}}
# Held-out test of the shipped model on all four label groups (python -m coarsegraph.train reevaluate, splits.json).
EXCITATION_EVALUATION = 'evaluation-abcd.json'
EXCITATIONS = {'labels-a-top-gaussian': '顶面高斯', 'labels-b-end-axial': '端面轴向', 'labels-c-two-patch': '双贴片反相', 'labels-d-comb': '梳状叉指'}
TRAINED_FACES = ('zmax', 'zmin', 'xmin', 'xmax')  # top/bottom (A, C) and end faces (B); the plate is mirror-symmetric


def model_digest(path):
    """SHA-256 of a model file as committed (LF line endings), so a checkout that rewrote them to CRLF still matches its evaluation."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def model_path(name=DEFAULT_MODEL):
    require(name in MODELS, f'粗网格模型需为 {"、".join(MODELS)} 之一。')
    return MODELS_DIRECTORY / name / 'model.json'


def _get(scene, path):
    value = scene
    for key in path.split('.'): value = value.get(key) if isinstance(value, dict) else None
    return value


def by_excitation(path=MODEL):
    """Held-out test results per excitation form, from the model's four-group evaluation (None if it is not there)."""
    file = Path(path).parent / EXCITATION_EVALUATION
    if not file.exists(): return None
    evaluation = json.loads(file.read_text(encoding='utf-8'))
    require(evaluation['model_sha256'] == model_digest(path), f'{EXCITATION_EVALUATION} 不是这个模型的评估。')
    gll = {r['key']: r for r in evaluation['evaluation']['baseline_gll']['rows']}; rows = []
    for group, label in EXCITATIONS.items():
        test = [r for r in evaluation['evaluation']['learned']['rows'] if r['split'] == 'test' and r['key'].split(':')[0] == group]
        if not test: continue
        rows.append({'group': group, 'label': label, 'cases': len(test), 'passed': sum(r['passed'] for r in test), 'baseline_passed': sum(gll[r['key']]['passed'] for r in test),
                     'worst_waveform_l2': max(r['worst_waveform_l2'] for r in test), 'worst_a2': max(r['worst_a2'] for r in test),
                     'worst_increment_a2': max(r['worst_increment_a2'] for r in test)})
    return rows


def model_summary(path=MODEL):
    """What the interface shows about the shipped model; the four-group evaluation, when present, supersedes the training-time test."""
    model = json.loads(Path(path).read_text(encoding='utf-8')); test = model['summary']['learned']['test']; base = model['summary']['baseline_gll']['test']
    forms = by_excitation(path)
    if forms:
        evaluation = json.loads((Path(path).parent / EXCITATION_EVALUATION).read_text(encoding='utf-8'))['summary']
        test = evaluation['learned']['test']; base = evaluation['baseline_gll']['test']
    return {'name': Path(path).parent.name, 'label': MODELS.get(Path(path).parent.name, Path(path).parent.name), 'degree': model['degree'], 'h_mm': model['h_mm'], 'parameters': model['parameter_count'],
            'range_verification': {k: v for k, v in RANGE_VERIFICATION.get(Path(path).parent.name, {}).items() if k != 'covered'},
            'test_cases': test['pairs'], 'test_passed': test['passed'], 'baseline_test_passed': base['passed'],
            'test_worst_waveform_l2': test['worst_waveform_l2'], 'test_worst_increment_a2': test['worst_increment_a2'], 'test_by_excitation': forms or [],
            'coarse_nodes_example': model['cost']['coarse_nodes'], 'model_sha256': model_digest(path)}


def _tested(forms, group):
    row = next((r for r in forms or [] if r['group'] == group), None)
    return f'留出测试 {row["passed"]}/{row["cases"]} 通过' if row else '未经测试'


def out_of_range(scene, model_name=DEFAULT_MODEL):
    """What the scene has outside the range the named model was trained on or verified in (information; the run is still allowed)."""
    out = []; covered = RANGE_VERIFICATION.get(model_name, {}).get('covered', {}); beyond = False
    for path, label, lo, hi, unit in TRAINING_RANGE:
        v = _get(scene, path)
        if not isinstance(v, (int, float)) or lo - 1e-9 <= v <= hi + 1e-9: continue
        trained = f'训练 {lo:g}{"–" + format(hi, "g") if hi != lo else ""} {unit}'
        if path in covered:
            if covered[path][0] - 1e-9 <= v <= covered[path][1] + 1e-9: continue   # verified there (research/coarse-range)
            out.append(f'{label} {v:g} {unit}（{trained}，核验到 {covered[path][1]:g} {unit}）')
        else: out.append(f'{label} {v:g} {unit}（{trained}）'); beyond = beyond or path != 'geometry.thickness_mm'
    if beyond and not covered: out.append('快速模型在训练范围外：核验中波形差多在 5% 内，二次谐波随传播距离变差、多数超过 5%；需要二次谐波请换精确模型 h 2 mm')
    sources = [x for x in [scene.get('source'), *(scene.get('sources') or [])] if isinstance(x, dict)]; forms = by_excitation(model_path(model_name))
    if len(sources) > 1:
        patch = next((r for r in forms or [] if r['group'] == 'labels-c-two-patch'), None)
        if patch and patch['passed'] == patch['cases']: out.append(f'{len(sources)} 个激励源（训练只含同类双贴片反相，{_tested(forms, "labels-c-two-patch")}；其他多源组合需逐例核对）')
        else: out.append(f'{len(sources)} 个激励源（同类双贴片{_tested(forms, "labels-c-two-patch")}，主要是非线性增量的 2f 超过 5%；需要二次谐波时请换精确模型 h 2 mm）')
    for src in sources:
        name = src.get('name') or '主激励'; signal = src.get('signal') if isinstance(src.get('signal'), dict) else {'type': 'tone_burst', 'window': 'hann'}
        profile = src.get('profile') if isinstance(src.get('profile'), dict) else {'type': 'uniform'}
        if isinstance(src.get('frequency_mhz'), (int, float)) and not .2 - 1e-9 <= src['frequency_mhz'] <= .4 + 1e-9: out.append(f'{name} 频率 {src["frequency_mhz"]:g} MHz（训练 0.2–0.4）')
        if signal.get('type') != 'tone_burst' or signal.get('window', 'hann') != 'hann': out.append(f'{name} 信号 {signal.get("type")}（训练只有 Hann 窗单频脉冲）')
        elif isinstance(src.get('cycles'), (int, float)) and src['cycles'] < 5 - 1e-9: out.append(f'{name} {src["cycles"]:g} 个周期（训练 5 个，周期越少频带越宽）')
        if src.get('face') not in TRAINED_FACES: out.append(f'{name} 加载面 {src.get("face")}（训练为顶面和端面）')
        if isinstance(src.get('radius_mm'), (int, float)) and src['radius_mm'] > 0: out.append(f'{name} 圆形加载区 半径 {src["radius_mm"]:g} mm（训练只有整面加载）')
        if profile.get('type') == 'comb': out.append(f'{name} 梳状分布（{_tested(forms, "labels-d-comb")}，齿宽 {profile.get("tooth_width_mm")} mm 在粗网格上表示不准；请用冻结网络、h ≤ 1.5 mm）')
        elif profile.get('type') == 'gaussian' and isinstance(profile.get('sigma_mm'), (int, float)) and profile['sigma_mm'] < 1.5 - 1e-9: out.append(f'{name} 高斯宽度 {profile["sigma_mm"]:g} mm（训练 1.5–2 mm）')
    if not any(b.get('type') == 'symmetry' and b.get('face') == 'ymax' for b in scene.get('boundaries') or []): out.append('没有 y 最大面对称（训练为半宽模型；全宽板核验 1 例通过）')
    return out


def check(scene, name=DEFAULT_MODEL):
    """Hard limits of the fast engine; raises ValueError."""
    from physics.materials import is_custom
    require(isinstance(scene, dict) and scene.get('schema') == 'workai.hyperelastic.solid.v2', '快速引擎只接受 v2 场景。')
    require((scene.get('geometry') or {}).get('kind') == 'plate', '快速引擎只在平板上训练过；孔、缺口和管请用冻结网络。')
    # research/coarse-range: of 9 defect cases the accurate model passes every gate in 3 (side-drilled hole twice, notch once); the mesh
    # and the layer spacing next to a defect are set by the defect, so the coarse grid is only 1–5 times faster there.
    require(not scene.get('defects'), '含缺陷的模型请用冻结网络：粗网格算子在缺陷算例的核验中只有部分通过（反射波的非线性增量 2f 超过 5%），'
            '而且缺陷附近的网格和时间步由缺陷尺寸决定，粗网格只快 1–5 倍。')
    require(not is_custom(scene.get('material')), '快速引擎的算子只用训练所用的铝训练过；其他材料请用冻结网络。')
    model = json.loads(model_path(name).read_text(encoding='utf-8'))
    return model


def prepare(scene, name, network):
    """What the worker's record path needs to run a coarse model: (scene on the coarse grid, mesh carrying the learned element
    weights, network carrying the learned derivative filter, the model's time layer, record extras).

    `network` from graphnet.model.load(model degree, long_record=True). The record itself is computed by `graphnet.rollout`
    (`coarsegraph.record`), so the fast engine has what the fine network has: records of any length written in chunks and
    resumable, compact records for large models, selectable controls, every core."""
    from coarsegraph.record import setup
    check(scene, name); require(network['material_check']['pass'], '冻结能量网络与训练材料不一致。')
    coarse, mesh, learned, layer, model = setup(scene, model_path(name), network); summary = model_summary(model_path(name))
    extra = {'engine': 'workai_coarse_graph_offline', 'engine_scope': 'learned_coarse_graph_operator',
             'coarse_model': dict(summary, out_of_training_range=out_of_range(scene, name),
                                  accuracy_note='学习的粗网格算子：精度是模型记录的留出测试和范围核验的结果，不是其他场景的误差界；超出训练与核验范围的场景需逐例核对。')}
    return coarse, mesh, learned, layer, extra
