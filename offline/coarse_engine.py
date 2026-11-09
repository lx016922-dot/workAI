"""Fast engine: the learned coarse-graph operator (coarsegraph) inside the offline software.

The network is the same frozen graph network (energy layer, state update); only the element operators
(derivative filter, stiffness and mass quadrature) are the learned coarse ones, on the model's own coarse
grid (degree and element size stored in the model). It is a research model: its accuracy is the held-out
test result recorded in the model, and scenes outside its training range must be checked case by case.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from solid.schema import require

REPOSITORY = Path(__file__).resolve().parents[1]
MODEL = REPOSITORY / 'coarsegraph' / 'models' / 'p4-h3-harmonic' / 'model.json'
LAWS = ('linear', 'stvk', 'murnaghan')
MAX_FRAMES = 4001
FIELD_BUDGET_BYTES = 768 * 1024 ** 2
# Training range of the shipped model (coarsegraph/specs): 48 × 2 mm half-width aluminium plates, symmetry at y = 2 mm.
TRAINING_RANGE = [('geometry.length_mm', '板长', 48, 48, 'mm'), ('geometry.width_mm', '半宽', 2, 2, 'mm'), ('geometry.thickness_mm', '板厚', 1.6, 2.4, 'mm'),
                  ('acquisition.duration_us', '记录', 0, 30, 'μs')]


def _get(scene, path):
    value = scene
    for key in path.split('.'): value = value.get(key) if isinstance(value, dict) else None
    return value


def model_summary(path=MODEL):
    """What the interface shows about the shipped model."""
    model = json.loads(Path(path).read_text(encoding='utf-8')); test = model['summary']['learned']['test']; base = model['summary']['baseline_gll']['test']
    return {'name': Path(path).parent.name, 'degree': model['degree'], 'h_mm': model['h_mm'], 'parameters': model['parameter_count'],
            'test_cases': test['pairs'], 'test_passed': test['passed'], 'baseline_test_passed': base['passed'],
            'test_worst_waveform_l2': test['worst_waveform_l2'], 'test_worst_increment_a2': test['worst_increment_a2'],
            'coarse_nodes_example': model['cost']['coarse_nodes'], 'model_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def out_of_range(scene):
    """Training-range items the scene leaves (information; the run is still allowed)."""
    out = []
    for path, label, lo, hi, unit in TRAINING_RANGE:
        v = _get(scene, path)
        if isinstance(v, (int, float)) and not lo - 1e-9 <= v <= hi + 1e-9: out.append(f'{label} {v:g} {unit}（训练 {lo:g}{"–" + format(hi, "g") if hi != lo else ""} {unit}）')
    for src in [scene['source'], *(scene.get('sources') or [])]:
        if not .2 - 1e-9 <= src['frequency_mhz'] <= .4 + 1e-9: out.append(f'{src.get("name") or "主激励"} 频率 {src["frequency_mhz"]:g} MHz（训练 0.2–0.4）')
    if not any(b.get('type') == 'symmetry' and b.get('face') == 'ymax' for b in scene.get('boundaries') or []): out.append('没有 y 最大面对称（训练为半宽模型）')
    return out


def check(scene):
    """Hard limits of the fast engine; raises ValueError."""
    from physics.materials import is_custom
    require(isinstance(scene, dict) and scene.get('schema') == 'workai.hyperelastic.solid.v2', '快速引擎只接受 v2 场景。')
    require((scene.get('geometry') or {}).get('kind') == 'plate', '快速引擎只在平板上训练过；孔、缺口和管请用冻结网络。')
    require(not is_custom(scene.get('material')), '快速引擎的算子只用训练所用的铝训练过；其他材料请用冻结网络。')
    model = json.loads(MODEL.read_text(encoding='utf-8'))
    frames = round(scene['acquisition']['duration_us'] / scene['acquisition']['sample_step_us']) + 1
    require(frames <= MAX_FRAMES, f'快速引擎一次最多 {MAX_FRAMES} 个采样点（短记录）；长记录请用冻结网络。')
    return model


def run(scene, callback=None):
    """mesh, fields (linear, stvk, full), per-law diagnostics, times and record extras."""
    from coarsegraph.run import load_model
    from coarsegraph.simulate import Coarse
    from graphnet.model import load
    model = check(scene); model, (D, q, m) = load_model(MODEL); network = load(3)
    require(network['material_check']['pass'], '冻结能量网络与训练材料不一致。')
    c = Coarse(scene, model['degree'], model['h_mm'], model['cfl']); D, volume, mass = c.operators(D, q, m)
    require(c.frames * len(mass) * 3 * 8 * 3 <= FIELD_BUDGET_BYTES, '粗网格完整位移场超过内存预算，请缩短记录。')
    stability = c.stability(D, volume, mass, network['weights']); require(stability <= 1.5, f'该场景在粗网格上的时间步不稳定（{stability:.2f}）。')
    fields, rows = [], []
    for law in range(3):
        started = time.monotonic()
        field, energy, strain, jac = c.fields(D, volume, mass, law, network['weights'])
        stored = energy[:, 1] + energy[:, 2] + energy[:, 4]; balance = float(np.max(np.abs(stored - energy[:, 3])) / max(np.max(np.abs(energy[:, 3])), 1e-30))
        columns = ['time_us', 'kinetic_j', 'strain_j', 'external_work_j'] + (['dissipated_j'] if c.damped else [])
        rows.append({'law': LAWS[law], 'elapsed_s': time.monotonic() - started, 'internal_steps': c.steps, 'internal_step_us': c.dt * 1e6, 'layer_step_us': c.dt * 1e6, 'recurrent_layers': c.steps,
                     'nodes': len(mass), 'elements': len(c.mesh['elements']), 'degree': model['degree'], 'h_mm': model['h_mm'], 'stability_number': stability,
                     'max_strain': float(strain), 'min_J': float(jac), 'external_work_balance_relative': balance, 'energy_balance_within_1_percent': balance <= .01,
                     'energy_columns': columns, 'energy_samples': energy[:, :len(columns)].tolist(), 'neural_forward_only': True, 'numerical_correction': False,
                     'engine': 'coarse_graph_learned_operator'})
        fields.append(field)
        if callback: callback(rows[-1])
    summary = model_summary()
    extra = {'engine': 'workai_coarse_graph_offline', 'engine_scope': 'learned_coarse_graph_operator_short_record',
             'coarse_model': dict(summary, out_of_training_range=out_of_range(scene),
                                  accuracy_note='学习的粗网格算子：精度是模型记录的留出测试结果，不是其他场景的误差界；超出训练范围的场景需逐例核对。')}
    return c.mesh, fields, rows, c.times_us, extra
