"""Frozen graph network prediction through the version-2 input layer (short records)."""
import time
import numpy as np
from solid.schema import require
from graphwave.inference import inverse_mapping
from .inputs import build, signal_table, resolution
from .recurrence import forward

LAWS = ('linear', 'stvk', 'murnaghan')
FORBIDDEN = ('spectral3d.solver', 'spectral3d.curved_solver', 'spectral3d.implicit', 'solid.fem')


def layers(scene, mesh, cfl):
    """The frozen predictor's layer spacing: Δt from CFL × node spacing / 10000 m/s, an integer number of layers per sample."""
    require(.05 <= cfl <= .3, '网络层间距的验证范围为 CFL 0.05–0.3。')
    sample = scene['acquisition']['sample_step_us'] * 1e-6
    subdivisions = int(np.ceil(sample / (cfl * float(mesh['min_node_spacing']) / 10000.)))
    frames = round(scene['acquisition']['duration_us'] / scene['acquisition']['sample_step_us']) + 1
    steps = (frames - 1) * subdivisions
    require(steps <= 150000 and frames * len(mesh['mass']) * 3 * 8 <= 512 * 1024 ** 2, '网络递推层数或场存储超出预算（≤150000 层，单组场 ≤512 MiB）。')
    return sample / subdivisions, subdivisions, frames, steps


def predict(scene, mesh, model, laws=(0, 1, 2), callback=None):
    """Fields (frames × nodes × 3) per requested law and one diagnostics row per law."""
    weights = model['weights']; degree = int(mesh['degree']); cfl = scene['numerics']['cfl']
    require(f'derivative_{degree}' in weights, f'所选权重没有 {degree} 阶图层。')
    inputs = build(scene, mesh)
    if inputs is None:
        from graphwave.predict import predict as frozen
        fields, diagnostics = frozen(scene, mesh, weights, cfl=cfl, callback=callback)
        rows = []
        for law, d in enumerate(diagnostics):
            rows.append(dict(d, input_layer='frozen_v1', internal_steps=d['recurrent_layers'], internal_step_us=d['layer_step_us'], energy_columns=['time_us', 'kinetic_j', 'strain_j', 'external_work_j'],
                             nodes=len(mesh['mass']), elements=len(mesh['elements']), degree=degree, h_mm=float(mesh['h_mm'])))
        return {'fields': {LAWS[i]: fields[i] for i in laws}, 'diagnostics': [rows[i] for i in laws], 'route': 'frozen_v1', 'inputs': None}
    dt, subdivisions, frames, steps = layers(scene, mesh, cfl)
    signals = signal_table(inputs, np.arange(steps + 1) * dt * 1e6)
    damped = inputs['alpha'] is not None; alpha = inputs['alpha'] if damped else np.zeros(len(mesh['mass']))
    inverse = inverse_mapping(mesh); fields = {}; rows = []
    for law in laws:
        started = time.monotonic()
        field, energy, strain, jac = forward(mesh['elements'], weights[f'derivative_{degree}'], inverse, mesh['volume_weight'], mesh['mass'], inputs['loads'], signals,
                                             inputs['free'], alpha, damped, weights['quadratic_tensor'], weights['cubic_tensor'], weights['state_weights'], law, dt, subdivisions, frames)
        stored = energy[:, 1] + energy[:, 2] + energy[:, 4]; scale = max(np.max(np.abs(energy[:, 3])), np.max(np.abs(stored)), 1e-30)
        balance = float(np.max(np.abs(stored - energy[:, 3])) / scale)
        columns = ['time_us', 'kinetic_j', 'strain_j', 'external_work_j'] + (['dissipated_j'] if damped else [])
        rows.append({'law': LAWS[law], 'input_layer': 'v2_multi_source', 'elapsed_s': time.monotonic() - started, 'recurrent_layers': steps, 'layer_step_us': dt * 1e6,
                     'internal_steps': steps, 'internal_step_us': dt * 1e6, 'nodes': len(mesh['mass']), 'elements': len(mesh['elements']), 'degree': degree, 'h_mm': float(mesh['h_mm']),
                     'max_strain': float(strain), 'min_J': float(jac), 'external_work_balance_relative': balance, 'energy_balance_within_1_percent': balance <= .01,
                     'dissipated_energy_j': float(energy[-1, 4]), 'energy_columns': columns, 'energy_samples': energy[:, :len(columns)].tolist(),
                     'signal_peaks': [float(np.max(np.abs(v))) for v in signals], 'loaded_area_mm2': inputs['area_m2'] * 1e6, 'resolution': resolution(scene, mesh),
                     'neural_forward_only': True, 'numerical_correction': False, 'reference_responses_read': False, **inputs['summary']})
        fields[LAWS[law]] = field
        if callback: callback(rows[-1])
    return {'fields': fields, 'diagnostics': rows, 'route': 'v2_input_layer', 'inputs': inputs}
