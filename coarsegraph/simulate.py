"""Coarse-graph forward runs and fine-reference loading.

A coarse run uses the version-2 input layer (loads, signals, constraints,
absorbing field) on a coarse GLL graph, the frozen energy network and state
update, and the learned element operators from `coarsegraph.operator`. The
element messages are the frozen compiled kernels; nothing here calls a
numerical solver.
"""
import copy
import json
from pathlib import Path
import numpy as np
from solid.schema import require
from physics.receivers import receiver_points
from spectral3d.mesh import make_mesh, interpolation
from graphwave.inference import inverse_mapping, graph_message
from graphnet.inputs import build, signal_table
from graphnet.predict import layers
from graphnet.recurrence import forward
from .operator import apply

LAWS = {'linear': 0, 'stvk': 1, 'murnaghan': 2}


class Coarse:
    """Everything about one scene on one coarse graph that does not depend on the learned operator."""

    def __init__(self, scene, degree, h_mm, cfl=.3):
        s = copy.deepcopy(scene); s['numerics']['h_mm'] = h_mm; s['numerics']['cfl'] = cfl; self.scene = s
        self.mesh = make_mesh(s, degree, h_mm); self.inputs = build(s, self.mesh); require(self.inputs is not None, '粗网格算子只接受 v2 场景。')
        receivers = receiver_points(s); self.names = [rx['name'] for rx in receivers]
        self.taps = [interpolation(self.mesh, rx['point_mm']) for rx in receivers]; self.directions = np.array([rx['direction'] for rx in receivers], dtype=float)
        self.dt, self.subdivisions, self.frames, self.steps = layers(s, self.mesh, cfl)
        self.signals = signal_table(self.inputs, np.arange(self.steps + 1) * self.dt * 1e6)
        self.damped = self.inputs['alpha'] is not None; self.alpha = self.inputs['alpha'] if self.damped else np.zeros(len(self.mesh['mass']))
        self.inverse = inverse_mapping(self.mesh); self.rho = s['material']['rho_kg_m3']
        self.times_us = np.arange(self.frames) * s['acquisition']['sample_step_us']

    def operators(self, D, q, m):
        return apply(self.mesh, self.rho, D, q, m)

    def stability(self, D, volume, mass, weights, iterations=60, seed=0):
        """dt · sqrt(λ_max(M⁻¹K)) by power iteration on the linear law; central differences need < 2."""
        rng = np.random.default_rng(seed); v = rng.normal(size=(len(mass), 3)) * 1e-12; v *= self.inputs['free']; lam = 0.
        for _ in range(iterations):
            if np.ndim(D) == 3:
                from .kernel import graph_message as axis_message
                k = axis_message(v, self.mesh['elements'], D[0], D[1], D[2], self.inverse, volume, weights['quadratic_tensor'], weights['cubic_tensor'], 0)[0] / mass[:, None]
            else: k = graph_message(v, self.mesh['elements'], D, self.inverse, volume, weights['quadratic_tensor'], weights['cubic_tensor'], 0)[0] / mass[:, None]
            k *= self.inputs['free']; norm = np.linalg.norm(k); lam = norm / max(np.linalg.norm(v), 1e-300); v = k / norm * 1e-12
        return float(self.dt * np.sqrt(lam))

    def run(self, D, volume, mass, law, weights, probes_out=False):
        """Projected waveforms (frames × receivers) and the energy log (plus 3-component probes on request)."""
        law = LAWS[law] if isinstance(law, str) else law
        if np.ndim(D) == 3:
            from .kernel import forward as axis_forward
            fields, energy, strain, jac = axis_forward(self.mesh['elements'], np.ascontiguousarray(D[0]), np.ascontiguousarray(D[1]), np.ascontiguousarray(D[2]), self.inverse, volume, mass,
                                                       self.inputs['loads'], self.signals, self.inputs['free'], self.alpha, self.damped, weights['quadratic_tensor'], weights['cubic_tensor'],
                                                       weights['state_weights'], law, self.dt, self.subdivisions, self.frames)
        else:
            fields, energy, strain, jac = forward(self.mesh['elements'], D, self.inverse, volume, mass, self.inputs['loads'], self.signals, self.inputs['free'], self.alpha, self.damped,
                                                  weights['quadratic_tensor'], weights['cubic_tensor'], weights['state_weights'], law, self.dt, self.subdivisions, self.frames)
        probes = np.stack([np.einsum('n,tnc->tc', w, fields[:, ids]) for ids, w in self.taps], 1)
        projected = np.einsum('trc,rc->tr', probes, self.directions)
        return (projected, energy, probes) if probes_out else (projected, energy)


def fine_projected(out_dir, names):
    """Projected waveforms of a finished run (sweep layout), in the order of `names`."""
    out = Path(out_dir); record = json.loads((out / 'result.json').read_text(encoding='utf-8')); lines = np.load(out / 'lines.npz'); table = {}
    for rx in record['receivers']: table[rx['name']] = np.asarray(rx['projected_m'])
    for line in record['receiver_lines']:
        projected = lines[f'line_{line["name"]}_projected_m']
        for i in range(projected.shape[1]): table[f'{line["name"]}[{i}]'] = projected[:, i]
    require(set(names) <= set(table), '参考结果缺少接收点。')
    return np.stack([table[n] for n in names], 1), np.asarray(record['times_us'])


def label_pairs(campaign_dir):
    """(parameters without constitutive, linear case dir, nonlinear case dir) for every completed control pair."""
    campaign = Path(campaign_dir); manifest = json.loads((campaign / 'manifest.json').read_text(encoding='utf-8')); done = [c for c in manifest['cases'] if c.get('phase') == 'completed']
    pairs = []
    for a in done:
        pa = dict(a['parameters'])
        if pa.pop('constitutive', None) != 'linear': continue
        for b in done:
            pb = dict(b['parameters'])
            if pb.pop('constitutive', None) == 'murnaghan' and pa == pb:
                pairs.append({'campaign': campaign.name, 'parameters': pa, 'linear': str(campaign / 'cases' / a['case_id']), 'murnaghan': str(campaign / 'cases' / b['case_id']),
                              'key': campaign.name + ':' + ','.join(f'{k}={v}' for k, v in sorted(pa.items()))})
    return pairs
