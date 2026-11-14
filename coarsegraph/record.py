"""Learned coarse operators through the chunked record of the version-2 input layer.

`coarsegraph.simulate.Coarse` (training and evaluation) runs the frozen short-record kernel on one thread and
keeps every node of every sample in memory. A coarse model is the frozen network with other element operators:
the learned derivative filter replaces the GLL one, and the learned quadrature and mass weights replace the
element volume weights and the assembled mass. `graphnet.rollout` takes exactly these three things from the
mesh and the model, so a coarse model can use it unchanged and gets what the fine network has there: chunked
and resumable records of any length, compact records, selectable controls, every core.

The layer is the one the operators were trained with: one stage, the spacing rule of `graphnet.predict.layers` for the
model's CFL (a central-difference layer and a velocity-Verlet stage are the same scheme; with an absorbing layer
the two forms of the damping term differ by about 2e-7 of the waveform, tested).
"""
import copy
from pathlib import Path
import numpy as np
from solid.schema import require
from .operator import apply
from .run import load_model

GLL = 'gll'   # the coarse grid without learning: the frozen operators on the model's grid


def model_layer(scene, mesh):
    """The time layer a coarse model was trained with, for `graphnet.rollout`: one stage, CFL × smallest node spacing / 10000 m/s
    (the spacing rule of `graphnet.predict.layers`, without its short-record budget). `scene` carries the model's CFL."""
    from graphnet.model import speed_bound
    from graphnet.timestep import frozen_plan
    return {'composition': [1.], 'subdivisions': int(frozen_plan(scene, mesh, (1.,), speed_bound(scene))['subdivisions']), 'time_order': 2,
            'reason': '粗网格算子训练时的时间层（单级，CFL × 最小节点间距 / 10000 m/s）'}


def setup(scene, model_path, network, learned=True):
    """(scene on the coarse grid, mesh with the element weights, network with the derivative filter, the model's layer) for `graphnet.rollout`.

    `network` from graphnet.model.load(degree, long_record=True). `learned` False keeps the GLL operators (the baseline a
    model is compared with)."""
    from spectral3d.feature_mesh import make_mesh
    model, (D, q, m) = load_model(model_path); require(model.get('axes', 'shared') == 'shared', '分块记录只支持共享算子模型。')
    degree = model['degree']; s = copy.deepcopy(scene); s['numerics']['h_mm'] = model['h_mm']; s['numerics']['cfl'] = model['cfl']
    mesh = make_mesh(s, degree, model['h_mm']); plan = model_layer(s, mesh)
    if learned:
        filter_, volume, mass = apply(mesh, s['material']['rho_kg_m3'], D, q, m)
        mesh = dict(mesh, volume_weight=volume, mass=mass); network = dict(network, weights=dict(network['weights'], **{f'derivative_{degree}': filter_}))
    return s, mesh, network, plan, model


def simulate(scene, model_path, directory, network=None, learned=True, identity=None, **settings):
    """Run a coarse model (or its GLL baseline) on a scene into a record directory; returns the manifest. `settings` go to `graphnet.rollout`."""
    from graphnet.model import load
    from graphnet.rollout import rollout
    model = load_model(model_path)[0]; network = network or load(model['degree'], long_record=True)
    s, mesh, net, plan, _ = setup(scene, model_path, network, learned)
    return rollout(s, mesh, net, Path(directory), identity or {'coarse_model': Path(model_path).parent.name, 'learned': learned}, time_plan=settings.pop('time_plan', plan), **settings)


def projected(record, scene):
    """Receiver waveforms projected on the receiver directions: {control: samples × receivers}."""
    directions = np.array([r['direction'] for r in scene['receivers']], dtype=float)
    return {name: np.einsum('trc,rc->tr', record['probes'][:, :, i], directions) for i, name in enumerate(record['manifest']['controls'])}
