"""Independent physical and PyTorch export checks for the continued model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from .model import GraphWave
from .continue_training import atomic, digest
from spectral3d.mesh import make_mesh, gll
from offline.domain import templates
from operator3d.physics import piola


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(2026102409)
    rng = np.random.default_rng(2026102409)
    model = GraphWave(degrees=(3, 4, 5))
    model.load_state_dict(torch.load(args.model / 'best.pt', weights_only=True)['model_state'])
    frozen = torch.load(args.parent / 'best.pt', weights_only=True)['model_state']
    for name, value in frozen.items():
        assert torch.equal(model.state_dict()[name], value), name
    derivative_errors = {}
    for degree in (3, 4, 5):
        expected = gll(degree)[2]
        actual = model.derivative(degree).detach().numpy()
        derivative_errors[str(degree)] = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
        np.testing.assert_allclose(actual.sum(-1), 0, rtol=0, atol=1e-14)
    assert derivative_errors['5'] < 1e-12
    scenes = templates()
    gradients = torch.tensor(rng.normal(size=(2048, 3, 3)) * 1e-4, dtype=torch.float64)
    material_errors = {}
    for law, name in enumerate(['linear', 'stvk', 'murnaghan']):
        expected = piola(gradients, scenes['plate']['material'], name)[0]
        actual = model.stress(gradients, law)
        material_errors[name] = float((torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)).detach())
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-6)
    exports = []
    for kind, scene in scenes.items():
        mesh = make_mesh(scene, 5, 10.)
        inverse = (mesh['inverse_jacobian'] if 'inverse_jacobian' in mesh else
                   np.broadcast_to(np.eye(3)[None, None, None, None] * (2 / mesh['size'])[:, None, None, None, None, :],
                                   (*mesh['elements'].shape, 3, 3)))
        current = torch.tensor(rng.normal(size=mesh['points'].shape) * 1e-9, dtype=torch.float64)
        cells = torch.tensor(mesh['elements'].astype(np.int64))
        inv, volume = torch.tensor(inverse.copy()), torch.tensor(mesh['volume_weight'])
        output = {'degree': np.array(5), 'current': current.numpy(), 'cells': mesh['elements'],
                  'inverse': inverse, 'volume': mesh['volume_weight'], 'gradients': gradients.numpy()}
        for law in range(3):
            message = model.graph_message(current, cells, inv, volume, law, 5)
            torch.testing.assert_close(message.sum(0), torch.zeros(3, dtype=torch.float64), atol=1e-10, rtol=0)
            output['message_' + str(law)] = message.detach().numpy()
            output['stress_' + str(law)] = model.stress(gradients, law).detach().numpy()
        path = args.output / (kind + '-torch.npz')
        np.savez(path, **output)
        exports.append({'kind': kind, 'file': path.name, 'sha256': digest(path)})
    atomic(args.output / 'physics.json', {'passed': True, 'weights_sha256': digest(args.model / 'weights.npz'),
           'parent_parameters_bit_identical': True, 'filter_relative_l2': derivative_errors,
           'constitutive_relative_l2': material_errors, 'four_geometry_force_conservation_passed': True,
           'torch_exports': exports, 'transient_accuracy_claimed': False})
    print(json.dumps({'passed': True, 'filter_relative_l2': derivative_errors, 'constitutive_relative_l2': material_errors}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
