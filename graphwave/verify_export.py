"""Produce independent PyTorch outputs for verifying the compiled forward."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .model import GraphWave
from spectral3d.mesh import make_mesh
from spectralop.data import digest
from operator3d.train import atomic


def run(args):
    torch.set_num_threads(2);torch.manual_seed(2026102402);model=GraphWave();saved=torch.load(args.model/'candidate.pt',weights_only=True);model.load_state_dict(saved['model_state']);rng=np.random.default_rng(2026102402)
    scene=json.loads(args.input.read_text())['scene'];mesh=make_mesh(scene,4,10.)
    inverse=np.ascontiguousarray(mesh['inverse_jacobian']) if 'inverse_jacobian' in mesh else np.ascontiguousarray(np.broadcast_to(np.eye(3)[None,None,None,None]*(2/mesh['size'])[:,None,None,None,None,:],(*mesh['elements'].shape,3,3)))
    current=torch.tensor(rng.normal(size=mesh['points'].shape)*1e-9,dtype=torch.float64,requires_grad=True)
    cells=torch.tensor(mesh['elements'].astype(np.int64));inv=torch.tensor(inverse);volume=torch.tensor(mesh['volume_weight'])
    result={'current':current.detach().numpy(),'cells':mesh['elements'],'inverse':inverse,'volume':mesh['volume_weight']}
    gradients=torch.tensor(rng.normal(size=(257,3,3))*2e-5,dtype=torch.float64)
    result['gradients']=gradients.numpy()
    for law in range(3):
        message=model.graph_message(current,cells,inv,volume,law,4);result['message_'+str(law)]=message.detach().numpy();result['stress_'+str(law)]=model.stress(gradients,law).detach().numpy()
        # Tied scatter must conserve the total internal force.
        torch.testing.assert_close(message.sum(0),torch.zeros(3,dtype=torch.float64),atol=1e-11,rtol=0)
    with np.load(args.model/'candidate-weights.npz',allow_pickle=False) as export:
        for key,value in model.export()['derivatives'].items():np.testing.assert_array_equal(value,export['derivative_'+key])
    np.savez(args.output,**result)
    atomic(args.output.with_suffix('.json'),{'torch_outputs_sha256':digest(args.output),'checkpoint_sha256':digest(args.model/'candidate.pt'),'weights_sha256':digest(args.model/'candidate-weights.npz'),'rigid_translation_conservation_passed':True,'geometry_kind':scene['geometry']['kind'],'verification_source_sha256':digest(Path(__file__))})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
