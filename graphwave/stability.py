"""Sample frozen network material stability throughout the guarded strain range."""
import argparse,json,time
from pathlib import Path
import torch
from .model import GraphWave,mandel,STRAIN_SCALE,PAIRS,TRIPLES
from .release import atomic,digest


def run(a):
    torch.set_num_threads(2);torch.manual_seed(2026102407);started=time.monotonic();model=GraphWave();model.load_state_dict(torch.load(a.model/'best.pt',weights_only=True)['model_state'])
    count=256;raw=torch.randn(count,3,3,dtype=torch.float64);strain=(raw+raw.transpose(-1,-2))*.5;strain=strain/torch.linalg.matrix_norm(strain,dim=(-2,-1))[:,None,None]
    strain=strain*torch.linspace(1e-7,.001,count,dtype=torch.float64)[:,None,None]
    eig,v=torch.linalg.eigh(torch.eye(3,dtype=torch.float64)+2*strain);stretch=(v*eig.sqrt()[:,None,:])@v.transpose(-1,-2)
    q,_=torch.linalg.qr(torch.randn(count,3,3,dtype=torch.float64));q[:,:,0]*=torch.sign(torch.linalg.det(q))[:,None]
    f=q@stretch;g=f-torch.eye(3,dtype=torch.float64)
    tangent=torch.func.vmap(torch.func.jacrev(lambda value:model.stress(value,2)))(g).detach()
    symmetry=float((tangent-tangent.permute(0,3,4,1,2)).abs().max()/tangent.abs().max())
    normals=torch.randn(count,12,3,dtype=torch.float64);normals=normals/torch.linalg.vector_norm(normals,dim=-1,keepdim=True)
    acoustic=torch.einsum('bijkl,bdj,bdl->bdik',tangent,normals,normals)
    minimum=float(torch.linalg.eigvalsh((acoustic+acoustic.transpose(-1,-2))*.5).min())
    z=mandel(strain)/STRAIN_SCALE
    energy=1e3*(sum(w*torch.prod(z[:,list(pair)],dim=1) for w,pair in zip(model.quadratic,PAIRS))+STRAIN_SCALE*sum(w*torch.prod(z[:,list(term)],dim=1) for w,term in zip(model.cubic,TRIPLES)))
    minimum_energy=float(energy.detach().min());minimum_j=float(torch.linalg.det(f).min());assert minimum>0 and minimum_energy>0 and minimum_j>0 and symmetry<1e-12
    report={'schema':'workai.graphwave_sampled_material_stability.v1','passed':True,'checkpoint_sha256':digest(a.model/'best.pt'),'source_sha256':digest(Path(__file__)),
        'strain_states':count,'propagation_directions_per_state':12,'largest_green_strain_norm':.001,'finite_rotations_included':True,
        'smallest_sampled_acoustic_eigenvalue_pa':minimum,'smallest_sampled_energy_density_pa':minimum_energy,'minimum_J':minimum_j,'major_symmetry_relative_error':symmetry,
        'elapsed_s':time.monotonic()-started,'weights_changed':False,'scope':'Sampled local material checks; not additional transient generalization tests or a global ellipticity proof.'}
    atomic(a.output,report);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
