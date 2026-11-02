"""Exact quarter-domain reflection reduction for uniform axial plate loading."""
import copy
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from spectral3d.mesh import boundary_data
from .graded_mesh import make_graded_mesh,_edges
from .modal_sparse import stiffness


def make_quarter_mesh(scene,degree,h_mm,ratios,counts):
    ratios=np.asarray(ratios,dtype=float)
    if ratios.shape!=(3,) or not np.isfinite(ratios).all() or np.any(ratios<1) or np.any(ratios>100):
        raise ValueError('Three finite grading ratios in [1, 100] required')
    source=scene['source'];direction=np.asarray(source['direction'],dtype=float)
    if (scene['geometry']['kind']!='plate' or source['radius_mm']!=0 or
        source['face'] not in ['xmin','xmax'] or scene['boundary']['fixed_face'] not in ['xmin','xmax'] or
        direction.shape!=(3,) or not np.isfinite(direction).all() or direction[0]==0 or np.any(direction[1:]!=0)):
        raise ValueError('Quarter domain requires uniform axial traction and a fixed rectangular plate')
    reduced=copy.deepcopy(scene)
    for key in ['width_mm','thickness_mm']:reduced['geometry'][key]*=.5
    mesh=make_graded_mesh(reduced,degree,h_mm,(ratios[0],1.,1.),counts)
    lengths=np.array([reduced['geometry'][key] for key in ['length_mm','width_mm','thickness_mm']])*1e-3
    edges=[np.asarray(mesh['axis_edges_m'][0])]+[_edges(lengths[a],counts[a],ratios[a],False,False) for a in [1,2]]
    axes=[np.r_[np.concatenate([left+(mesh['gll'][:-1]+1)*(right-left)/2 for left,right in zip(edge[:-1],edge[1:])]),edge[-1]] for edge in edges]
    points=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    cells=mesh['elements'];origin=points[cells[:,0,0,0]];size=points[cells[:,-1,-1,-1]]-origin
    quadrature=np.einsum('i,j,k->ijk',*([mesh['weights_1d']]*3))
    volume=np.prod(size,axis=1)[:,None,None,None]/8*quadrature
    mass=np.zeros(len(points));np.add.at(mass,cells.ravel(),(volume*scene['material']['rho_kg_m3']).ravel())
    mesh.update(points=points,origin=origin,size=size,volume_weight=volume,mass=mass,
                min_node_spacing=min(np.diff(axis).min() for axis in axes),grading_ratios=list(ratios),
                axis_edges_m=[edge.tolist() for edge in edges],reflection_planes_m=lengths[1:].tolist(),
                physical_domain_multiplicity=4)
    return mesh


def material_symmetry(quadratic,cubic=None):
    """Reject learned tensors whose exact energy breaks the retained symmetry."""
    errors=[]
    for signs in [np.array([1,1,1,-1,1,-1]),np.array([1,1,1,-1,-1,1])]:
        for tensor,parity in [(quadratic,signs[:,None]*signs[None,:]),
                (cubic,signs[:,None,None]*signs[None,:,None]*signs[None,None,:])]:
            if tensor is None:continue
            tensor=np.asarray(tensor)
            if tensor.shape!=parity.shape or not np.isfinite(tensor).all():
                raise ValueError('Invalid learned material tensor')
            norm=np.linalg.norm(tensor)
            errors.append(float(np.linalg.norm(tensor-tensor*parity)/norm) if norm else 0.)
    if max(errors)>1e-12:
        raise ValueError('Learned material tensors do not preserve axial reflection symmetry')
    return max(errors)


def free_dofs(mesh,scene):
    _,fixed,_=boundary_data(mesh,scene)
    free=np.ones((len(mesh['points']),3),dtype=bool);free[fixed]=False
    for axis,plane in zip([1,2],mesh['reflection_planes_m']):
        free[np.isclose(mesh['points'][:,axis],plane,rtol=0,atol=1e-14),axis]=False
    return free.ravel()


def modes(mesh,scene,derivative,quadratic,count):
    material_symmetry(quadratic)
    matrix=stiffness(mesh,derivative,quadratic);free=free_dofs(mesh,scene)
    mass=np.repeat(mesh['mass'],3);scale=1/np.sqrt(mass[free])
    normalized=diags(scale)@matrix[free][:,free]@diags(scale)
    start=np.random.default_rng(2026102524).normal(size=free.sum())
    values,vectors=eigsh(normalized,k=count,sigma=0,which='LM',tol=1e-11,v0=start,maxiter=5000)
    order=np.argsort(values);values=values[order];vectors=vectors[:,order]
    if np.any(values<=0):raise ValueError('Nonpositive quarter-domain graph mode')
    fields=np.zeros((count,len(mass)));fields[:,free]=(vectors*scale[:,None]).T
    return np.sqrt(values)/(2*np.pi),fields.reshape(count,-1,3)


def sampling(mesh,coordinates):
    from .modal_calibration import modal_sampling
    coordinates=np.asarray(coordinates);folded=coordinates.copy();signs=np.ones_like(coordinates)
    for axis,plane in zip([1,2],mesh['reflection_planes_m']):
        if np.any(coordinates[:,axis]<-1e-14) or np.any(coordinates[:,axis]>2*plane+1e-14):
            raise ValueError('Receiver outside physical reflected plate')
        mask=coordinates[:,axis]>plane
        folded[mask,axis]=2*plane-coordinates[mask,axis];signs[mask,axis]=-1
    interpolate=modal_sampling(mesh,folded)
    return lambda fields:interpolate(fields)*signs[None,:,:]
