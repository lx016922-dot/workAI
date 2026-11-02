"""Conforming Cartesian spectral elements, exact plate and through-width notch.

The GLL rule defines a positive diagonal mass and a discrete energy. Increasing
both polynomial degree and element count is required to check spatial error.
"""
import math
import numpy as np
from numpy.polynomial.legendre import Legendre


def gll(degree):
    if type(degree) is not int or not 2 <= degree <= 6:
        raise ValueError('Use polynomial degree 2–6.')
    poly = Legendre.basis(degree)
    x = np.r_[-1., poly.deriv().roots(), 1.]
    w = 2/(degree*(degree+1)*poly(x)**2)
    bary = np.array([1/np.prod(x[i]-np.delete(x,i)) for i in range(len(x))])
    d = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j: d[i,j] = bary[j]/bary[i]/(x[i]-x[j])
        d[i,i] = -d[i].sum()
    return x,w,d,bary


def make_mesh(scene, degree=3, h_mm=1.6):
    g=scene['geometry'];kind=g['kind']
    if kind in ['pipe','hole']:
        from .curved_mesh import make_mesh as curved_mesh
        return curved_mesh(scene,degree,h_mm)
    if kind not in ['plate','notch']:
        raise ValueError('This reference implementation supports plates and through-width notches only.')
    if not np.isfinite(h_mm) or h_mm <= 0: raise ValueError('Positive element size required.')
    x,w,d,bary=gll(degree)
    limits=[[0,g['length_mm']],[0,g['width_mm']],[0,g['thickness_mm']]]
    if kind=='notch':
        a=g['notch_x_mm']-g['notch_width_mm']/2;b=g['notch_x_mm']+g['notch_width_mm']/2
        z=g['thickness_mm']-g['notch_depth_mm']
        if not 0<a<b<g['length_mm'] or not 0<z<g['thickness_mm']: raise ValueError('Invalid notch.')
        limits[0]=[0,a,b,g['length_mm']];limits[2]=[0,z,g['thickness_mm']]
    axes=[];edges=[]
    for cuts in limits:
        edge=np.concatenate([np.linspace(a,b,math.ceil((b-a)/h_mm)+1)[:-1] for a,b in zip(cuts[:-1],cuts[1:])]+[[cuts[-1]]])*.001
        axis=np.r_[np.concatenate([a+(x[:-1]+1)*(b-a)/2 for a,b in zip(edge[:-1],edge[1:])]),edge[-1]]
        edges.append(edge);axes.append(axis)
    points=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    index=np.arange(len(points)).reshape(*(len(a) for a in axes));elements=[];origins=[];sizes=[]
    for i in range(len(edges[0])-1):
        for j in range(len(edges[1])-1):
            for k in range(len(edges[2])-1):
                origin=np.array([edges[q][v] for q,v in enumerate([i,j,k])]);size=np.array([edges[q][v+1] for q,v in enumerate([i,j,k])])-origin
                center=(origin+size/2)*1000
                if kind=='notch' and a<center[0]<b and center[2]>z: continue
                elements.append(index[i*degree:(i+1)*degree+1,j*degree:(j+1)*degree+1,k*degree:(k+1)*degree+1]);origins.append(origin);sizes.append(size)
    elements=np.asarray(elements,dtype=np.int32);used=np.unique(elements);remap=np.full(len(points),-1,dtype=np.int32);remap[used]=np.arange(len(used));points=points[used];elements=remap[elements]
    if len(points)>100000 or elements.size>2000000: raise ValueError('Reference mesh memory budget exceeded.')
    sizes=np.array(sizes);weights=np.einsum('i,j,k->ijk',w,w,w)
    volume_weight=np.prod(sizes,axis=1)[:,None,None,None]/8*weights
    mass=np.zeros(len(points));np.add.at(mass,elements.ravel(),(volume_weight*scene['material']['rho_kg_m3']).ravel())
    if np.any(mass<=0): raise ValueError('Nonpositive mass.')
    return {'points':points,'elements':elements,'origin':np.array(origins),'size':sizes,'degree':degree,
            'gll':x,'weights_1d':w,'derivative':d,'barycentric':bary,'volume_weight':volume_weight,
            'mass':mass,'min_node_spacing':float(min(np.min(np.diff(a)) for a in axes)),'h_mm':h_mm}


def boundary_data(mesh,scene):
    if str(mesh.get('mapping_kind',''))=='isoparametric':
        from .curved_mesh import boundary_data as curved_boundary
        return curved_boundary(mesh,scene)
    source=scene['source'];face=source['face'];points=mesh['points'];load=np.zeros_like(points)
    if face not in ['xmin','xmax','ymin','ymax','zmin','zmax'] or source['radius_mm']!=0:
        raise ValueError('Spectral reference currently requires a complete planar loading face.')
    axis='xyz'.index(face[0]);end=0 if face.endswith('min') else -1
    bound=points[:,axis].min() if end==0 else points[:,axis].max();area=0.;n=mesh['degree']+1
    for e,nodes in enumerate(mesh['elements']):
        loc=[slice(None)]*3;loc[axis]=end;ids=nodes[tuple(loc)]
        if not np.allclose(points[ids,axis],bound,rtol=0,atol=1e-13): continue
        tangential=[q for q in range(3) if q!=axis]
        weights=np.outer(mesh['weights_1d'],mesh['weights_1d'])*np.prod(mesh['size'][e,tangential])/4
        area+=weights.sum()
        np.add.at(load,ids.ravel(),(weights[...,None]*np.array(source['direction'])*source['amplitude_mpa']*1e6).reshape(n*n,3))
    if area<=0: raise ValueError('Empty loading surface.')
    fixed=np.array([],dtype=np.int32);f=scene['boundary']['fixed_face']
    if f!='none':
        axis='xyz'.index(f[0]);bound=points[:,axis].min() if f.endswith('min') else points[:,axis].max()
        fixed=np.flatnonzero(abs(points[:,axis]-bound)<1e-13).astype(np.int32)
    if np.any(load[fixed]): raise ValueError('Load intersects fixed boundary.')
    return load,fixed,float(area)


def interpolation(mesh,point_mm):
    if str(mesh.get('mapping_kind',''))=='isoparametric':
        from .curved_mesh import interpolation as curved_interpolation
        return curved_interpolation(mesh,point_mm)
    point=np.asarray(point_mm,dtype=float)*.001
    if point.shape!=(3,) or not np.isfinite(point).all(): raise ValueError('Invalid receiver.')
    inside=np.flatnonzero(np.all(point>=mesh['origin']-1e-13,axis=1)&np.all(point<=mesh['origin']+mesh['size']+1e-13,axis=1))
    if not len(inside): raise ValueError('Receiver is outside the material, including the notch void.')
    e=int(inside[0]);local=2*(point-mesh['origin'][e])/mesh['size'][e]-1;weights=[]
    for value in local:
        delta=value-mesh['gll'];hit=np.flatnonzero(abs(delta)<1e-12)
        if len(hit): row=np.eye(mesh['degree']+1)[hit[0]]
        else: row=mesh['barycentric']/delta;row/=row.sum()
        weights.append(row)
    return mesh['elements'][e].ravel(),np.einsum('i,j,k->ijk',*weights).ravel()
