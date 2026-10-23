"""Conforming isoparametric GLL meshes for annular pipes and through holes.

Geometry derivatives come from the same shape functions as displacements.
The curved boundary is polynomial and must be included in h/p convergence.
"""
import math
import numpy as np
from .mesh import gll


def prepare(points,elements,degree,rho,h_mm):
    x,w,d,bary=gll(degree);points=np.asarray(points);elements=np.asarray(elements,dtype=np.int32);local=points[elements]
    jac=np.stack([np.einsum('ia,eajkc->eijkc',d,local),np.einsum('ja,eiakc->eijkc',d,local),np.einsum('ka,eijac->eijkc',d,local)],axis=-1)
    determinant=np.linalg.det(jac)
    if np.any(determinant<=1e-20):raise ValueError('Nonpositive or degenerate curved Jacobian.')
    inverse=np.linalg.inv(jac);volume=determinant*np.einsum('i,j,k->ijk',w,w,w);mass=np.zeros(len(points));np.add.at(mass,elements.ravel(),(rho*volume).ravel())
    spacing=float(np.linalg.svd(jac,compute_uv=False)[...,-1].min()*np.diff(x).min())
    if spacing<1e-7 or len(points)>100000 or elements.size>2000000:raise ValueError('Curved mesh quality or size exceeds the reference budget.')
    return {'points':points,'elements':elements,'degree':degree,'gll':x,'weights_1d':w,'derivative':d,'barycentric':bary,'volume_weight':volume,'mass':mass,
            'jacobian':jac,'inverse_jacobian':inverse,'min_node_spacing':spacing,'h_mm':h_mm,'mapping_kind':'isoparametric',
            'bounds_min':local.min(axis=(1,2,3)),'bounds_max':local.max(axis=(1,2,3))}


def make_mesh(scene,degree=3,h_mm=1.6):
    g=scene['geometry'];kind=g['kind'];x,_,_,_=gll(degree);elements=[];points=[];index={}
    if kind not in ['pipe','hole'] or not np.isfinite(h_mm) or h_mm<=0:raise ValueError('Unsupported curved mesh request.')
    def append(mapped):
        ids=[]
        for point in mapped.reshape(-1,3)*.001:
            key=tuple(np.rint(point*1e12).astype(np.int64))
            if key not in index:index[key]=len(points);points.append(point)
            ids.append(index[key])
        elements.append(np.array(ids).reshape((degree+1,)*3))
    def nodes(a,b):return a+(x+1)*(b-a)/2
    if kind=='pipe':
        length=g['length_mm'];ri=g['inner_radius_mm'];ro=g['outer_radius_mm']
        axial=np.linspace(0,length,math.ceil(length/h_mm)+1);radial=np.linspace(ri,ro,math.ceil((ro-ri)/h_mm)+1)
        angles=np.linspace(0,2*np.pi,max(8,4*math.ceil(2*np.pi*ro/h_mm/4))+1)
        for a,b in zip(axial[:-1],axial[1:]):
            for r,s in zip(radial[:-1],radial[1:]):
                for t,u in zip(angles[:-1],angles[1:]):
                    xx,rr,tt=np.meshgrid(nodes(a,b),nodes(r,s),nodes(t,u),indexing='ij');append(np.stack([xx,rr*np.cos(tt),rr*np.sin(tt)],axis=-1))
    else:
        center=np.array([g['hole_x_mm'],g['hole_y_mm']]);radius=g['hole_radius_mm'];length=g['length_mm'];width=g['width_mm'];thick=g['thickness_mm']
        corners=np.array([[length,width],[0,width],[0,0],[length,0],[length,width]])
        theta=np.unwrap(np.arctan2(corners[:,1]-center[1],corners[:,0]-center[0]));outer_distance=np.linalg.norm(corners[:-1]-center,axis=1).max()
        # Shared graded radial layers keep neighbors conforming and resolve the hole.
        radial_count=max(2,math.ceil((outer_distance-radius)/h_mm));q=np.linspace(0,1,radial_count+1);radial=(np.exp(1.2*q)-1)/np.expm1(1.2)
        axial=np.linspace(0,thick,math.ceil(thick/h_mm)+1)
        for side in range(4):
            angles=np.linspace(theta[side],theta[side+1],max(2,math.ceil((theta[side+1]-theta[side])*outer_distance/h_mm))+1)
            for r,s in zip(radial[:-1],radial[1:]):
                for t,u in zip(angles[:-1],angles[1:]):
                    for a,b in zip(axial[:-1],axial[1:]):
                        rr,tt,zz=np.meshgrid(nodes(r,s),nodes(t,u),nodes(a,b),indexing='ij');cx=np.cos(tt);cy=np.sin(tt)
                        if side==0:distance=(width-center[1])/cy
                        elif side==1:distance=-center[0]/cx
                        elif side==2:distance=-center[1]/cy
                        else:distance=(length-center[0])/cx
                        distance=radius+rr*(distance-radius)
                        append(np.stack([center[0]+distance*cx,center[1]+distance*cy,zz],axis=-1))
    return prepare(points,elements,degree,scene['material']['rho_kg_m3'],h_mm)


def boundary_faces(mesh):
    faces={}
    for e,element in enumerate(mesh['elements']):
        for axis in range(3):
            for end in [0,-1]:
                selection=[slice(None)]*3;selection[axis]=end;face=element[tuple(selection)]
                key=tuple(sorted(map(int,[face[0,0],face[-1,0],face[-1,-1],face[0,-1]])))
                if key in faces:faces[key]=None
                else:faces[key]=(e,axis,end,face)
    return [v for v in faces.values() if v is not None]


def boundary_data(mesh,scene):
    source=scene['source'];face=source['face'];points=mesh['points'];force=np.zeros_like(points)
    if face not in ['xmin','xmax','ymin','ymax','zmin','zmax'] or source['radius_mm']!=0:raise ValueError('Curved reference currently requires a complete planar loading face.')
    axis='xyz'.index(face[0]);bound=points[:,axis].min() if face.endswith('min') else points[:,axis].max();area=0.;outer=boundary_faces(mesh)
    for e,local_axis,end,ids in outer:
        if not np.all(abs(points[ids,axis]-bound)<1e-12):continue
        selection=[slice(None)]*3;selection[local_axis]=end;jac=mesh['jacobian'][e][tuple(selection)];other=[i for i in range(3) if i!=local_axis]
        weights=np.linalg.norm(np.cross(jac[...,other[0]],jac[...,other[1]]),axis=-1)*np.outer(mesh['weights_1d'],mesh['weights_1d']);area+=weights.sum()
        np.add.at(force,ids.ravel(),(weights[...,None]*np.array(source['direction'])*source['amplitude_mpa']*1e6).reshape(-1,3))
    if area<=0:raise ValueError('No planar loading face; use a pipe x end.')
    fixed=np.array([],dtype=np.int32);fixed_face=scene['boundary']['fixed_face']
    if fixed_face!='none':
        axis='xyz'.index(fixed_face[0]);bound=points[:,axis].min() if fixed_face.endswith('min') else points[:,axis].max()
        boundary=[ids.ravel() for _,_,_,ids in outer if np.all(abs(points[ids,axis]-bound)<1e-12)]
        if not boundary:raise ValueError('Fixed boundary has no planar faces.')
        fixed=np.unique(np.concatenate(boundary))
    if np.any(force[fixed]):raise ValueError('Load overlaps fixed boundary.')
    return force,fixed,float(area)


def shape(mesh,local):
    x=mesh['gll'];rows=[];derivatives=[]
    for value in local:
        row=[];derivative=[]
        for i in range(len(x)):
            ids=[j for j in range(len(x)) if j!=i];den=np.prod(x[i]-x[ids]);row.append(np.prod(value-x[ids])/den)
            derivative.append(sum(np.prod(value-x[[k for k in ids if k!=j]]) for j in ids)/den)
        rows.append(row);derivatives.append(derivative)
    weights=np.einsum('i,j,k->ijk',*rows)
    gradient=np.stack([np.einsum('i,j,k->ijk',derivatives[0],rows[1],rows[2]),np.einsum('i,j,k->ijk',rows[0],derivatives[1],rows[2]),np.einsum('i,j,k->ijk',rows[0],rows[1],derivatives[2])],axis=-1)
    return weights.ravel(),gradient.reshape(-1,3)


def interpolation(mesh,point_mm,local_tolerance=1e-8):
    point=np.asarray(point_mm)*.001
    if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('Invalid receiver.')
    margin=.03*(mesh['bounds_max']-mesh['bounds_min'])+1e-12
    candidates=np.flatnonzero(np.all(point>=mesh['bounds_min']-margin,axis=1)&np.all(point<=mesh['bounds_max']+margin,axis=1))
    for e in candidates:
        ids=mesh['elements'][e].ravel();nodes=mesh['points'][ids];local=np.zeros(3)
        for _ in range(18):
            w,grad=shape(mesh,local);residual=w@nodes-point
            if np.linalg.norm(residual)<5e-16:
                if np.all(abs(local)<=1+local_tolerance):return ids,w
                break
            try:step=np.linalg.solve(nodes.T@grad,residual)
            except np.linalg.LinAlgError:break
            local-=step
            if np.max(abs(local))>4:break
    raise ValueError('Receiver is outside the curved material or in a void.')
