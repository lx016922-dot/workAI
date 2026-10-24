"""Gmsh/OpenCASCADE tetrahedra, actual boundary faces, no voxel substitutes."""
import numpy as np
from .schema import require

def make_mesh(scene,asset=None):
    import gmsh
    g=scene['geometry'];h=scene['numerics']['h_mm']
    gmsh.initialize(['workai','-nopopup']);gmsh.option.setNumber('General.Terminal',0)
    try:
        gmsh.model.add('aluminum')
        o=gmsh.model.occ
        if g['kind']=='cad':
            from .cad import import_cad
            require(asset is not None,'导入模型缺少已验证的几何文件。')
            import_cad(gmsh,asset[0],asset[1],g['stl_unit_mm'])
        elif g['kind']=='pipe':
            outer=o.addCylinder(0,0,0,g['length_mm'],0,0,g['outer_radius_mm'])
            inner=o.addCylinder(-.01,0,0,g['length_mm']+.02,0,0,g['inner_radius_mm'])
            o.cut([(3,outer)],[(3,inner)])
        else:
            box=o.addBox(0,0,0,g['length_mm'],g['width_mm'],g['thickness_mm'])
            if g['kind']=='hole':
                hole=o.addCylinder(g['hole_x_mm'],g['hole_y_mm'],-.01,0,0,g['thickness_mm']+.02,g['hole_radius_mm'])
                o.cut([(3,box)],[(3,hole)])
            elif g['kind']=='notch':
                slot=o.addBox(g['notch_x_mm']-g['notch_width_mm']/2,-.01,g['thickness_mm']-g['notch_depth_mm'],g['notch_width_mm'],g['width_mm']+.02,g['notch_depth_mm']+.01)
                o.cut([(3,box)],[(3,slot)])
        if g['kind']!='cad':o.synchronize()
        gmsh.option.setNumber('General.NumThreads',2)
        gmsh.option.setNumber('Mesh.MeshSizeMin',h*.5);gmsh.option.setNumber('Mesh.MeshSizeMax',h)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',12)
        gmsh.option.setNumber('Mesh.ElementOrder',1)
        gmsh.option.setNumber('Mesh.Algorithm3D',1)
        gmsh.model.mesh.generate(3);gmsh.model.mesh.optimize('Netgen')
        tags,points,_=gmsh.model.mesh.getNodes();types,_,connectivity=gmsh.model.mesh.getElements(3)
        require(list(types)==[4],'网格必须只含一阶四面体。')
        index={int(tag):i for i,tag in enumerate(tags)}
        tetra=np.array([index[int(tag)] for tag in connectivity[0]],dtype=np.int32).reshape(-1,4)
        points=np.asarray(points).reshape(-1,3)*.001
        used=np.unique(tetra);remap=np.full(len(points),-1,dtype=np.int32);remap[used]=np.arange(len(used))
        points=points[used];tetra=remap[tetra]
        require(len(tetra)<=scene['numerics']['max_elements'],f'实际网格 {len(tetra)} 单元超过预算；请缩小结构或调整频率与网格。')
        require(len(points)<=12000,'实际节点数超过 12000；请先缩小当前验证工况。')
        return prepare_mesh(points,tetra)
    finally:gmsh.finalize()

def prepare_mesh(points,tetra):
    points=np.asarray(points,dtype=np.float64);tetra=np.asarray(tetra,dtype=np.int32).copy()
    dm=np.transpose(points[tetra[:,1:]]-points[tetra[:,0,None]],(0,2,1))
    det=np.linalg.det(dm);negative=det<0
    tetra[negative,1],tetra[negative,2]=tetra[negative,2].copy(),tetra[negative,1].copy()
    dm=np.transpose(points[tetra[:,1:]]-points[tetra[:,0,None]],(0,2,1));det=np.linalg.det(dm)
    require(np.all(det>1e-24),'网格存在零体积或极度退化单元。')
    inverse=np.linalg.inv(dm);grad=np.empty((len(tetra),4,3));grad[:,1:]=inverse;grad[:,0]=-inverse.sum(axis=1)
    volume=det/6
    local=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
    faces=tetra[:,local].reshape(-1,3)
    _,idx,count=np.unique(np.sort(faces,axis=1),axis=0,return_index=True,return_counts=True)
    require(count.max()<=2,'四面体连接不是流形网格。')
    surface=faces[idx[count==1]]
    facevec=np.cross(points[surface[:,1]]-points[surface[:,0]],points[surface[:,2]]-points[surface[:,0]])
    area=np.linalg.norm(facevec,axis=1)/2
    # h_alt is the smallest altitude; safer than a longest-edge CFL estimate.
    facearea=[]
    for loc in local:
        a,b,c=np.transpose(points[tetra[:,loc]],(1,0,2));facearea.append(np.linalg.norm(np.cross(b-a,c-a),axis=1)/2)
    min_altitude=np.min(3*volume[None,:]/np.asarray(facearea))
    require(min_altitude>1e-8,'最小四面体高度过小，需要修复网格。')
    return {'points':np.ascontiguousarray(points),'tetra':np.ascontiguousarray(tetra),'grad':np.ascontiguousarray(grad),
            'volume':volume,'surface':surface,'area':area,'normal':facevec/(2*area[:,None]),'min_altitude':float(min_altitude)}

def face_selection(mesh,name):
    axis='xyz'.index(name[0]);end=0 if name.endswith('min') else 1
    points=mesh['points'];bound=[points[:,axis].min(),points[:,axis].max()][end]
    tol=max(np.ptp(points,axis=0).max()*1e-7,1e-10)
    mask=np.all(abs(points[mesh['surface'],axis]-bound)<tol,axis=1)
    require(mask.any(),f'{name} 没有可加载的平面端面；请为管选择 x 端面。')
    return mask

def nodal_load(mesh,source):
    center=np.asarray(source['center_mm'])*.001
    if source['face']=='surface_patch':
        delta=mesh['points'][mesh['surface']].mean(axis=1)-center
        normal=np.array(source['patch_normal']);normal/=np.linalg.norm(normal)
        selected=(np.linalg.norm(delta,axis=1)<=source['radius_mm']*.001)&((mesh['normal']@normal)>.5)
    else:selected=face_selection(mesh,source['face'])
    if source['radius_mm']>0 and source['face']!='surface_patch':
        axis='xyz'.index(source['face'][0]);delta=mesh['points'][mesh['surface']].mean(axis=1)-center
        delta[:,axis]=0;selected &= np.linalg.norm(delta,axis=1)<=source['radius_mm']*.001
    require(selected.any(),'激励孔径内没有表面三角形；请扩大孔径或细化网格。')
    force=np.zeros_like(mesh['points']);indices=np.flatnonzero(selected);weights=mesh['area'][selected]/3
    if source.get('profile'):
        # Version-2 spatial profile, evaluated at the centroids of the selected triangles.
        from physics.loads import region_weights
        centroids=mesh['points'][mesh['surface'][indices]].mean(axis=1)
        profile=region_weights(centroids,dict(source,radius_mm=0.),None if source['face']=='surface_patch' else 'xyz'.index(source['face'][0]))
        keep=profile!=0;require(keep.any(),'载荷分布在加载区域内处处为零；请检查梳状参数。')
        indices=indices[keep];weights=weights[keep]*profile[keep];selected=np.zeros_like(selected);selected[indices]=True
    for j in range(3):
        np.add.at(force,mesh['surface'][indices,j],weights[:,None]*np.asarray(source['direction'])*source['amplitude_mpa']*1e6)
    return force,selected

def prepare_excitation(mesh,scene):
    """Version-2 loads, constraints and damping on a tetrahedral mesh; None for version 1."""
    import math
    from physics import scene as scene_v2
    from physics.boundaries import build_constraints,build_damping
    if not scene_v2.is_v2(scene):return None
    loads=[];rows=[];selected=np.zeros(len(mesh['surface']),dtype=bool)
    for src in scene_v2.sources(scene):
        force,mask=nodal_load(mesh,src);loads.append((force,src));selected|=mask
        rows.append({'name':src.get('name','primary'),'face':src['face'],'loaded_area_mm2':float(mesh['area'][mask].sum()*1e6),'triangles':int(mask.sum())})
    constrained,constraint_rows=build_constraints(mesh['points'],scene['boundary']['fixed_face'],scene_v2.boundaries(scene))
    total=sum(force for force,_ in loads)
    require(not np.any(total[constrained]!=0),'激励作用在固定或对称约束的位移分量上；请改变加载方向或边界类型。')
    material=scene['material'];speed=math.sqrt((material['lambda_pa']+2*material['mu_pa'])/material['rho_kg_m3'])
    alpha,damping_rows=build_damping(mesh['points'],scene_v2.boundaries(scene),speed)
    return {'loads':loads,'constrained':constrained,'alpha':alpha,'selected':selected,'area_m2':float(mesh['area'][selected].sum()),
            'summary':{'excitation_model':'multi_source_v2','sources':rows,'constraints':constraint_rows,'damping':damping_rows}}

def locate(mesh,point_mm):
    p=np.asarray(point_mm)*.001;v=mesh['points'][mesh['tetra']]
    candidates=np.flatnonzero(np.all(p>=v.min(axis=1)-1e-10,axis=1)&np.all(p<=v.max(axis=1)+1e-10,axis=1))
    for index in candidates:
        weights=mesh['grad'][index,1:]@(p-v[index,0]);weights=np.r_[1-weights.sum(),weights]
        if weights.min()>=-1e-7 and weights.max()<=1+1e-7:return int(index),np.clip(weights,0,1)/np.clip(weights,0,1).sum()
    raise ValueError('接收点不在实际实体网格内（孔洞和管内腔不属于材料）。')
