"""Bounded STEP/STL single-solid import. Input is geometry data, never code."""
import hashlib,json,re,struct
from pathlib import Path
import numpy as np
from .schema import require

def read_asset(root,asset_id):
    require(isinstance(asset_id,str) and re.fullmatch(r'asset-[a-f0-9]{64}',asset_id) is not None,'几何文件编号无效。')
    directory=Path(root)/'solid/assets'/asset_id
    metadata=json.loads((directory/'metadata.json').read_text())
    require(metadata['format'] in ['step','stl'],'仅支持 STEP/STL。')
    path=directory/('geometry.'+metadata['format']);require(path.is_file() and not path.is_symlink() and path.stat().st_size<=8*1024**2,'几何文件无效或过大。')
    raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==metadata['sha256']==asset_id[6:],'几何文件摘要不匹配。')
    return path,metadata

def stl_is_closed(raw):
    require(len(raw)<=8*1024**2,'STL 超过 8 MiB。')
    count=struct.unpack('<I',raw[80:84])[0] if len(raw)>=84 else 0
    if 84+count*50==len(raw):
        require(4<=count<=100000,'STL 需要 4–100000 个三角形。')
        triangles=np.array([struct.unpack_from('<9f',raw,84+i*50+12) for i in range(count)]).reshape(-1,3,3)
    else:
        text=raw.decode('ascii');values=re.findall(r'\bvertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)',text)
        require(len(values)%3==0 and 12<=len(values)<=300000,'ASCII STL 三角形无效。')
        triangles=np.array(values,dtype=float).reshape(-1,3,3)
    require(np.isfinite(triangles).all(),'STL 含无效坐标。')
    extent=float(np.ptp(triangles.reshape(-1,3),axis=0).max());require(1e-9<extent<1e7,'STL 尺寸无效。')
    quantized=np.rint((triangles-triangles.min(axis=(0,1)))/(extent*1e-9)).astype(np.int64)
    _,inverse=np.unique(quantized.reshape(-1,3),axis=0,return_inverse=True);faces=inverse.reshape(-1,3)
    require(all(len(set(row))==3 for row in faces),'STL 存在退化三角形。')
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);canonical=np.sort(edges,axis=1)
    unique,inv,counts=np.unique(canonical,axis=0,return_inverse=True,return_counts=True)
    require(np.all(counts==2),'STL 必须闭合且无非流形边；不会自动填补孔口。')
    signs=np.where(edges[:,0]<edges[:,1],1,-1);orientation=np.bincount(inv,weights=signs,minlength=len(unique))
    require(np.all(orientation==0),'STL 三角形朝向不一致，请在 CAD 软件中修复。')
    return len(faces)

def import_cad(gmsh,path,metadata,scale):
    if metadata['format']=='step':
        gmsh.option.setString('Geometry.OCCTargetUnit','MM')
        gmsh.model.occ.importShapes(str(path));gmsh.model.occ.synchronize()
        volumes=gmsh.model.getEntities(3);require(len(volumes)==1,'STEP 需要一个封闭实体；请先在 CAD 中合并装配体。')
        require(scale==1,'STEP 按文件单位转换为 mm，不接受额外 STL 单位缩放。')
    else:
        stl_is_closed(path.read_bytes());gmsh.merge(str(path))
        if scale!=1:gmsh.model.mesh.affineTransform([scale,0,0,0,0,scale,0,0,0,0,scale,0])
        gmsh.model.mesh.classifySurfaces(np.pi/4,True,True,np.pi)
        gmsh.model.mesh.createGeometry()
        surfaces=[tag for _,tag in gmsh.model.getEntities(2)];require(bool(surfaces),'STL 未形成可参数化曲面。')
        loop=gmsh.model.geo.addSurfaceLoop(surfaces);gmsh.model.geo.addVolume([loop]);gmsh.model.geo.synchronize()
    bounds=gmsh.model.getBoundingBox(-1,-1);extent=np.array(bounds[3:])-bounds[:3]
    require(np.all(extent>.001) and np.all(extent<=1000),'导入实体包围盒需为 0.001–1000 mm；请核对单位。')
    return list(bounds)
