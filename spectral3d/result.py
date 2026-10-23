"""Browser result contract for a computed high-order displacement field."""
import gzip
import hashlib
import numpy as np


def surface_triangles(elements):
    faces={}
    for element in elements:
        for axis in range(3):
            for end in [0,-1]:
                face=np.take(element,end,axis=axis)
                key=tuple(sorted([face[0,0],face[-1,0],face[-1,-1],face[0,-1]]))
                if key in faces:faces.pop(key)
                else:faces[key]=face
    triangles=[]
    for face in faces.values():
        for i in range(len(face)-1):
            for j in range(len(face)-1):
                a,b,c,d=face[i,j],face[i+1,j],face[i+1,j+1],face[i,j+1]
                triangles.extend([[a,b,c],[a,c,d]])
    return np.asarray(triangles,dtype=np.int64)


def result_record(job_id,scene,mesh,computed):
    field=computed['fields_m'];raw=np.asarray(field,dtype='<f8').tobytes()
    if not np.isfinite(field).all() or len(raw)>192*1024**2 or len(mesh['points'])>30000 or mesh['elements'].size>100000:
        raise ValueError('高阶位移场超出浏览器大小限制。')
    surface=surface_triangles(mesh['elements'])
    if len(surface)>50000:raise ValueError('高阶表面网格超过显示预算。')
    packed=gzip.compress(raw,compresslevel=5,mtime=0);points=mesh['points']*1000
    axis='xyz'.index(scene['source']['face'][0]);bound=points[:,axis].min() if scene['source']['face'].endswith('min') else points[:,axis].max()
    loaded=np.flatnonzero(np.all(abs(points[surface,axis]-bound)<1e-8,axis=1))
    sha=lambda value:hashlib.sha256(value).hexdigest()
    record={'schema':'workai.spectral_reference.result.v1','engine':'workai_spectral_reference_3d',
            'job_id':job_id,'response_part':'full','scene':scene,'accepted_for_solver_routing':False,
            'engine_scope':'numerical_solution_not_neural_prediction','nodes_mm':points.tolist(),
            'tetra':[],'surface':surface.tolist(),'loaded_faces':loaded.tolist(),
            'times_us':computed['times_us'].tolist(),'diagnostics':computed['diagnostics'],
            'spectral':{'degree':int(mesh['degree']),'gll':mesh['gll'].tolist(),
                        'elements':mesh['elements'].reshape(len(mesh['elements']),-1).tolist(),
                        **({'mapping_kind':'isoparametric'} if 'mapping_kind' in mesh else
                           {'mapping_kind':'cartesian','origins_mm':(mesh['origin']*1000).tolist(),'sizes_mm':(mesh['size']*1000).tolist()})},
            'field':{'layout':'time,node,xyz','dtype':'little_endian_float64','unit':'m','shape':list(field.shape),
                     'compressed_bytes':len(packed),'uncompressed_bytes':len(raw),
                     'sha256':sha(packed),'uncompressed_sha256':sha(raw)}}
    return record,packed
