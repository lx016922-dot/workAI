"""One immutable local 3-D job. No remote API, shell expressions, or AI routing."""
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import signal
import time
import numpy as np
from .schema import require
from physics.scene import validate_scene
from .mesh import make_mesh,prepare_mesh,nodal_load,face_selection,locate,prepare_excitation
from .fem import solve

def now():return datetime.now(timezone.utc).isoformat()
def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False));temporary.replace(path)

def validate_mesh_reference(scene,record,scope):
    require(scope in ['time','controls'],'参考网格用途无效。')
    for key in ['geometry','material','boundary','acquisition','receivers']:
        require(scene[key]==record['scene'][key],'参考网格的几何、材料、边界、采样和接收点必须一致。')
    require(scene['numerics']['h_mm']==record['scene']['numerics']['h_mm'],'参考网格的目标尺寸必须一致。')
    expected=record['scene']
    require(scene.get('boundaries',[])==expected.get('boundaries',[]) and len(scene.get('sources',[]))==len(expected.get('sources',[])),'参考网格的边界条件和激励数量必须一致。')
    if scope=='time':
        for key in ['source','constitutive']:require(scene[key]==expected[key],'时间步对照必须保持物理场景不变。')
        require(scene.get('sources',[])==expected.get('sources',[]),'时间步对照必须保持全部激励不变。')
    else:
        require(scene['numerics']==expected['numerics'],'本构对照必须使用相同数值设置。')
        require(expected['constitutive']=='linear','五组本构对照的参考必须是线性基准。')
        a=dict(scene['source']);b=dict(expected['source']);amplitude=a.pop('amplitude_mpa');baseline=b.pop('amplitude_mpa')
        require(a==b and amplitude in [baseline,2*baseline],'本构对照仅允许切换本构及 A/2A 幅值。')
        for x,y in zip(scene.get('sources',[]),expected.get('sources',[])):
            a=dict(x);b=dict(y);require(a.pop('amplitude_mpa')==b.pop('amplitude_mpa')*amplitude/baseline and a==b,'本构对照的额外激励必须按相同比例缩放。')

def run(directory,mesh_reference=None,mesh_scope='time',output_precision='float32',time_budget_s=1800):
    directory=Path(directory);started=time.monotonic();status={'job_id':directory.name,'phase':'meshing','started_at_utc':now()}
    def publish(data):status.update(data,updated_at_utc=now(),elapsed_s=time.monotonic()-started);write_json(directory/'status.json',status)
    def timeout(*_):raise TimeoutError('计算超过当前时间预算；当前任务停止。')
    require(type(time_budget_s) is int and 1<=time_budget_s<=1800,'计算时间预算无效。')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(time_budget_s)
    publish({})
    try:
        raw=(directory/'scene.json').read_bytes();scene=validate_scene(json.loads(raw))
        sources={p.name:p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py'))}
        snapshot=directory/'source_snapshot';snapshot.mkdir(exist_ok=False)
        for name,data in sources.items():(snapshot/name).write_bytes(data)
        scene_sha=hashlib.sha256(raw).hexdigest()
        asset=None
        if scene['geometry']['kind']=='cad':
            from .cad import read_asset
            source,metadata=read_asset(directory.parents[2],scene['geometry']['asset_id'])
            copied=directory/source.name;copied.write_bytes(source.read_bytes());write_json(directory/'geometry-metadata.json',metadata);asset=(copied,metadata)
        if mesh_reference is None:mesh=make_mesh(scene,asset)
        else:
            reference=Path(mesh_reference);reference_bytes=(reference/'result.json').read_bytes()
            reference_status=json.loads((reference/'status.json').read_text())
            require(reference_status['phase']=='completed' and hashlib.sha256(reference_bytes).hexdigest()==reference_status['result_sha256'],'参考网格完成记录不匹配。')
            record=json.loads(reference_bytes)
            validate_mesh_reference(scene,record,mesh_scope)
            for name in ['fem.py','mesh.py','schema.py']:
                require(hashlib.sha256(sources[name]).hexdigest()==record['source_sha256'][name],'参考计算的物理实现版本不同。')
            with np.load(reference/'mesh.npz',allow_pickle=False) as cached:
                points=cached['points'];tetra=cached['tetra']
            require(np.array_equal(points*1000,record['nodes_mm']) and np.array_equal(tetra,record['tetra']),'参考网格与结果记录不匹配。')
            mesh=prepare_mesh(points,tetra)
        for rx in scene['receivers']:locate(mesh,rx['point_mm'])
        if scene.get('receiver_lines'):raise ValueError('网页三维任务暂不支持接收线；请使用 spectral3d.run、solid.run 或 sweep。')
        excitation=prepare_excitation(mesh,scene)
        if excitation is None:
            load,selected=nodal_load(mesh,scene['source'])
            fixed=np.array([],dtype=int)
            if scene['boundary']['fixed_face']!='none':fixed=np.unique(mesh['surface'][face_selection(mesh,scene['boundary']['fixed_face'])])
            if np.any(load[fixed]!=0):raise ValueError('激励与固定节点重叠；请调整端面/孔径。')
            fields,diagnostics=solve(scene,mesh,load,fixed,publish,output_precision)
        else:
            selected=excitation['selected']
            fields,diagnostics=solve(scene,mesh,None,None,publish,output_precision,excitation=excitation)
        publish({'phase':'exporting'})
        field_bytes=fields.astype('<f8' if output_precision=='float64' else '<f4',copy=False).tobytes();packed=gzip.compress(field_bytes,compresslevel=5,mtime=0)
        (directory/'field.bin.gz').write_bytes(packed)
        np.savez_compressed(directory/'mesh.npz',points=mesh['points'],tetra=mesh['tetra'],surface=mesh['surface'])
        manifest={'schema':'workai.hyperelastic.result.v1','job_id':directory.name,'scene':scene,'scene_sha256':scene_sha,
                  'engine':'workai_explicit_hyperelastic_3d','engine_scope':'experimental_numerical_solver_not_neural_surrogate',
                  'mesh_reference_job':Path(mesh_reference).name if mesh_reference else None,
                  'mesh_reference_scope':mesh_scope if mesh_reference else None,
                  'mesh_sha256':hashlib.sha256((directory/'mesh.npz').read_bytes()).hexdigest(),
                  'source_sha256':{name:hashlib.sha256(data).hexdigest() for name,data in sources.items()},
                  'nodes_mm':(mesh['points']*1000).tolist(),'tetra':mesh['tetra'].tolist(),'surface':mesh['surface'].tolist(),
                  'loaded_faces':np.flatnonzero(selected).tolist(),'load_area_mm2':float(mesh['area'][selected].sum()*1e6),
                  'times_us':(np.arange(len(fields))*scene['acquisition']['sample_step_us']).tolist(),'diagnostics':diagnostics,
                  'field':{'name':'field.bin.gz','layout':'time,node,xyz','dtype':'little_endian_'+output_precision,'unit':'m','shape':list(fields.shape),
                           'compressed_bytes':len(packed),'uncompressed_bytes':len(field_bytes),'sha256':hashlib.sha256(packed).hexdigest(),'uncompressed_sha256':hashlib.sha256(field_bytes).hexdigest()},
                  'completed_at_utc':now()}
        write_json(directory/'result.json',manifest)
        publish({'phase':'completed','result_sha256':hashlib.sha256((directory/'result.json').read_bytes()).hexdigest(),
                 'nodes':len(mesh['points']),'elements':len(mesh['tetra']),'frames':len(fields),'diagnostics':{k:v for k,v in diagnostics.items() if k!='energy_samples'}})
        print(json.dumps({'completed':True,'job_id':directory.name,'nodes':len(mesh['points']),'elements':len(mesh['tetra']),'seconds':time.monotonic()-started,'energy_balance_error':diagnostics['external_work_balance_relative']}),flush=True)
    except BaseException as error:
        publish({'phase':'failed','error':str(error)});raise
    finally:signal.alarm(0)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job-dir',required=True);args=parser.parse_args();run(args.job_dir)
