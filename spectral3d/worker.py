"""Bounded interactive high-order solve; no AI or remote computation."""
import argparse,json,signal,time
from pathlib import Path
import numpy as np
from physics.scene import validate_scene,is_v2,spectral_admissible,summary
from .campaign import atomic,now,digest
from .mesh import make_mesh,boundary_data,interpolation
from .loads import prepare_excitation,loaded_surface_faces
from .result import result_record,surface_triangles


def run(directory):
    directory=Path(directory);started=time.monotonic();status={'job_id':directory.name,'phase':'meshing','started_at_utc':now()}
    def publish(**values):
        status.update(values,updated_at_utc=now(),elapsed_s=time.monotonic()-started);atomic(directory/'status.json',status)
    def timeout(*_):raise TimeoutError('高阶计算超过 30 分钟预算，任务已停止。')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1800)
    try:
        scene_path=directory/'scene.json';raw=json.loads(scene_path.read_text());settings=json.loads((directory/'settings.json').read_text())
        if set(settings)!={'degree'} or type(settings['degree']) is not int or settings['degree'] not in [3,4]:raise ValueError('高阶阶次应为 3 或 4。')
        h=raw['numerics']['h_mm']
        if type(h) not in [int,float] or not np.isfinite(h) or not .02<=h<=100:raise ValueError('高阶单元尺寸必须在 0.02–100 mm 范围内。')
        raw['numerics']['h_mm']=max(.02,h/settings['degree']);scene=validate_scene(raw);scene['numerics']['h_mm']=h
        if scene['geometry']['kind'] not in ['plate','pipe','hole','notch']:raise ValueError('高阶计算支持参数化板、管、通孔和缺口。')
        if is_v2(scene):spectral_admissible(scene)
        elif scene['source']['face'] not in ['xmin','xmax'] or scene['source']['radius_mm']!=0 or scene['boundary']['fixed_face'] not in ['none','xmin','xmax']:
            raise ValueError('当前高阶计算使用完整 X 端面加载，固定面可选另一 X 端面。')
        hashes={};source=Path(__file__).resolve().parents[1]
        for module in ['solid','spectral3d']:
            target=directory/'source_snapshot'/module;target.mkdir(parents=True,exist_ok=False)
            for file in (source/module).glob('*.py'):
                (target/file.name).write_bytes(file.read_bytes());hashes[module+'/'+file.name]=digest(file)
        publish();mesh=make_mesh(scene,settings['degree'],scene['numerics']['h_mm'])
        frames=round(scene['acquisition']['duration_us']/scene['acquisition']['sample_step_us'])+1
        if len(mesh['points'])>30000 or len(mesh['elements'])>scene['numerics']['max_elements'] or mesh['elements'].size>100000 or len(mesh['points'])*frames*24>192*1024**2 or len(surface_triangles(mesh['elements']))>50000:
            raise ValueError('高阶网格或场存储超出预算；请增大网格尺寸或缩短记录。')
        excitation=prepare_excitation(mesh,scene)
        if excitation is None:boundary_data(mesh,scene)
        for receiver in scene['receivers']:interpolation(mesh,receiver['point_mm'])
        np.savez_compressed(directory/'mesh.npz',**mesh)
        def progress(value):publish(phase='running',step=value.get('frame',0),steps_total=frames-1)
        publish(phase='running',step=0,steps_total=frames-1,nodes=len(mesh['points']))
        if scene['geometry']['kind']=='hole':
            from .implicit import solve
            computed=solve(scene,mesh,substeps=4,keep_fields=True,callback=progress,excitation=excitation)
        else:
            from .solver import solve
            computed=solve(scene,mesh,cfl=scene['numerics']['cfl'],keep_fields=True,callback=progress,excitation=excitation)
        publish(phase='exporting',step=frames-1)
        record,packed=result_record(directory.name,scene,mesh,computed)
        if excitation is not None:record.update(loaded_faces=loaded_surface_faces(record['surface'],excitation['loads']),excitation=summary(scene))
        record.update(scene_sha256=digest(scene_path),settings_sha256=digest(directory/'settings.json'),
                      mesh_sha256=digest(directory/'mesh.npz'),source_sha256=hashes)
        np.savez_compressed(directory/'probes.npz',probes_m=computed['probes_m'],times_us=computed['times_us'])
        (directory/'field.bin.gz').write_bytes(packed);atomic(directory/'result.json',record)
        publish(phase='completed',result_sha256=digest(directory/'result.json'),accepted_for_solver_routing=False)
    except BaseException as error:
        publish(phase='failed',error=str(error));raise
    finally:signal.alarm(0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job-dir',type=Path,required=True);run(parser.parse_args().job_dir)
