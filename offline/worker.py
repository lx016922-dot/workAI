"""One local inference job. No remote service or numerical solver fallback."""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from graphwave.predict import predict
from spectral3d.mesh import boundary_data, interpolation
from spectral3d.feature_mesh import make_mesh
from spectral3d.result import result_record, surface_triangles
from .domain import MODEL, PARTS, V2, digest, model_info, validate_request
from graphwave.long_rollout import rollout
from .recording import preview, display_preview


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for attempt in range(40):   # Windows refuses the replace while the application is reading the file: retry for a moment
        try:
            return temp.replace(path)
        except PermissionError:
            if attempt == 39: raise
            time.sleep(.05)


def export_results(directory, scene, mesh, fields, diagnostics, info, times=None, recording=None, extra=None):
    if times is None:
        times = np.arange(fields[0].shape[0]) * scene['acquisition']['sample_step_us']
    exports = {}
    for part in PARTS:
        field = {'full': lambda: fields[2], 'linear': lambda: fields[0], 'stvk': lambda: fields[1],
                 'geometric': lambda: fields[1] - fields[0], 'material': lambda: fields[2] - fields[1]}[part]()
        d = dict(diagnostics[{'linear': 0, 'stvk': 1}.get(part, 2)])
        d.update(output_dtype='float64', internal_step_us=d['layer_step_us'])
        record, packed = result_record(directory.name, scene, mesh, {'fields_m': field, 'times_us': times, 'diagnostics': d})
        record.update(schema='workai.offline.graphwave.result.v1', engine='workai_graphwave_offline',
                      engine_scope='continued_graph_network_long_record' if recording else 'original_frozen_graph_network', response_part=part,
                      neural_model={'weights_sha256': info['weights_sha256'], 'numerical_correction': False,
                                    'neural_forward_only': True, 'original_acceptance_restored': info['original_acceptance_restored']},
                      controls=diagnostics)
        if extra:
            record.update(extra)
        if recording:
            record['recording'] = {k: recording[k] for k in ('schema','duration_us','sample_step_us','frame_count','fingerprint','mesh_sha256')}
            record['recording'].update(animation_only=True, analysis_source='complete_float64_chunks')
        (directory / (part + '.bin.gz')).write_bytes(packed)
        write_json(directory / (part + '.json'), record)
        exports[part] = {'result_sha256': digest(directory / (part + '.json')), 'field_sha256': digest(directory / (part + '.bin.gz'))}
    write_json(directory / 'exports.json', exports)


def export_display(directory, scene, mesh, display, parts, diagnostics, info, times, recording, extra):
    """The five responses on the display mesh (large models and compact records): float32 animation frames, receivers from the record."""
    from spectral3d.result import display_record
    exports = {}; compact = recording.get('record_mode') == 'compact'
    for part in PARTS:
        d = dict(diagnostics[{'linear': 0, 'stvk': 1}.get(part, 2)])
        d.update(output_dtype='float32', internal_step_us=d['layer_step_us'])
        record, packed = display_record(directory.name, scene, mesh, display, parts[part], times, d, extra['loaded_faces'])
        record.update(schema='workai.offline.graphwave.result.v1', engine='workai_graphwave_offline', engine_scope='continued_graph_network_long_record', response_part=part,
                      neural_model={'weights_sha256': info['weights_sha256'], 'numerical_correction': False,
                                    'neural_forward_only': True, 'original_acceptance_restored': info['original_acceptance_restored']},
                      controls=diagnostics)
        record.update(extra)
        record['recording'] = {k: recording[k] for k in ('schema', 'duration_us', 'sample_step_us', 'frame_count', 'fingerprint', 'mesh_sha256')}
        record['recording'].update(animation_only=True, analysis_source='receiver_float64_probes' if compact else 'complete_float64_chunks',
                                   record_mode='compact' if compact else 'full', time_plan=recording.get('time_plan'))
        if compact: record['recording'].update(receivers=recording['receivers'], snapshot_frames=recording['snapshot_frames'])
        (directory / (part + '.bin.gz')).write_bytes(packed)
        write_json(directory / (part + '.json'), record)
        exports[part] = {'result_sha256': digest(directory / (part + '.json')), 'field_sha256': digest(directory / (part + '.bin.gz'))}
    write_json(directory / 'exports.json', exports)


def run_coarse(directory, scene, info, publish, name):
    """Fast engine: the learned coarse-graph operator on its own coarse grid (short records)."""
    from .coarse_engine import run as coarse_run
    publish(phase='running', engine='coarse', coarse_model=name, controls_completed=0, message='粗网格算子：首次运行需要编译，随后计算三组控制')
    def progress(d):
        if not d['energy_balance_within_1_percent']:
            raise ValueError('粗网格输出能量收支未通过 1% 检查。')
        publish(controls_completed=['linear', 'stvk', 'murnaghan'].index(d['law']) + 1, nodes=d['nodes'], message=d['law'] + ' 控制完成')
    mesh, fields, diagnostics, times, extra = coarse_run(scene, callback=progress, name=name)
    if len(surface_triangles(mesh['elements'])) > 50000:
        raise ValueError('网格超过离线显示预算。')
    from spectral3d.loads import loaded_surface_faces, prepare_excitation
    from physics.scene import summary
    extra.update(loaded_faces=loaded_surface_faces(surface_triangles(mesh['elements']), prepare_excitation(mesh, scene)['loads']), excitation=summary(scene), input_layer='v2_multi_source')
    forbidden = ['spectral3d.solver', 'spectral3d.curved_solver', 'spectral3d.implicit', 'solid.fem']
    if any(name in sys.modules for name in forbidden):
        raise RuntimeError('预测进程意外加载了数值求解模块。')
    publish(phase='exporting', message='正在保存动画及五种响应')
    export_results(directory, scene, mesh, fields, diagnostics, dict(info, weights_sha256=digest(MODEL / 'weights.npz')), times, None, extra)
    publish(phase='completed', message='粗网格算子计算完成', controls_completed=3, numerical_correction=False)


def run(directory):
    directory = Path(directory)
    start = time.monotonic()
    status = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
    def publish(**values):
        status.update(values, elapsed_s=time.monotonic() - start, updated_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(directory / 'status.json', status)
    try:
        request = validate_request(json.loads((directory / 'request.json').read_text(encoding='utf-8')))
        info = model_info()
        scene = request['scene']
        if request.get('engine') == 'coarse':
            return run_coarse(directory, scene, info, publish, request['coarse_model'])
        network_v2 = scene.get('schema') == V2
        is_long = network_v2 or scene['acquisition']['duration_us'] >= 5000
        if is_long and 'long_model' not in info:
            raise ValueError('软件包缺少长记录模型，请使用完整的新版本。')
        weight_path = MODEL / (info['long_model']['weights_file'] if is_long else 'weights.npz')
        with np.load(weight_path, allow_pickle=False) as saved:
            weights = dict(saved)
        publish(phase='meshing', message='正在生成几何连接图')
        mesh = make_mesh(scene, request['degree'], scene['numerics']['h_mm'])
        extra = None; browser_full = True; record = 'full'
        if network_v2:
            # The machine limits the model, not the display: a large model keeps a compact record and is shown on a display mesh.
            from spectral3d.display import surface_faces
            from spectral3d.loads import prepare_excitation, loaded_surface_faces
            from physics.scene import summary
            from .resources import BROWSER_ENTRIES, BROWSER_NODES, BROWSER_TRIANGLES, memory_problem, record_mode
            frames = round(scene['acquisition']['duration_us'] / scene['acquisition']['sample_step_us']) + 1
            record = record_mode(request.get('record', 'auto'), frames, len(mesh['points']))
            problem = memory_problem(len(mesh['points']), record)
            if problem:
                raise ValueError(f'模型太大：{problem}。请增大单元尺寸、缩小模型或使用对称面。')
            browser_full = (record == 'full' and len(mesh['points']) <= BROWSER_NODES and mesh['elements'].size <= BROWSER_ENTRIES and
                            len(surface_faces(mesh['elements'])) * 2 * int(mesh['degree']) ** 2 <= BROWSER_TRIANGLES)
            excitation = prepare_excitation(mesh, scene)
            extra = {'excitation': summary(scene), 'input_layer': 'v2_multi_source', 'engine_scope': 'graph_network_v2_input_layer_record'}
            if browser_full: extra['loaded_faces'] = loaded_surface_faces(surface_triangles(mesh['elements']), excitation['loads'])
        else:
            if len(mesh['points']) > 30000 or mesh['elements'].size > 100000 or len(mesh['points']) * 281 * 24 > 192 * 1024**2 or len(surface_triangles(mesh['elements'])) > 50000:
                raise ValueError('网格超过离线显示预算，请增大网格尺寸。')
            boundary_data(mesh, scene)
        for rx in scene['receivers']:
            interpolation(mesh, rx['point_mm'])
        publish(phase='running', nodes=len(mesh['points']), controls_completed=0,
                duration_us=scene['acquisition']['duration_us'] if is_long else None, resumable=bool(is_long),
                **({'record_mode': record} if network_v2 else {}),
                message='首次运行需要编译网络，随后计算三组控制')
        def progress(d):
            if not d['energy_balance_within_1_percent']:
                raise ValueError('网络输出能量收支未通过 1% 检查。')
            publish(controls_completed=['linear', 'stvk', 'murnaghan'].index(d['law']) + 1, message=d['law'] + ' 控制完成')
        if is_long:
            temporal_path = MODEL / info['long_model']['time_weights_file']
            temporal = json.loads(temporal_path.read_text(encoding='utf-8'))
            resumed = time.monotonic(); first = {}
            def long_progress(m):
                # Remaining time from the samples computed in this run (a resumed record starts part-way).
                first.setdefault('frames', m['completed_frames'] - m['chunks'][-1]['frame_count'])
                done = m['completed_frames'] - first['frames']; left = m['frame_count'] - m['completed_frames']
                plan = m.get('time_plan') or {}
                publish(completed_time_us=m['completed_time_us'],duration_us=m['duration_us'], remaining_s=(time.monotonic() - resumed) * left / max(done, 1),
                        **({'time_plan': {k: plan.get(k) for k in ('scheme', 'time_order', 'layer_step_us', 'evaluations_per_control', 'frozen_evaluations_per_control', 'reason')}} if plan else {}),
                        message=f"已保存 {m['completed_time_us']/1000:.3f} / {m['duration_us']/1000:g} ms {'精简' if m.get('record_mode') == 'compact' else '完整'}记录")
                if (directory / 'pause.request').exists() and m['phase'] != 'completed':
                    raise InterruptedError('已保存检查点，可继续计算。')
            identity = {'weights_sha256':digest(weight_path),'time_weights_sha256':digest(temporal_path),
                        'input_sha256':digest(directory/'request.json')}
            if network_v2:
                from graphnet.model import load as load_network
                from graphnet.rollout import rollout as network_rollout
                network = load_network(request['degree'], long_record=True)
                if network['weights_sha256'] != identity['weights_sha256'] or network['time_weights_sha256'] != identity['time_weights_sha256']:
                    raise ValueError('v2 输入层读取的权重与软件包长记录模型不一致。')
                finished = network_rollout(scene, mesh, network, directory / 'record', identity, callback=long_progress,
                                           time_plan=request.get('time_stepping', 'auto'), record=record)
                if directory.parent.name == 'jobs':   # the application's data directory: remember this machine's rate for later estimates
                    from .resources import save_throughput
                    save_throughput(directory.parent.parent, len(mesh['points']), finished['time_plan']['evaluations_per_control'], finished['elapsed_s'])
            else:
                rollout(scene, mesh, weights, temporal, directory / 'record', identity, callback=long_progress)
            if browser_full: recording,times,fields,diagnostics = preview(directory/'record')
            else:
                from spectral3d.display import loaded_triangles
                recording, times, parts, diagnostics, display = display_preview(directory / 'record')
                extra['loaded_faces'] = loaded_triangles(display, excitation['loads'])
            if not all(d['energy_balance_within_1_percent'] for d in diagnostics):
                raise ValueError('长记录能量收支未通过 1% 检查；完整原始记录已保留供检查。')
            info = dict(info, weights_sha256=digest(weight_path))
        else:
            fields, diagnostics = predict(scene, mesh, weights, callback=progress)
            times = recording = None
        forbidden = ['spectral3d.solver', 'spectral3d.curved_solver', 'spectral3d.implicit', 'solid.fem']
        if any(name in sys.modules for name in forbidden):
            raise RuntimeError('预测进程意外加载了数值求解模块。')
        publish(phase='exporting', message='正在保存动画及五种响应')
        if browser_full: export_results(directory, scene, mesh, fields, diagnostics, info, times, recording, extra)
        else: export_display(directory, scene, mesh, display, parts, diagnostics, info, times, recording, extra)
        publish(phase='completed', message='本机预测完成', controls_completed=3, remaining_s=0,
                weights_sha256=info['weights_sha256'], numerical_correction=False)
    except InterruptedError as error:
        publish(phase='paused',message=str(error))
    except Exception as error:
        message = str(error)
        if 'strain/J/finite-energy guard' in message:  # the frozen kernels' guard text (short records, coarse engine)
            from graphnet.rollout import GUARD_MESSAGE
            message = GUARD_MESSAGE + '。'
        publish(phase='failed', message=message)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True, type=Path)
    run(parser.parse_args().job_dir)
