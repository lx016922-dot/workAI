"""One local inference job. No remote service or numerical solver fallback."""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from graphwave.predict import predict
from spectral3d.mesh import make_mesh, boundary_data, interpolation
from spectral3d.result import result_record, surface_triangles
from .domain import MODEL, PARTS, digest, model_info, validate_request
from graphwave.long_rollout import rollout
from .recording import preview


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def export_results(directory, scene, mesh, fields, diagnostics, info, times=None, recording=None):
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
        if recording:
            record['recording'] = {k: recording[k] for k in ('schema','duration_us','sample_step_us','frame_count','fingerprint','mesh_sha256')}
            record['recording'].update(animation_only=True, analysis_source='complete_float64_chunks')
        (directory / (part + '.bin.gz')).write_bytes(packed)
        write_json(directory / (part + '.json'), record)
        exports[part] = {'result_sha256': digest(directory / (part + '.json')), 'field_sha256': digest(directory / (part + '.bin.gz'))}
    write_json(directory / 'exports.json', exports)


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
        is_long = request['scene']['acquisition']['duration_us'] >= 5000
        if is_long and 'long_model' not in info:
            raise ValueError('软件包缺少长记录模型，请使用完整的新版本。')
        weight_path = MODEL / (info['long_model']['weights_file'] if is_long else 'weights.npz')
        with np.load(weight_path, allow_pickle=False) as saved:
            weights = dict(saved)
        scene = request['scene']
        publish(phase='meshing', message='正在生成几何连接图')
        mesh = make_mesh(scene, request['degree'], scene['numerics']['h_mm'])
        if len(mesh['points']) > 30000 or mesh['elements'].size > 100000 or len(mesh['points']) * 281 * 24 > 192 * 1024**2 or len(surface_triangles(mesh['elements'])) > 50000:
            raise ValueError('网格超过离线显示预算，请增大网格尺寸。')
        boundary_data(mesh, scene)
        for rx in scene['receivers']:
            interpolation(mesh, rx['point_mm'])
        publish(phase='running', nodes=len(mesh['points']), controls_completed=0,
                duration_us=scene['acquisition']['duration_us'] if is_long else None,
                message='首次运行需要编译网络，随后计算三组控制')
        def progress(d):
            if not d['energy_balance_within_1_percent']:
                raise ValueError('网络输出能量收支未通过 1% 检查。')
            publish(controls_completed=['linear', 'stvk', 'murnaghan'].index(d['law']) + 1, message=d['law'] + ' 控制完成')
        if is_long:
            temporal_path = MODEL / info['long_model']['time_weights_file']
            temporal = json.loads(temporal_path.read_text(encoding='utf-8'))
            def long_progress(m):
                publish(completed_time_us=m['completed_time_us'],duration_us=m['duration_us'],
                        message=f"已保存 {m['completed_time_us']/1000:.3f} / {m['duration_us']/1000:g} ms 完整记录")
                if (directory / 'pause.request').exists() and m['phase'] != 'completed':
                    raise InterruptedError('已保存检查点，可继续计算。')
            rollout(scene, mesh, weights, temporal, directory / 'record',
                {'weights_sha256':digest(weight_path),'time_weights_sha256':digest(temporal_path),
                 'input_sha256':digest(directory/'request.json')},callback=long_progress)
            recording,times,fields,diagnostics = preview(directory/'record')
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
        export_results(directory, scene, mesh, fields, diagnostics, info, times, recording)
        publish(phase='completed', message='本机预测完成', controls_completed=3,
                weights_sha256=info['weights_sha256'], numerical_correction=False)
    except InterruptedError as error:
        publish(phase='paused',message=str(error))
    except Exception as error:
        publish(phase='failed', message=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True, type=Path)
    run(parser.parse_args().job_dir)
