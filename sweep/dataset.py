"""Stack the receiver-line records of a campaign into one training dataset (NPZ + manifest)."""
import hashlib
import json
from pathlib import Path
import numpy as np
from solid.schema import require
from .spec import get_path


def get_path_safe(parameters, key): return parameters.get(key)


def build_dataset(campaign_dir, output, line=None, dtype='float32', parameter_paths=None):
    campaign_dir = Path(campaign_dir); output = Path(output); require(not output.exists(), '数据集文件已存在；请换一个路径。')
    manifest = json.loads((campaign_dir / 'manifest.json').read_text(encoding='utf-8')); spec = json.loads((campaign_dir / 'spec.json').read_text(encoding='utf-8'))
    entries = [c for c in manifest['cases'] if c.get('phase') == 'completed']; require(entries, '没有已完成的用例。')
    paths = parameter_paths or sorted({k for c in entries for k in c['parameters'] if type(get_path_safe(c['parameters'], k)) in (int, float)}); require(paths, '没有可用的数值参数路径。')
    times = None; distance = None; traces = []; receivers = []; parameters = []; splits = []; ids = []; names = None; scenes_sha = []; categorical = []
    for entry in entries:
        out = campaign_dir / 'cases' / entry['case_id'] / 'out'; record = json.loads((out / 'result.json').read_text(encoding='utf-8')); scene = record['scene']
        z = np.load(out / 'lines.npz'); name = line or (record['receiver_lines'][0]['name'] if record['receiver_lines'] else None); require(name and f'line_{name}_projected_m' in z, f'用例 {entry["case_id"]} 没有接收线 {name}。')
        t = z['times_us']; d = z[f'line_{name}_distance_mm']; u = z[f'line_{name}_projected_m']
        if times is None: times, distance, names = t, d, [rx['name'] for rx in record['receivers']]
        require(np.array_equal(times, t) and np.allclose(distance, d, atol=1e-9) and names == [rx['name'] for rx in record['receivers']], '所有用例需要相同的时间网格、接收线采样与接收点集合才能堆叠。')
        traces.append(u.astype(dtype)); receivers.append(np.stack([[rx['u_x_m'], rx['u_z_m']] for rx in record['receivers']], axis=0).astype(dtype) if record['receivers'] else np.zeros((0, 2, len(t)), dtype))
        parameters.append([float(get_path(scene, p)) for p in paths]); splits.append(entry['split']); ids.append(entry['case_id']); scenes_sha.append(hashlib.sha256(json.dumps(scene, sort_keys=True).encode()).hexdigest())
        categorical.append(json.dumps({k: v for k, v in entry['parameters'].items() if k not in paths}, sort_keys=True))
    arrays = {'times_us': times, 'line_distance_mm': distance, 'line_projected_m': np.stack(traces), 'receiver_u_m': np.stack(receivers), 'parameters': np.array(parameters, dtype=np.float64),
              'parameter_names': np.array(paths), 'categorical_parameters': np.array(categorical), 'split': np.array(splits), 'case_ids': np.array(ids), 'receiver_names': np.array(names), 'scene_sha256': np.array(scenes_sha)}
    meta = {'schema': 'workai.sweep.dataset.v1', 'campaign': manifest['name'], 'campaign_spec_sha256': manifest['spec_sha256'], 'engine': manifest['engine'], 'line': name, 'dtype': dtype, 'cases': len(ids),
            'splits': {s: int(np.sum(np.array(splits) == s)) for s in ('train', 'validation', 'test')}, 'units': {'line_projected_m': 'm (projected displacement per unit width loading)', 'times_us': 'us', 'line_distance_mm': 'mm'},
            'shapes': {k: list(v.shape) for k, v in arrays.items()}, 'accepted_for_solver_routing': False, 'source_sha256': manifest['source_sha256'], 'degree': manifest['degree']}
    np.savez_compressed(output, manifest=json.dumps(meta, ensure_ascii=False), **arrays)
    meta['dataset_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest(); output.with_suffix('.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return meta


def load_dataset(path):
    """Arrays plus manifest; splits are selected with `select(data, 'train')`."""
    with np.load(path, allow_pickle=False) as z: data = {k: z[k] for k in z.files}
    data['manifest'] = json.loads(str(data['manifest'])); return data


def select(data, split):
    mask = data['split'] == split
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(mask) and k not in ('times_us', 'line_distance_mm', 'parameter_names', 'receiver_names') else v) for k, v in data.items()}
