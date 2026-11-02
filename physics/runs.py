"""Shared on-disk layout of a solver run (2-D or 3-D): result.json, lines.npz, optional fields.npz and CSV."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .scene import summary


def write_run_outputs(run, output, schema, engine, dimension, write_csv=False, extra=None, lines_extra=None, fields_extra=None, engine_scope='numerical_solution_not_neural_prediction'):
    """`run` holds scene, mesh (with 'points'), receivers (from receiver_points), computed (probes_m, fields_m, times_us, diagnostics)."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    scene = run['scene']; receivers = run['receivers']; computed = run['computed']; probes = np.asarray(computed['probes_m']); times = np.asarray(computed['times_us'])
    components = probes.shape[2]; axes = ['x', 'z'] if components == 2 else ['x', 'y', 'z']; pick = [0, 2] if components == 2 else [0, 1, 2]
    files = {}
    def record(path): files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    points = [i for i, rx in enumerate(receivers) if rx['group'] is None]; lines = {}
    for i, rx in enumerate(receivers):
        if rx['group'] is not None: lines.setdefault(rx['group'], []).append(i)
    npz = {'times_us': times, **(lines_extra or {})}
    for name, indices in lines.items():
        direction = np.asarray(receivers[indices[0]]['direction'])[pick]
        npz[f'line_{name}_points_mm'] = np.array([receivers[i]['point_mm'] for i in indices]); npz[f'line_{name}_distance_mm'] = np.array([receivers[i]['distance_mm'] for i in indices])
        npz[f'line_{name}_u_m'] = probes[:, indices]; npz[f'line_{name}_projected_m'] = probes[:, indices] @ direction
    np.savez_compressed(output / 'lines.npz', **npz); record(output / 'lines.npz')
    if computed.get('fields_m') is not None:
        np.savez_compressed(output / 'fields.npz', fields_m=computed['fields_m'], times_us=times, nodes_mm=np.asarray(run['mesh']['points']) * 1e3, **(fields_extra or {})); record(output / 'fields.npz')
    if write_csv:
        for name, indices in lines.items():
            with (output / f'line-{name}.csv').open('w', newline='') as handle:
                writer = csv.writer(handle); writer.writerow(['time_us'] + [f"x{receivers[i]['distance_mm']:.4f}mm" for i in indices])
                for t, row in zip(times, npz[f'line_{name}_projected_m']): writer.writerow([t] + list(row))
            record(output / f'line-{name}.csv')
    result = {'schema': schema, 'engine': engine, 'engine_scope': engine_scope, 'dimension': dimension, 'degree': run.get('degree'), 'scene': scene, 'excitation': summary(scene) if scene.get('schema', '').endswith('v2') else None,
              'mesh': {'nodes': len(run['mesh']['points']), **{k: v for k, v in run.get('mesh_summary', {}).items()}}, 'times_us': times.tolist(), 'diagnostics': computed['diagnostics'],
              'receivers': [dict({'name': receivers[i]['name'], 'point_mm': receivers[i]['point_mm'], 'direction': receivers[i]['direction']}, **{f'u_{a}_m': probes[:, i, j].tolist() for j, a in enumerate(axes)},
                                 projected_m=(probes[:, i] @ np.asarray(receivers[i]['direction'])[pick]).tolist()) for i in points],
              'receiver_lines': [{'name': name, 'count': len(indices), 'direction': receivers[indices[0]]['direction'], 'file': 'lines.npz'} for name, indices in lines.items()],
              'files': files, 'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'accepted_for_solver_routing': False}
    if extra: result.update(extra)
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=1) + '\n', encoding='utf-8')
    return result
