"""COMSOL reference data: case definitions, format check, import and comparison.

    python -m coarsegraph.comsol cases   [--labels DIR ...]          # regenerate docs/comsol from the label specs
    python -m coarsegraph.comsol check   DATA_DIR                      # validate exported files before sending
    python -m coarsegraph.comsol import  DATA_DIR --output OUT          # convert to the sweep/run label layout
    python -m coarsegraph.comsol compare --comsol OUT [--labels DIR ...]  # COMSOL vs spectral labels, and COMSOL convergence

File contract (docs/comsol-data-request.md):
  DATA_DIR/<case_id>_<control>[_<variant>].csv   control ∈ linear, stvk, murnaghan; variant ∈ mesh2, dt2, amp2
  DATA_DIR/<case_id>.json                        run metadata (template in docs/comsol/metadata-template.json)
CSV: optional comment lines starting with % or #, then one header row
  time_us, <rx>_ux_m, <rx>_uy_m, <rx>_uz_m, ...   (receivers in the order of the case file; `time_s` also accepted)
and one row per output time 0, 0.05, …, 30 μs.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
from solid.schema import require

REPO = Path(__file__).resolve().parents[1]
CASES = REPO / 'docs' / 'comsol' / 'cases'
CONTROLS = ('linear', 'stvk', 'murnaghan')
VARIANTS = ('mesh2', 'dt2', 'amp2')
FAMILIES = {'A': 'labels-a-top-gaussian', 'B': 'labels-b-end-axial', 'C': 'labels-c-two-patch'}
PRIORITY = {'P1': ['A-f0.40-T1.6', 'B-f0.40-T1.6', 'C-f0.25-T2.0'], 'P2': ['A-f0.30-T2.4', 'A-f0.40-T2.0', 'B-f0.30-T2.4', 'C-f0.35-T2.0']}
METADATA_KEYS = ('comsol_version', 'element', 'max_element_size_mm', 'elements', 'dofs', 'integration_order', 'time_stepping', 'tolerances', 'load', 'absorbing', 'material_check', 'runs')


def case_id(family, frequency, thickness):
    return f'{family}-f{frequency:.2f}-T{thickness:.1f}'


def column_prefix(name):
    return name.replace('[', '_').replace(']', '')


def columns(case):
    return ['time_us'] + [f'{column_prefix(r["name"])}_{c}_m' for r in case['output']['receivers'] for c in ('ux', 'uy', 'uz')]


def load_case(cid):
    path = CASES / f'{cid}.json'; require(path.is_file(), f'未知工况 {cid}；可用工况见 docs/comsol/cases/。')
    return json.loads(path.read_text(encoding='utf-8'))


# ---------------------------------------------------------------- case files

def _absorbing(scene):
    from physics.boundaries import alpha_max_per_s, RAMP_POWER
    m = scene['material']; speed = math.sqrt((m['lambda_pa'] + 2 * m['mu_pa']) / m['rho_kg_m3']); rows = []
    for b in scene.get('boundaries', []):
        if b['type'] != 'absorbing': continue
        L = scene['geometry']['length_mm']; start, end = ((b['length_mm'], 0.) if b['face'] == 'xmin' else (L - b['length_mm'], L))
        rows.append({'face': b['face'], 'inner_edge_x_mm': start, 'outer_face_x_mm': end, 'length_mm': b['length_mm'], 'ramp_power': RAMP_POWER[b['ramp']],
                     'alpha_max_per_s': alpha_max_per_s(b, speed), 'formula': 'alpha(x) = alpha_max * clip((length - |x - outer_face_x|) / length, 0, 1) ** ramp_power',
                     'body_load': 'f = -rho0 * alpha(x) * du/dt   (per reference volume, all three components)'})
    return rows


def _sources(scene):
    from physics.scene import sources
    rows = []
    for s in sources(scene):
        profile = s.get('profile') or {'type': 'uniform'}; signal = s.get('signal') or {'type': 'tone_burst', 'window': 'hann'}
        require(signal == {'type': 'tone_burst', 'window': 'hann'}, 'COMSOL 工况只使用 Hann 窗 tone burst。')
        rows.append({'name': s.get('name', 'primary'), 'face': s['face'], 'center_mm': s['center_mm'], 'direction': s['direction'], 'amplitude_mpa': s['amplitude_mpa'],
                     'frequency_mhz': s['frequency_mhz'], 'cycles': s['cycles'], 'delay_us': s['delay_us'], 'phase_deg': s['phase_deg'], 'window': 'hann', 'profile': profile,
                     'traction': 'T0 = amplitude * w(X) * s(t) * direction, nominal traction per reference area (dead load, fixed direction)'})
    return rows


def build_cases(label_dirs=()):
    """docs/comsol/cases/*.json and docs/comsol/cases.csv from the label sweep specs (deterministic), with reference peaks from labels when given."""
    from sweep.spec import load_spec, plan
    from physics.receivers import receiver_points
    from .simulate import label_pairs, fine_projected
    references = {}
    for directory in label_dirs:
        for pair in label_pairs(directory):
            scene = json.loads(Path(pair['murnaghan'], 'scene.json').read_text(encoding='utf-8'))
            names = [r['name'] for r in receiver_points(scene)]; N, _ = fine_projected(Path(pair['murnaghan'], 'out'), names)
            family = [k for k, v in FAMILIES.items() if v == pair['campaign']][0]
            cid = case_id(family, scene['source']['frequency_mhz'], scene['geometry']['thickness_mm'])
            peak = int(np.argmax(np.abs(N[:, 0]))); references[cid] = {'P_far_projected_peak_nm': float(N[peak, 0] * 1e9), 'P_far_peak_time_us': float(peak * scene['acquisition']['sample_step_us']),
                                                                         'max_abs_projected_nm': float(np.max(np.abs(N)) * 1e9)}
    CASES.mkdir(parents=True, exist_ok=True); rows = []
    split = json.loads((REPO / 'coarsegraph' / 'specs' / 'splits.json').read_text(encoding='utf-8'))
    for family, campaign in FAMILIES.items():
        spec = load_spec(REPO / 'coarsegraph' / 'specs' / f'{campaign}.json')
        for entry in plan(spec):
            scene = entry['scene']
            if scene['constitutive'] != 'murnaghan': continue
            f = scene['source']['frequency_mhz']; T = scene['geometry']['thickness_mm']; cid = case_id(family, f, T)
            priority = next((p for p, ids in PRIORITY.items() if cid in ids), 'P3')
            rule = split[campaign]; label_split = rule.get('all') or next((s for s in ('train', 'validation', 'test') if any(all(abs(scene_value(scene, k) - v) < 1e-12 for k, v in item.items()) for item in rule.get(s, []))), None)
            receivers = [{'name': r['name'], 'point_mm': [round(v, 9) for v in r['point_mm']], 'direction': r['direction'], 'columns': [f'{column_prefix(r["name"])}_{c}_m' for c in ('ux', 'uy', 'uz')]} for r in receiver_points(scene)]
            g = scene['geometry']; symmetric = [b['face'] for b in scene.get('boundaries', []) if b['type'] == 'symmetry']
            case = {'schema': 'workai.comsol.case.v1', 'case_id': cid, 'family': family, 'priority': priority, 'label_split': label_split,
                    'geometry': {'kind': 'plate', 'x_mm': [0., g['length_mm']], 'y_mm': [0., g['width_mm']], 'z_mm': [0., T], 'symmetry_planes': symmetric,
                                 'note': 'half model: the y = width face is a symmetry plane (u_y = 0); every other face is traction-free except the loads'},
                    'material': scene['material'], 'controls': list(CONTROLS), 'sources': _sources(scene), 'absorbing': _absorbing(scene),
                    'output': {'t_start_us': 0., 't_end_us': scene['acquisition']['duration_us'], 'step_us': scene['acquisition']['sample_step_us'],
                               'samples': round(scene['acquisition']['duration_us'] / scene['acquisition']['sample_step_us']) + 1, 'receivers': receivers},
                    'reference': references.get(cid), 'scene_v2': {k: v for k, v in scene.items() if k != 'constitutive'}}
            (CASES / f'{cid}.json').write_text(json.dumps(case, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
            ref = references.get(cid) or {}
            rows.append([cid, family, priority, label_split, f, T, len(case['sources']), '+'.join(a['face'] for a in case['absorbing']),
                         ref.get('P_far_projected_peak_nm'), ref.get('P_far_peak_time_us'), ref.get('max_abs_projected_nm')])
    rows.sort(key=lambda r: (r[2], r[0]))
    with (REPO / 'docs' / 'comsol' / 'cases.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle); writer.writerow(['case_id', 'family', 'priority', 'label_split', 'frequency_mhz', 'thickness_mm', 'sources', 'absorbing_faces',
                                                      'ref_P_far_projected_peak_nm', 'ref_P_far_peak_time_us', 'ref_max_abs_projected_nm'])
        for r in rows: writer.writerow([('' if v is None else (f'{v:.6g}' if isinstance(v, float) else v)) for v in r])
    return rows


def scene_value(scene, path):
    from sweep.spec import get_path
    return get_path(scene, path)


# ---------------------------------------------------------------- reading and checking

def read_csv(path, case):
    """times_us (frames), displacements (frames × receivers × 3) in metres; raises ValueError with a precise message."""
    lines = [l for l in Path(path).read_text(encoding='utf-8-sig').splitlines() if l.strip() and not l.lstrip().startswith(('%', '#'))]
    require(lines, f'{Path(path).name}：没有数据。')
    header = [h.strip() for h in next(csv.reader([lines[0]]))]; expected = columns(case)
    scale = 1.
    if header and header[0] == 'time_s': header = ['time_us'] + header[1:]; scale = 1e6
    require(header == expected, f'{Path(path).name}：表头与工况 {case["case_id"]} 不符。应为 {len(expected)} 列，前几列 {expected[:4]}…，实际 {header[:4]}…（共 {len(header)} 列）。')
    data = np.array([[float(v) for v in row] for row in csv.reader(lines[1:])], dtype=float)
    require(data.ndim == 2 and data.shape[1] == len(expected), f'{Path(path).name}：存在列数不一致的数据行。')
    require(np.all(np.isfinite(data)), f'{Path(path).name}：存在非有限数值。')
    times = data[:, 0] * scale; out = case['output']; grid = out['t_start_us'] + np.arange(out['samples']) * out['step_us']
    require(len(times) == out['samples'], f'{Path(path).name}：需要 {out["samples"]} 个时间点（{out["t_start_us"]}–{out["t_end_us"]} μs，步长 {out["step_us"]} μs），实际 {len(times)} 个。')
    require(np.max(np.abs(times - grid)) <= 1e-6, f'{Path(path).name}：输出时间与 0、{out["step_us"]}、… μs 的网格不一致（最大偏差 {np.max(np.abs(times - grid)):.3g} μs）。')
    return grid, data[:, 1:].reshape(len(grid), -1, 3)


def significant_digits(path, sample=400):
    """Printed precision of the displacement columns: the largest digit count among sampled non-zero entries
    (a 5-digit export never shows more than 5 digits; trailing zeros of a fixed-precision export count)."""
    lines = [l for l in Path(path).read_text(encoding='utf-8-sig').splitlines() if l.strip() and not l.lstrip().startswith(('%', '#'))][1:]
    rows = list(csv.reader(lines[:: max(1, len(lines) // 50)])); digits = []
    for row in rows:
        for value in row[1:][:: max(1, len(row) // 8)]:
            mantissa = value.strip().lower().split('e')[0].lstrip('+-').replace('.', '').lstrip('0')
            if mantissa and float(value) != 0: digits.append(len(mantissa))
            if len(digits) >= sample: break
    return max(digits) if digits else None


def export_csv(case, times_us, U, path, comment=None):
    """Write displacements (frames × receivers × 3, metres) in the delivery format (used for tests and for sending our own references)."""
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        if comment: handle.write(f'% {comment}\n')
        writer = csv.writer(handle); writer.writerow(columns(case))
        for t, row in zip(times_us, U.reshape(len(times_us), -1)): writer.writerow([f'{t:.6f}'] + [f'{v:.10e}' for v in row])


def projected(case, U):
    return np.einsum('trc,rc->tr', U, np.array([r['direction'] for r in case['output']['receivers']], dtype=float))


def parse_name(path):
    stem = Path(path).stem; parts = stem.split('_')
    require(len(parts) in (2, 3) and parts[1] in CONTROLS and (len(parts) == 2 or parts[2] in VARIANTS), f'{Path(path).name}：文件名应为 <case_id>_<linear|stvk|murnaghan>[_<mesh2|dt2|amp2>].csv。')
    return parts[0], parts[1], parts[2] if len(parts) == 3 else 'base'


def check(directory):
    """Per-file errors and warnings; returns (report, ok)."""
    directory = Path(directory); report = {'files': [], 'cases': {}, 'errors': [], 'warnings': []}
    files = sorted(directory.glob('*.csv')); require(files, f'{directory} 中没有 CSV 文件。')
    for path in files:
        try:
            cid, control, variant = parse_name(path); case = load_case(cid); times, U = read_csv(path, case)
        except ValueError as error:
            report['errors'].append(str(error)); continue
        P = projected(case, U); peak = float(np.max(np.abs(U))); row = {'file': path.name, 'case_id': cid, 'control': control, 'variant': variant, 'max_abs_displacement_m': peak,
                                                                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        report['files'].append(row); report['cases'].setdefault(cid, {}).setdefault(variant, []).append(control)
        digits = significant_digits(path); row['significant_digits'] = digits
        if digits is not None and digits < 10: report['errors'].append(f'{path.name}：数值只有约 {digits} 位有效数字；非线性增量约为位移的 1e-4，需要至少 12 位有效数字导出（科学计数法）。')
        ref = case.get('reference'); expected_max = (ref or {}).get('max_abs_projected_nm', 30.) * 1e-9
        if peak > 1e-5 or peak > 100 * expected_max: report['warnings'].append(f'{path.name}：位移峰值 {peak:.3g} m，约为预期（{expected_max:.2g} m 量级）的 {peak / expected_max:.0f} 倍；请确认单位是米而不是毫米，载荷是 1 MPa。')
        if peak < 1e-11 or peak < expected_max / 100: report['warnings'].append(f'{path.name}：位移峰值 {peak:.3g} m，远小于预期（{expected_max:.2g} m 量级）；请确认载荷幅值 1 MPa 与单位。')
        if ref and control == 'murnaghan' and variant == 'base':
            i = int(np.argmax(np.abs(P[:, 0]))); value = P[i, 0] * 1e9; expected = ref['P_far_projected_peak_nm']
            if not .5 <= abs(value / expected) <= 2: report['warnings'].append(f'{path.name}：P_far 投影位移峰值 {value:.4g} nm，参考值约 {expected:.4g} nm，差别超过 2 倍；请检查载荷幅值、单位和方向。')
            elif np.sign(value) != np.sign(expected): report['warnings'].append(f'{path.name}：P_far 投影位移峰值符号与参考相反（{value:.4g} nm 对 {expected:.4g} nm）；请检查载荷方向。')
            if abs(i * case['output']['step_us'] - ref['P_far_peak_time_us']) > 2.: report['warnings'].append(f'{path.name}：P_far 峰值时刻 {i * case["output"]["step_us"]:.2f} μs，参考约 {ref["P_far_peak_time_us"]:.2f} μs；请检查时间零点或延迟。')
        if control != 'linear' and variant == 'base':
            linear = directory / f'{cid}_linear.csv'
            if linear.is_file():
                try:
                    _, UL = read_csv(linear, case); ratio = float(np.max(np.abs(U - UL)) / max(np.max(np.abs(UL)), 1e-300))
                    row['max_increment_over_linear'] = ratio
                    if ratio < 1e-7: report['warnings'].append(f'{path.name}：与线性对照几乎完全相同（相对差 {ratio:.1e}）；请确认几何非线性和 Murnaghan 本构已开启，且求解器容差足够小。')
                    if ratio > 1e-2: report['warnings'].append(f'{path.name}：与线性对照差别 {ratio:.1e}，远大于 1 MPa 下的预期（约 1e-4）；请检查本构参数与载荷。')
                except ValueError: pass
    for cid, variants in report['cases'].items():
        missing = [c for c in ('linear', 'murnaghan') if c not in variants.get('base', [])]
        if missing: report['errors'].append(f'{cid}：缺少基准运行 {missing}。')
        meta = directory / f'{cid}.json'
        if not meta.is_file(): report['errors'].append(f'{cid}：缺少元数据文件 {cid}.json。')
        else:
            try:
                m = json.loads(meta.read_text(encoding='utf-8')); absent = [k for k in METADATA_KEYS if k not in m]
                if absent: report['errors'].append(f'{cid}.json：缺少字段 {absent}。')
            except json.JSONDecodeError as error: report['errors'].append(f'{cid}.json：不是合法 JSON（{error}）。')
    for cid in PRIORITY['P1']:
        if cid not in report['cases']: report['warnings'].append(f'P1 工况 {cid} 尚未提供。')
        elif not {'mesh2', 'dt2'} <= set(report['cases'][cid]): report['warnings'].append(f'P1 工况 {cid} 缺少收敛变体 mesh2 / dt2。')
    return report, not report['errors']


# ---------------------------------------------------------------- import and compare

def import_data(directory, output):
    """One campaign-like directory per variant (base, mesh2, …) in the label layout used by coarsegraph and sweep."""
    from physics.receivers import receiver_points
    from physics.runs import write_run_outputs
    directory = Path(directory); output = Path(output); report, ok = check(directory); require(ok, '数据检查未通过：' + '；'.join(report['errors']))
    manifests = {}
    for row in report['files']:
        case = load_case(row['case_id']); times, U = read_csv(directory / row['file'], case); meta = json.loads((directory / f'{row["case_id"]}.json').read_text(encoding='utf-8'))
        scene = copy.deepcopy(case['scene_v2']); scene['constitutive'] = row['control']
        if row['variant'] == 'amp2':
            for s in [scene['source']] + scene.get('sources', []): s['amplitude_mpa'] *= 2
        receivers = receiver_points(scene); require([r['name'] for r in receivers] == [r['name'] for r in case['output']['receivers']], '接收点顺序与工况文件不一致。')
        campaign = output / row['variant']; cid = f'{row["case_id"]}-{row["control"]}'; out = campaign / 'cases' / cid / 'out'
        out.parent.mkdir(parents=True, exist_ok=True); (out.parent / 'scene.json').write_text(json.dumps(scene, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        runs = meta.get('runs') or {}; run_meta = runs.get(f'{row["control"]}{"" if row["variant"] == "base" else "_" + row["variant"]}', {})
        diagnostics = {'source': 'COMSOL', 'nodes': meta.get('dofs', 0) // 3 if meta.get('dofs') else None, 'internal_steps': run_meta.get('time_steps'), 'elapsed_s': run_meta.get('solve_time_s'),
                       'external_work_balance_relative': None, 'max_strain': run_meta.get('max_green_strain'), 'comsol_metadata': meta}
        write_run_outputs({'scene': scene, 'mesh': {'points': np.zeros((0, 3))}, 'receivers': receivers,
                           'computed': {'probes_m': U, 'fields_m': None, 'times_us': times, 'diagnostics': diagnostics}}, out, 'workai.comsol.import.v1', 'comsol_import', '3d',
                          extra={'comsol_file': row['file'], 'comsol_file_sha256': row['sha256']}, engine_scope='external_numerical_reference')
        result_sha = hashlib.sha256((out / 'result.json').read_bytes()).hexdigest()
        manifests.setdefault(row['variant'], []).append({'case_id': cid, 'phase': 'completed', 'result_sha256': result_sha, 'split': case['label_split'],
                                                         'parameters': {'family': case['family'], 'source.frequency_mhz': scene['source']['frequency_mhz'], 'geometry.thickness_mm': scene['geometry']['thickness_mm'],
                                                                        'constitutive': row['control']},
                                                         'nodes': diagnostics['nodes'], 'internal_steps': diagnostics['internal_steps'], 'elapsed_s': diagnostics['elapsed_s']})
    for variant, cases in manifests.items():
        (output / variant / 'manifest.json').write_text(json.dumps({'schema': 'workai.comsol.campaign.v1', 'name': f'comsol-{variant}', 'engine': 'comsol_import', 'cases': cases,
                                                                    'source_files': [r for r in report['files'] if r['variant'] == variant]}, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return {v: len(c) for v, c in manifests.items()}


def _pairs(campaign_dir):
    from .simulate import label_pairs
    out = {}
    for pair in label_pairs(campaign_dir):
        p = pair['parameters']; family = p.get('family') or [k for k, v in FAMILIES.items() if v == Path(campaign_dir).name][0]
        out[case_id(family, p['source.frequency_mhz'], p['geometry.thickness_mm'])] = pair
    return out


def _load_pair(pair, names):
    from .simulate import fine_projected
    N, times = fine_projected(Path(pair['murnaghan'], 'out'), names); L, _ = fine_projected(Path(pair['linear'], 'out'), names)
    return times, N, L


def compare(comsol_dir, label_dirs=()):
    """Spectral labels against COMSOL (COMSOL as reference), and COMSOL base against its own refined variants."""
    from .evaluate import case_metrics
    comsol_dir = Path(comsol_dir); base = _pairs(comsol_dir / 'base'); rows = []
    labels = {}
    for d in label_dirs: labels.update(_pairs(d))
    for cid, pair in sorted(base.items()):
        case = load_case(cid); names = [r['name'] for r in case['output']['receivers']]; f = case['sources'][0]['frequency_mhz']
        t, N, L = _load_pair(pair, names); row = {'case_id': cid}
        if cid in labels:
            _, Ns, Ls = _load_pair(labels[cid], names); row['spectral_vs_comsol'] = case_metrics(t, f, Ns, Ls, N, N - L)
        for variant in ('mesh2', 'dt2'):
            refined = _pairs(comsol_dir / variant) if (comsol_dir / variant / 'manifest.json').exists() else {}
            if cid in refined:
                _, Nr, Lr = _load_pair(refined[cid], names); row[f'comsol_base_vs_{variant}'] = case_metrics(t, f, N, L, Nr, Nr - Lr)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0]); sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('cases'); c.add_argument('--labels', type=Path, nargs='*', default=[])
    k = sub.add_parser('check'); k.add_argument('directory', type=Path)
    i = sub.add_parser('import'); i.add_argument('directory', type=Path); i.add_argument('--output', type=Path, required=True)
    m = sub.add_parser('compare'); m.add_argument('--comsol', type=Path, required=True); m.add_argument('--labels', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    if args.command == 'cases': print(json.dumps({'cases': len(build_cases(args.labels))}))
    elif args.command == 'check':
        report, ok = check(args.directory)
        for row in report['files']: print(f"✓ {row['file']}  峰值 {row['max_abs_displacement_m']:.3g} m" + (f"  非线性/线性相对差 {row['max_increment_over_linear']:.2e}" if 'max_increment_over_linear' in row else ''))
        for w in report['warnings']: print('注意：' + w)
        for e in report['errors']: print('错误：' + e)
        print('检查通过' if ok else '检查未通过'); sys.exit(0 if ok else 1)
    elif args.command == 'import': print(json.dumps(import_data(args.directory, args.output)))
    else: print(json.dumps(compare(args.comsol, args.labels), ensure_ascii=False, indent=1))


if __name__ == '__main__': main()
