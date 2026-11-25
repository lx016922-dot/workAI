"""COMSOL defect runs (docs/comsol/defects) against the software on the same scenes.

    python research/defects/comsol_compare.py run --free-ends D0-intact D1-open-crack ...          # the software without absorbing layers
    python research/defects/comsol_compare.py run --free-ends --h 1.0 D0-intact D1-open-crack ...  # the same on a finer mesh (h in mm, degree 4)
    python research/defects/comsol_compare.py run D0-intact D1-open-crack ...                      # the scenes as specified (absorbing layers)
    python research/defects/comsol_compare.py report                                               # comsol-results/comparison.json, Markdown tables, figures/comsol-*.png

COMSOL: `comsol-results/<date>/<case>.csv` (701 samples, u_x, u_y, u_z at the ten receivers; small-strain linear
elasticity, quartic hexahedra, generalized-α without dissipation) and `<case>_mesh2.csv` (h × 0.7).
Software: the scene of `docs/comsol/defects/cases/<case>.json` through the record path of the application
(`graphnet.rollout`: automatic mesh unless `--h`, automatic time layer, the linear control), receivers at every
sample; written to `comsol-results/software*/<case>.npz`.

Compared at every receiver, on the components that carry the wave there (u_x and u_z; u_y is zero on the symmetry
plane): the waveform (relative L2 of the difference over the record), the amplitude and phase at the centre
frequency, and the same for the scattered field (the defect run minus the intact run of the same solver).

The delivery of 2026-11-14 behaves as a plate with free ends: its intact run equals the software without absorbing
layers to 0.02–0.08 % at every receiver and differs from the specified scene (absorbing layers) as soon as the end
reflections arrive. The comparison that tests the defects is therefore the one with free ends on both sides; the
one with the specified scene is kept to show the difference. The delivery of 2026-11-18 (D4, D7, D7 mesh2) records the
cause in its metadata: the saved models apply the body load with zero components, the damping expression sits in an
inactive field.
"""
import csv
import json
import sys
import tempfile
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
CASES = REPO / 'docs' / 'comsol' / 'defects' / 'cases'
RESULTS = HERE / 'comsol-results'
FREE = RESULTS / 'software-free-ends'
VARIANTS = {'两端自由（COMSOL 交付的实际状态）': FREE, '按要求的场景（两端吸收层）': RESULTS / 'software'}
INTACT = 'D0-intact'
F_MHZ = .3
GATE = .05            # waveform and amplitude at f, as for P1
SMALL = .02           # a component under this fraction of the receiver's largest peak is not compared (the other wave family's leakage)
WEAK = .05            # a scattered component under this fraction of the intact wave is listed apart: its relative difference is a ratio of small numbers
AXES = 'xyz'


def comsol(case):
    """(times, samples × receivers × 3, receiver names, metadata, delivery) of the newest COMSOL delivery that has this case."""
    folder = next(d for d in sorted((p for p in RESULTS.iterdir() if p.is_dir() and not p.name.startswith('software')), reverse=True) if (d / f'{case}.csv').exists())
    with (folder / f'{case}.csv').open(encoding='utf-8-sig') as stream:
        rows = [row for row in csv.reader(line for line in stream if not line.startswith(('%', '#')))]
    header = rows[0]; data = np.array(rows[1:], dtype=float); names = [h[:-5] for h in header[1::3]]
    assert header[0] == 'time_us' and all(header[1 + 3 * i + a] == f'{name}_u{AXES[a]}_m' for i, name in enumerate(names) for a in range(3))
    return data[:, 0], data[:, 1:].reshape(len(data), len(names), 3), names, json.loads((folder / f'{case}.json').read_text(encoding='utf-8-sig')), folder.name


def folder_of(free_ends, h):
    return Path(str(FREE if free_ends else RESULTS / 'software') + (f'-h{h:g}' if h else ''))


def run(names, free_ends=False, h=None):
    from offline.domain import validate_request
    from spectral3d.feature_mesh import make_mesh
    from graphnet.model import load
    from graphnet.rollout import read_record, rollout
    target = folder_of(free_ends, h); target.mkdir(parents=True, exist_ok=True)
    for name in names:
        raw = json.loads((CASES / f'{name}.json').read_text(encoding='utf-8'))
        if free_ends: raw['scene']['boundaries'] = [b for b in raw['scene']['boundaries'] if b['type'] != 'absorbing']
        if h: raw.pop('mesh'); raw['degree'] = 4; raw['scene']['numerics']['h_mm'] = h
        request = validate_request(raw); scene = request['scene']
        mesh = make_mesh(scene, request['degree'], scene['numerics']['h_mm']); model = load(request['degree'], long_record=True); started = time.monotonic()
        with tempfile.TemporaryDirectory() as tmp:   # what the application does for this request, with only the linear control
            manifest = rollout(scene, mesh, model, Path(tmp) / 'record', {'case': name}, time_plan='auto', record='compact', controls=('linear',), flush=True)
            record = read_record(Path(tmp) / 'record')
        energies = record['energies'][0]; balance = float(np.abs(energies[:, 1] + energies[:, 2] + energies[:, 4] - energies[:, 3]).max() / np.abs(energies[:, 3]).max())
        info = {'case': name, 'nodes': int(len(mesh['points'])), 'degree': int(request['degree']), 'h_mm': float(scene['numerics']['h_mm']), 'mesh': request.get('mesh_report'),
                'time_layer': {k: manifest['time_plan'][k] for k in ('scheme', 'layer_step_us', 'evaluations_per_control')}, 'seconds': time.monotonic() - started, 'energy_balance': balance}
        np.savez_compressed(target / f'{name}.npz', times_us=record['times_us'], probes_m=record['probes'][:, :, 0], names=np.array([r['name'] for r in scene['receivers']]))
        (target / f'{name}.json').write_text(json.dumps(info, ensure_ascii=False, indent=1, default=float) + '\n', encoding='utf-8')
        print(json.dumps({k: info[k] for k in ('case', 'nodes', 'h_mm', 'time_layer', 'seconds', 'energy_balance')}, ensure_ascii=False), flush=True)


def software(case, folder):
    with np.load(folder / f'{case}.npz') as z: return z['times_us'], z['probes_m'], [str(n) for n in z['names']]


def at_f(trace, dt_us):
    """Complex amplitude at the centre frequency (Hann window, as research/defects/run_cases.py)."""
    n = 1 << 16; spec = np.fft.rfft(trace * np.hanning(len(trace)), n); freqs = np.fft.rfftfreq(n, dt_us)
    return spec[np.argmin(np.abs(freqs - F_MHZ))]


def compare(a, b, dt, scale=None):
    """b against a (the reference) for one trace: waveform difference relative to `scale` (default: the reference), amplitude ratio and phase at f."""
    x, y = at_f(a, dt), at_f(b, dt)
    return {'waveform': float(np.linalg.norm(b - a) / (scale if scale is not None else np.linalg.norm(a))), 'amplitude': float(abs(y) / abs(x) - 1), 'phase_deg': float(np.degrees(np.angle(y / x)))}


def components(reference):
    """(receiver, axis) pairs worth comparing: u_x and u_z whose peak is at least SMALL of the receiver's largest."""
    peaks = np.abs(reference).max(axis=0)
    return [(r, a) for r in range(reference.shape[1]) for a in (0, 2) if peaks[r, a] >= SMALL * peaks[r].max()]


def table(pairs, names, reference, other, dt, scales=None):
    return {f'{names[r]}_u{AXES[a]}': compare(reference[:, r, a], other[:, r, a], dt, None if scales is None else scales[r, a]) for r, a in pairs}


def worst(rows, key, keep=None):
    values = [abs(v[key]) for k, v in rows.items() if keep is None or k in keep]
    return max(values) if values else float('nan')


def pair(pairs, names, reference, other, reference0, other0, dt):
    """`other` against `reference`: the total field and, with the intact runs of both, the scattered field."""
    row = {'total': table(pairs, names, reference, other, dt)}
    if reference0 is not None:
        # Scattered field: the run minus the intact run of the same solver. Its difference is given relative to the scattered field itself
        # and relative to the intact wave at that receiver (a small scattered field makes the first one large for a small error).
        rs, os = reference - reference0, other - other0; norms = np.linalg.norm(reference0, axis=0)
        row['scattered'] = table(pairs, names, rs, os, dt); row['scattered_vs_intact_wave'] = table(pairs, names, rs, os, dt, norms)
        row['scattered_to_intact'] = {f'{names[r]}_u{AXES[a]}': float(np.linalg.norm(rs[:, r, a]) / norms[r, a]) for r, a in pairs}
    return row


def harmonics(pairs, names, reference, other, dt):
    """2f / f and 3f / f of each solver at every compared component (closed defects: the contact makes the harmonics)."""
    def ratio(trace, k):
        n = 1 << 16; spec = np.abs(np.fft.rfft(trace * np.hanning(len(trace)), n)); freqs = np.fft.rfftfreq(n, dt)
        pick = lambda f: spec[np.argmin(np.abs(freqs - f))]
        return float(pick(k * F_MHZ) / pick(F_MHZ))
    return {f'{names[r]}_u{AXES[a]}': {f'{k}f_over_f': {'comsol': ratio(reference[:, r, a], k), 'software': ratio(other[:, r, a], k)} for k in (2, 3)} for r, a in pairs}


BANDS_MHZ = (.81, 2.)   # the automatic mesh's band (2 × the −30 dB edge of the 0.3 MHz burst), and a wider one


def band_limited(other, reference, pairs, dt, top):
    """Largest relative L2 difference over the compared components after removing content above `top` MHz (raised-cosine roll-off over its last 10 %)."""
    def low(x):
        spec = np.fft.rfft(x, axis=0); freqs = np.fft.rfftfreq(len(x), dt); gain = np.clip((top - freqs) / (.1 * top), 0., 1.); gain = .5 - .5 * np.cos(np.pi * gain)
        return np.fft.irfft(spec * gain[:, None, None], len(x), axis=0)
    a, b = low(other), low(reference)
    return float(max(np.linalg.norm(a[:, r, k] - b[:, r, k]) / np.linalg.norm(b[:, r, k]) for r, k in pairs))


def amplitude(case):
    raw = json.loads((CASES / f'{case}.json').read_text(encoding='utf-8'))['scene']
    return float(raw['source']['amplitude_mpa'])


def variant(folder):
    """Every case computed in `folder` against COMSOL (the base mesh, and mesh2 where it was delivered)."""
    cases = sorted(p.stem for p in folder.glob('*.npz')); out = {}
    t, c0, names, _, _ = comsol(INTACT); _, s0, software_names = software(INTACT, folder); dt = t[1] - t[0]; assert names == software_names and len(t) == len(s0)
    pairs = components(c0)
    for case in cases:
        # The scattered field subtracts the intact run at the same amplitude (D5 is driven at 2 MPa, the intact run at 1 MPa;
        # every run is linear in the load except through contact, which is homogeneous of degree one).
        scale = amplitude(case) / amplitude(INTACT); c0s, s0s = c0 * scale, s0 * scale
        _, c, _, meta, delivery = comsol(case); _, s, _ = software(case, folder); info = json.loads((folder / f'{case}.json').read_text(encoding='utf-8')); defect = case != INTACT
        row = {'comsol': {'delivery': delivery, 'elements': meta['elements'], 'dofs': meta['dofs'], 'solve_time_s': meta['runs']['linear']['solve_time_s'], 'max_abs_uy_m': meta['checks']['max_abs_uy_m']},
               'software': {k: info[k] for k in ('nodes', 'degree', 'h_mm', 'time_layer', 'seconds', 'energy_balance')}, **pair(pairs, names, c, s, c0s if defect else None, s0s, dt)}
        if 'contact' in meta and meta['contact']: row['comsol']['contact'] = meta['contact']; row['harmonics'] = harmonics(pairs, names, c, s, dt)
        try:
            _, fine, _, fine_meta, _ = comsol(case + '_mesh2')
            row['comsol_mesh2'] = {'elements': fine_meta['elements'], 'dofs': fine_meta['dofs'], 'solve_time_s': fine_meta['runs']['linear']['solve_time_s'],
                                   'base': pair(pairs, names, fine, c, c0s if defect else None, c0s, dt), 'software': pair(pairs, names, fine, s, c0s if defect else None, s0s, dt)}
            if 'harmonics' in row:
                # Closed defects: the clapping also radiates far above the band the meshes are sized for, so the waveform
                # difference is also given band-limited, and the harmonics are compared with the finer COMSOL mesh.
                row['comsol_mesh2']['band_limited_waveform'] = {f'{top:g}': {'base': band_limited(c, fine, pairs, dt, top), 'software': band_limited(s, fine, pairs, dt, top)}
                                                                for top in BANDS_MHZ}
                row['comsol_mesh2']['harmonics'] = {'software': harmonics(pairs, names, fine, s, dt), 'base': harmonics(pairs, names, fine, c, dt)}
        except StopIteration:
            pass
        finer = sorted(Path(str(folder)).parent.glob(folder.name + '-h*'))
        for other in finer:   # the software against itself on a finer mesh, and COMSOL against that
            if (other / f'{case}.npz').exists() and (other / f'{INTACT}.npz').exists():
                _, f, _ = software(case, other); _, f0, _ = software(INTACT, other); f0 = f0 * scale; fine_info = json.loads((other / f'{case}.json').read_text(encoding='utf-8'))
                row.setdefault('software_refined', {})[other.name.rsplit('-h', 1)[1]] = {
                    'nodes': fine_info['nodes'], 'seconds': fine_info['seconds'], 'software': pair(pairs, names, f, s, f0 if defect else None, s0s, dt), 'comsol': pair(pairs, names, f, c, f0 if defect else None, c0s, dt),
                    **({'comsol_mesh2': pair(pairs, names, f, comsol(case + '_mesh2')[1], f0 if defect else None, c0s, dt)} if 'comsol_mesh2' in row else {})}
        out[case] = row
    return out


def cells(row):
    """Waveform, amplitude and phase of the total field; the scattered field where it is at least WEAK of the intact wave; the weak components."""
    out = [f'{worst(row["total"], "waveform"):.2%}', f'{worst(row["total"], "amplitude"):.2%}', f'{worst(row["total"], "phase_deg"):.2f}°']
    if 'scattered' not in row: return out + ['—'] * 3
    strong = {k for k, v in row['scattered_to_intact'].items() if v >= WEAK}; weak = set(row['scattered']) - strong
    out += [f'{worst(row["scattered"], "waveform", strong):.1%}', f'{worst(row["scattered"], "amplitude", strong):.1%}']
    out.append(f'{len(weak)} 个：散射场差 {worst(row["scattered"], "waveform", weak):.1%}，相当于完好板的波的 {worst(row["scattered_vs_intact_wave"], "waveform", weak):.2%}' if weak else '无')
    return out


HEADER = ['| 对比 | 波形差 | f 幅值差 | f 相位差 | 散射场波形差 | 散射场 f 幅值差 | 弱散射分量（散射场不到完好板的波的 5%） |', '|---|---|---|---|---|---|---|']


def markdown(cases):
    lines = list(HEADER)
    for case, row in cases.items(): lines.append(f'| {case}：软件 对 COMSOL | ' + ' | '.join(cells(row)) + ' |')
    extra = []
    for case, row in cases.items():
        if 'comsol_mesh2' in row:
            extra += [f'| {case}：COMSOL 基准网格 对 COMSOL mesh2 | ' + ' | '.join(cells(row['comsol_mesh2']['base'])) + ' |', f'| {case}：软件 对 COMSOL mesh2 | ' + ' | '.join(cells(row['comsol_mesh2']['software'])) + ' |']
        for h, fine in row.get('software_refined', {}).items():
            extra += [f'| {case}：软件自动网格 对 软件 h {h} mm | ' + ' | '.join(cells(fine['software'])) + ' |', f'| {case}：COMSOL 基准网格 对 软件 h {h} mm | ' + ' | '.join(cells(fine['comsol'])) + ' |']
            if 'comsol_mesh2' in fine: extra.append(f'| {case}：COMSOL mesh2 对 软件 h {h} mm | ' + ' | '.join(cells(fine['comsol_mesh2'])) + ' |')
    closed = []
    for case, row in cases.items():
        if 'harmonics' in row:
            for k in (2, 3):
                values = [v[f'{k}f_over_f'] for v in row['harmonics'].values()]
                ratios = [v['software'] / v['comsol'] - 1 for v in values]
                closed.append(f'| {case} | {k}f / f | {min(v["comsol"] for v in values):.4f}–{max(v["comsol"] for v in values):.4f} | {min(ratios):+.1%} 到 {max(ratios):+.1%} |')
    for case, row in cases.items():
        fine = row.get('comsol_mesh2', {})
        if 'harmonics' in fine:
            for label, key in (('软件', 'software'), ('COMSOL 基准网格', 'base')):
                ratios = [v['2f_over_f']['software'] / v['2f_over_f']['comsol'] - 1 for v in fine['harmonics'][key].values()]
                bands = '，'.join(f"≤ {top} MHz 波形差 {fine['band_limited_waveform'][top][key]:.2%}" for top in fine['band_limited_waveform'])
                closed.append(f'| {case} | {label} 对 COMSOL mesh2：2f / f | | {min(ratios):+.1%} 到 {max(ratios):+.1%}（{bands}） |')
    tail = ['', '闭合缺陷的谐波（各接收点、各分量）：', '', '| 工况 | 谐波 | COMSOL 的范围 | 软件 / COMSOL − 1 |', '|---|---|---|---|'] + closed if closed else []
    return '\n'.join(lines + (['', '网格收敛（每行以后者为准）：', ''] + HEADER + extra if extra else []) + tail)


def figures():
    """figures/comsol-scattered.png (the scattered field of each defect, both solvers with free ends) and
    figures/comsol-d0-boundaries.png (the intact plate against the software with and without absorbing layers)."""
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed: no figures'); return
    target = HERE / 'figures'; target.mkdir(exist_ok=True)
    t, c0, names, _, _ = comsol(INTACT); _, s0, _ = software(INTACT, FREE)
    defects = [c for c in ('D1-open-crack', 'D2-open-delamination', 'D3-side-drilled-hole', 'D4-closed-crack-1MPa', 'D7-part-width-crack') if (FREE / f'{c}.npz').exists()]
    picks = [('top_20', 2), ('top_40', 2), ('mid_20', 0), ('mid_40', 0)]
    if defects:
        fig, axes = plt.subplots(len(defects), len(picks), figsize=(16, 3 * len(defects)), sharex=True, squeeze=False)
        for row, case in enumerate(defects):
            _, c, _, _, _ = comsol(case); _, s, _ = software(case, FREE)
            try:
                fine = comsol(case + '_mesh2')[1]
            except StopIteration:
                fine = None
            for col, (name, a) in enumerate(picks):
                i = names.index(name); ax = axes[row, col]
                ax.plot(t, (c[:, i, a] - c0[:, i, a]) * 1e9, color='#1f77b4', lw=1.4, label='COMSOL')
                if fine is not None: ax.plot(t, (fine[:, i, a] - c0[:, i, a]) * 1e9, color='#2ca02c', lw=1, ls=':', label='COMSOL mesh2 (h × 0.7)')
                ax.plot(t, (s[:, i, a] - s0[:, i, a]) * 1e9, color='#d62728', lw=1, ls='--', label='WorkAI')
                ax.set_title(f'{case}  {name}  u_{AXES[a]}', fontsize=9); ax.grid(alpha=.3)
                if col == 0: ax.set_ylabel('scattered field (nm)')
                if row == len(defects) - 1: ax.set_xlabel('t (μs)')
        axes[0, 0].legend(fontsize=7)
        fig.suptitle('Scattered field (defect run − intact run of the same solver), both with free ends', fontsize=11)
        fig.tight_layout(); fig.savefig(target / 'comsol-scattered.png', dpi=110); plt.close(fig)
    specified = RESULTS / 'software' / f'{INTACT}.npz'
    if specified.exists():
        _, s_abs, _ = software(INTACT, RESULTS / 'software'); i = names.index('top_45')
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.plot(t, c0[:, i, 2] * 1e9, color='#1f77b4', lw=1.6, label='COMSOL D0')
        ax.plot(t, s0[:, i, 2] * 1e9, color='#d62728', lw=1, ls='--', label='WorkAI, free ends')
        ax.plot(t, s_abs[:, i, 2] * 1e9, color='0.5', lw=1, label='WorkAI, absorbing layers as specified')
        ax.set_xlabel('t (μs)'); ax.set_ylabel('u_z at top_45 (nm)'); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_title('D0 intact plate: the COMSOL record carries the end reflections of a plate without absorbing layers', fontsize=10)
        fig.tight_layout(); fig.savefig(target / 'comsol-d0-boundaries.png', dpi=110); plt.close(fig)


def report():
    out = {'gate': GATE, 'frequency_mhz': F_MHZ, 'delivery': comsol(INTACT)[4], 'variants': {}}
    for name, folder in VARIANTS.items():
        if (folder / f'{INTACT}.npz').exists():
            out['variants'][name] = variant(folder); print(f'\n## {name}（各接收点、各分量中的最大值）\n'); print(markdown(out['variants'][name]))
    homogeneity(out)
    (RESULTS / 'comparison.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    figures()


def homogeneity(out, low='D4-closed-crack-1MPa', high='D5-closed-crack-2MPa'):
    """Zero-gap, unloaded contact is positively homogeneous of degree one: the 2 MPa run is exactly twice the 1 MPa run.
    Checked for each solver on its own (COMSOL to its iteration tolerance, the software to the bit)."""
    try:
        t, c_low, names, meta_low, _ = comsol(low); _, c_high, _, meta_high, _ = comsol(high)
    except StopIteration:
        return
    pairs = components(comsol(INTACT)[1]); row = {}
    for label, a, b in (('comsol', c_low, c_high), *((('software', software(low, FREE)[1], software(high, FREE)[1]),) if (FREE / f'{high}.npz').exists() else ())):
        row[label] = {'worst_component_rel_l2_of_high_minus_twice_low': float(max(np.linalg.norm(b[:, r, k] - 2 * a[:, r, k]) / np.linalg.norm(b[:, r, k]) for r, k in pairs)),
                      'max_abs_over_peak': float(np.abs(b - 2 * a).max() / np.abs(b).max())}
    ratios = [v['2f_over_f']['software'] / v['2f_over_f']['comsol'] - 1 for v in harmonics(pairs, names, c_low, c_high, t[1] - t[0]).values()]
    row['comsol_2f_over_f_change_from_1_to_2_mpa'] = [float(min(ratios)), float(max(ratios))]
    row['comsol_contact'] = {k: [meta_low['contact'][k], meta_high['contact'][k]] for k in ('max_penetration_m', 'max_contact_pressure_pa')}
    out['homogeneity'] = row
    print(f'\n一次齐次性（{high} − 2 × {low}，最差分量的相对 L2）：COMSOL {row["comsol"]["worst_component_rel_l2_of_high_minus_twice_low"]:.1e}'
          + (f'，软件 {row["software"]["worst_component_rel_l2_of_high_minus_twice_low"]:.1e}' if 'software' in row else ''))


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        arguments = sys.argv[2:]; h = float(arguments[arguments.index('--h') + 1]) if '--h' in arguments else None
        run([a for i, a in enumerate(arguments) if not a.startswith('--') and (i == 0 or arguments[i - 1] != '--h')], '--free-ends' in arguments, h)
    else: report()
