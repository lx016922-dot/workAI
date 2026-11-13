"""How far outside their training range do the coarse models hold?

    python -m coarsegraph.verify run --output research/coarse-range/results [--only NAME ...]
    python -m coarsegraph.verify report --results research/coarse-range/results
    python -m coarsegraph.verify table --results research/coarse-range/results      (Markdown)
    python -m coarsegraph.verify reference --output research/coarse-range/results --case distance-f0.4-t1.6 --h-mm 0.8

The shipped models were trained on 48 × 2 mm half-width plates, 30 μs records and receivers 6–21 mm from the
source. The learned operators are element operators shared by every element, so nothing ties them to that plate;
what has to be measured is how their error grows with propagation distance, plate width, other excitations and
defects. Every case here is computed three ways on the same scene:

* fine reference: the frozen network (the GLL spectral-element operators) on a 1 mm, degree-4 grid;
* each coarse model on its own grid with its learned operators;
* the same coarse grid with the GLL operators (what learning is compared with).

and judged with the gates of `coarsegraph.evaluate` (5 % on the waveform, f and 2f amplitudes of the complete
record and on the 2f amplitude of the nonlinear increment), separately for each group of receivers, so a long
plate shows the gates as a function of distance.

The training records were 30 μs long and the wave filled them, so the gates used the whole record. A receiver far
down a long plate is quiet for most of a long record, and a Hann-weighted 2f amplitude over the whole of it is
mostly leakage of the fundamental. The gates are therefore taken on the arrival window of each receiver group
(`arrival_window`: where the fine reference exceeds 2 % of its peak, padded by two periods); the whole-record
values are stored next to them. The receiver waveforms of every run are stored too (`*.npz`, single precision), so
the metrics can be recomputed without running anything.
"""
import argparse
import copy
import json
import tempfile
import time
from pathlib import Path
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
MODELS = {'p4-h3-harmonic': REPOSITORY / 'coarsegraph/models/p4-h3-harmonic/model.json', 'p4-h2-multisource': REPOSITORY / 'coarsegraph/models/p4-h2-multisource/model.json'}
FINE_H_MM, DEGREE = 1., 4
PAIR = ('linear', 'murnaghan')
MATERIAL = {'rho_kg_m3': 2700.0, 'lambda_pa': 58100000000.0, 'mu_pa': 26100000000.0, 'l_pa': -143000000000.0, 'm_pa': -297000000000.0, 'n_pa': -345000000000.0}
SCHEMA = 'workai.coarsegraph.range_verification.v1'


def plate(length=48., half=2., thickness=1.6, frequency=.3, duration=30., excitation='top', source_x=16., groups=(('near', 22., 37.),), points=6, edge_y=None, defects=None, symmetry=True):
    """A half-width aluminium plate like the training scenes, with receiver groups (name, first x, last x): a top line (z displacement) and a mid-plane line (x displacement)."""
    y = half if symmetry else half / 2; tone = {'type': 'tone_burst', 'window': 'hann'}
    top = lambda x, sign=-1., delay=0., sigma=2.: {'face': 'zmax', 'center_mm': [x, y, thickness], 'radius_mm': 0., 'direction': [0., 0., sign], 'amplitude_mpa': 1., 'frequency_mhz': frequency, 'cycles': 5.,
                                                 'delay_us': delay, 'phase_deg': 0., 'signal': tone, 'profile': {'type': 'gaussian', 'sigma_mm': sigma}}
    scene = {'schema': 'workai.hyperelastic.solid.v2', 'name': '粗网格核验',
             'geometry': {'kind': 'plate', 'length_mm': length, 'width_mm': half, 'thickness_mm': thickness, 'outer_radius_mm': 3., 'inner_radius_mm': 2., 'hole_radius_mm': .7, 'hole_x_mm': 6.,
                          'hole_y_mm': 2., 'notch_x_mm': 6., 'notch_width_mm': 1., 'notch_depth_mm': .6},
             'material': dict(MATERIAL), 'constitutive': 'murnaghan', 'boundary': {'fixed_face': 'none'}, 'acquisition': {'duration_us': duration, 'sample_step_us': .05},
             'numerics': {'h_mm': FINE_H_MM, 'cfl': .3, 'max_elements': 40000}}
    absorb = lambda face: {'type': 'absorbing', 'face': face, 'length_mm': 8., 'ramp': 'cubic', 'attenuation_db': 60.}
    if excitation == 'end':
        scene['source'] = {'face': 'xmin', 'center_mm': [0., y, thickness / 2], 'radius_mm': 0., 'direction': [1., 0., 0.], 'amplitude_mpa': 1., 'frequency_mhz': frequency, 'cycles': 5., 'delay_us': 0.,
                           'phase_deg': 0., 'signal': tone}
        scene['boundaries'] = [absorb('xmax')]
    else:
        scene['source'] = dict(top(source_x - 3. if excitation == 'two_patch' else source_x, sigma=1.5 if excitation == 'two_patch' else 2.), name='贴片 1')
        if excitation == 'two_patch': scene['sources'] = [dict(top(source_x + 3., sign=1., delay=1., sigma=1.5), name='贴片 2')]
        scene['boundaries'] = [absorb('xmin'), absorb('xmax')]
    if symmetry: scene['boundaries'].append({'type': 'symmetry', 'face': 'ymax'})
    receivers = []
    for name, first, last in groups:
        for i, x in enumerate(np.linspace(first, last, points)):
            receivers.append({'name': f'{name}:top{i}', 'point_mm': [float(x), y, thickness], 'direction': [0., 0., 1.]})
            receivers.append({'name': f'{name}:mid{i}', 'point_mm': [float(x), y, thickness / 2], 'direction': [1., 0., 0.]})
            if edge_y is not None: receivers.append({'name': f'{name}-edge:top{i}', 'point_mm': [float(x), edge_y, thickness], 'direction': [0., 0., 1.]})
    scene['receivers'] = receivers
    if defects: scene['defects'] = defects
    return scene


def cases():
    """The verification matrix: name → (what it probes, scene)."""
    out = {}; far = (('d10-25', 26., 41.), ('d45-60', 61., 76.), ('d85-100', 101., 116.), ('d125-140', 141., 156.))
    for thickness in (1.6, 2.4):
        for frequency in (.2, .3, .4):
            tag = f'f{frequency:g}-t{thickness:g}'
            out[f'distance-{tag}'] = ('propagation distance', plate(200., 2., thickness, frequency, 100., groups=far, points=4))
    for thickness in (1.6, 2.4):
        for frequency in (.2, .4):
            tag = f'f{frequency:g}-t{thickness:g}'
            out[f'width-{tag}'] = ('plate width', plate(48., 8., thickness, frequency, 30., edge_y=2.))
            out[f'end-{tag}'] = ('end-face excitation, long plate', plate(100., 2., thickness, frequency, 50., 'end', groups=(('d20-35', 20., 35.), ('d60-75', 60., 75.))))
    out['two-patch-f0.25-t2'] = ('two-patch excitation, long plate', plate(100., 2., 2., .25, 60., 'two_patch', groups=(('d10-25', 26., 41.), ('d45-60', 61., 76.))))
    out['full-width-f0.3-t1.6'] = ('no symmetry plane', plate(48., 4., 1.6, .3, 30., symmetry=False))
    around = (('before', 26., 41.), ('after', 59., 74.))
    for name, defect in (('open-crack', {'type': 'crack', 'x_mm': 50., 'depth_mm': .8}), ('notch', {'type': 'notch', 'x_mm': 50., 'width_mm': 1., 'depth_mm': .6}),
                         ('side-hole', {'type': 'hole', 'axis': 'y', 'x_mm': 50., 'z_mm': .8, 'radius_mm': .3}), ('open-delamination', {'type': 'delamination', 'x_start_mm': 46., 'x_end_mm': 54., 'depth_mm': .8}),
                         ('closed-crack', {'type': 'crack', 'x_mm': 50., 'depth_mm': .8, 'closed': True})):
        out[f'defect-{name}-f0.3-t1.6'] = ('defect', plate(100., 2., 1.6, .3, 60., groups=around, defects=[defect]))
    # Volume defects at the other corners of the range, and a hole through the thickness of a full-width plate (receivers on the centre line and 1 mm from the edge).
    out['defect-notch-f0.2-t1.6'] = ('defect', plate(100., 2., 1.6, .2, 60., groups=around, defects=[{'type': 'notch', 'x_mm': 50., 'width_mm': 1., 'depth_mm': .6}]))
    out['defect-notch-f0.4-t2.4'] = ('defect', plate(100., 2., 2.4, .4, 60., groups=around, defects=[{'type': 'notch', 'x_mm': 50., 'width_mm': 1., 'depth_mm': .9}]))
    out['defect-side-hole-f0.4-t2.4'] = ('defect', plate(100., 2., 2.4, .4, 60., groups=around, defects=[{'type': 'hole', 'axis': 'y', 'x_mm': 50., 'z_mm': 1.2, 'radius_mm': .4}]))
    out['defect-through-hole-f0.3-t1.6'] = ('defect', plate(100., 8., 1.6, .3, 60., groups=around, edge_y=1., symmetry=False, defects=[{'type': 'hole', 'axis': 'z', 'x_mm': 50., 'y_mm': 4., 'radius_mm': 1.}]))
    return out


def arrival_window(times, reference, frequency, level=.02, periods=2.):
    """(first, last) sample of the part of a record that holds the wave at a group of receivers (samples × receivers of the fine reference)."""
    envelope = np.max(np.abs(reference), axis=1); above = np.flatnonzero(envelope >= level * envelope.max()); pad = periods / frequency
    first = int(np.searchsorted(times, times[above[0]] - pad)); last = int(np.searchsorted(times, times[above[-1]] + pad, side='right')) - 1
    return max(0, first), min(len(times) - 1, max(last, first + 8))


def metrics(times, frequency, values, fine, ids):
    """Gate metrics of one receiver group on its arrival window, with the whole-record values next to them."""
    from .evaluate import case_metrics
    N, L, FN, FI = values['murnaghan'][:, ids], values['linear'][:, ids], fine['murnaghan'][:, ids], fine['murnaghan'][:, ids] - fine['linear'][:, ids]
    a, b = arrival_window(times, FN, frequency); cut = slice(a, b + 1)
    row = case_metrics(times[cut], frequency, N[cut], L[cut], FN[cut], FI[cut]); row['window_us'] = [float(times[a]), float(times[b])]
    row['whole_record'] = case_metrics(times, frequency, N, L, FN, FI)
    return row


def run_case(name, purpose, raw, directory, models=MODELS, waveforms=None):
    """Fine reference and every coarse model (learned and GLL) on one scene; the result dictionary. `waveforms`: a mapping that
    receives the projected receiver waveforms of every run."""
    from graphnet.inputs import network_scene
    from graphnet.model import load
    from graphnet.rollout import read_record, rollout
    from spectral3d.feature_mesh import make_mesh
    from .record import projected, simulate
    network = load(DEGREE, long_record=True); scene = network_scene(raw, DEGREE, network['material']); mesh = make_mesh(scene, DEGREE, FINE_H_MM); frequency = scene['source']['frequency_mhz']
    started = time.monotonic(); manifest = rollout(scene, mesh, network, Path(directory) / 'fine', {'verify': name}, time_plan='auto', record='compact', controls=PAIR)
    fine = projected(read_record(Path(directory) / 'fine'), scene); times = np.arange(len(fine['linear'])) * scene['acquisition']['sample_step_us']
    groups = {}
    for i, rx in enumerate(scene['receivers']): groups.setdefault(rx['name'].split(':')[0], []).append(i)
    if waveforms is not None: waveforms.update(times_us=times, **{f'fine:{k}': v for k, v in fine.items()})
    result = {'schema': SCHEMA, 'case': name, 'purpose': purpose, 'scene': raw, 'groups': {g: [scene['receivers'][i]['name'] for i in ids] for g, ids in groups.items()},
              'fine': {'h_mm': FINE_H_MM, 'degree': DEGREE, 'nodes': int(len(mesh['points'])), 'seconds': time.monotonic() - started, 'layer': manifest['time_plan']['scheme'],
                       'layer_step_us': manifest['layer_step_us'], 'evaluations_per_control': manifest['time_plan']['evaluations_per_control'], 'threads': manifest['layer_threads'],
                       'contact_pairs': int(len(mesh.get('contact_a', [])))}, 'runs': {}}
    for model_name, path in models.items():
        for learned in (True, False):
            tag = f'{model_name}:{"learned" if learned else "gll"}'; started = time.monotonic()
            try:
                coarse = simulate(raw, path, Path(directory) / tag.replace(':', '-'), network, learned, record='compact', controls=PAIR)
            except ValueError as error:
                result['runs'][tag] = {'error': str(error)}; continue
            values = projected(read_record(Path(directory) / tag.replace(':', '-')), scene); energy = read_record(Path(directory) / tag.replace(':', '-'))['energies']
            balance = max(float(np.max(np.abs(e[:, 1] + e[:, 2] + e[:, 4] - e[:, 3])) / max(np.max(np.abs(e[:, 3])), 1e-30)) for e in energy)
            row = {'nodes': coarse['nodes'], 'seconds': time.monotonic() - started, 'layer_step_us': coarse['layer_step_us'], 'evaluations_per_control': coarse['time_plan']['evaluations_per_control'],
                   'stability_number': coarse['time_plan']['stability_number'], 'energy_balance': balance, 'groups': {}}
            for g, ids in groups.items(): row['groups'][g] = metrics(times, frequency, values, fine, ids)
            if waveforms is not None: waveforms.update({f'{tag}:{k}': v for k, v in values.items()})
            row['passed'] = bool(all(m['passed'] for m in row['groups'].values())); result['runs'][tag] = row
    return result


def reference(name, h_mm, output):
    """How converged is the fine reference of a stored case? The same scene on a finer grid (h_mm), compared with the stored
    reference by the same gates; writes reference-<case>-h<h>.json."""
    from graphnet.inputs import network_scene
    from graphnet.model import load
    from graphnet.rollout import read_record, rollout
    from spectral3d.feature_mesh import make_mesh
    from .record import projected
    output = Path(output); stored = json.loads((output / f'{name}.json').read_text(encoding='utf-8')); data = np.load(output / f'{name}.npz')
    raw = copy.deepcopy(stored['scene']); raw['numerics']['h_mm'] = h_mm
    network = load(DEGREE, long_record=True); scene = network_scene(raw, DEGREE, network['material']); mesh = make_mesh(scene, DEGREE, h_mm); started = time.monotonic()
    with tempfile.TemporaryDirectory() as scratch:
        manifest = rollout(scene, mesh, network, Path(scratch) / 'finer', {'verify': name, 'h_mm': h_mm}, time_plan='auto', record='compact', controls=PAIR)
        finer = projected(read_record(Path(scratch) / 'finer'), scene)
    coarser = {law: data[f'fine:{law}'].astype(float) for law in PAIR}; times = data['times_us']; groups = {}
    for i, rx in enumerate(scene['receivers']): groups.setdefault(rx['name'].split(':')[0], []).append(i)
    result = {'schema': SCHEMA, 'case': name, 'purpose': 'convergence of the fine reference', 'reference_h_mm': stored['fine']['h_mm'], 'finer_h_mm': h_mm, 'finer_nodes': int(len(mesh['points'])),
              'finer_layer_step_us': manifest['layer_step_us'], 'seconds': time.monotonic() - started,
              'groups': {g: metrics(times, scene['source']['frequency_mhz'], coarser, finer, ids) for g, ids in groups.items()}}
    (output / f'reference-{name}-h{h_mm:g}.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')
    print(f"{name}: reference h {stored['fine']['h_mm']:g} mm against h {h_mm:g} mm ({result['finer_nodes']} nodes, {result['seconds']:.0f} s) |" + ' |'.join(
        f" {g}: {100 * m['worst_waveform_l2']:.2f} / {100 * m['worst_a1']:.2f} / {100 * m['worst_a2']:.2f} / {100 * m['worst_increment_a2']:.2f}" for g, m in result['groups'].items()), flush=True)


def run(output, only=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for name, (purpose, raw) in cases().items():
        if only and not any(part in name for part in only): continue
        target = output / f'{name}.json'
        if target.exists(): continue
        waveforms = {}
        with tempfile.TemporaryDirectory() as scratch: result = run_case(name, purpose, raw, scratch, waveforms=waveforms)
        np.savez_compressed(output / f'{name}.npz', **{k: v if k == 'times_us' else v.astype(np.float32) for k, v in waveforms.items()})   # single precision: 1e-7 of the peak
        target.write_text(json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')
        print(summary_line(result), flush=True)


def summary_line(result):
    line = f"{result['case']}: fine {result['fine']['nodes']} nodes {result['fine']['seconds']:.0f} s"
    for tag, row in result['runs'].items():
        if 'error' in row: line += f"\n  {tag:26s} refused: {row['error'][:70]}"; continue
        line += f"\n  {tag:26s} {row['nodes']:6d} nodes {row['seconds']:5.1f} s |" + ' |'.join(
            f" {g}: {100 * m['worst_waveform_l2']:.2f} / {100 * m['worst_a1']:.2f} / {100 * m['worst_a2']:.2f} / {100 * m['worst_increment_a2']:.2f} {'ok' if m['passed'] else 'FAIL'}" for g, m in row['groups'].items())
    return line


def report(results):
    """Print every stored case: worst waveform / A1 / A2 / increment-A2 errors (%) per receiver group."""
    for path in sorted(Path(results).glob('*.json')):
        if not path.name.startswith('reference-'): print(summary_line(json.loads(path.read_text(encoding='utf-8'))))


def stored(results):
    """Stored case results, in the order of `cases()` (then any others by name)."""
    found = {path.stem: json.loads(path.read_text(encoding='utf-8')) for path in sorted(Path(results).glob('*.json')) if not path.name.startswith('reference-')}
    order = [name for name in cases() if name in found]
    return [found[name] for name in order + sorted(set(found) - set(order))]


def table(results):
    """The stored results as Markdown: gates per receiver group, groups passed per kind of case, and cost."""
    rows = stored(results); tags = [f'{model}:{kind}' for model in MODELS for kind in ('learned', 'gll')]
    cell = lambda m: f"{100 * m['worst_waveform_l2']:.2f} / {100 * m['worst_a1']:.2f} / {100 * m['worst_a2']:.2f} / {100 * m['worst_increment_a2']:.1f} {'✓' if m['passed'] else '✗'}"
    lines = ['| 算例 | 接收点组 | ' + ' | '.join(tag.replace(':learned', ' 学习').replace(':gll', ' GLL') for tag in tags) + ' |', '|---|---|' + '---|' * len(tags)]
    counts = {}
    for r in rows:
        for g in r['groups']:
            lines.append(f"| {r['case']} | {g} | " + ' | '.join(cell(r['runs'][tag]['groups'][g]) if 'groups' in r['runs'].get(tag, {}) else '—' for tag in tags) + ' |')
            for tag in tags:
                if 'groups' not in r['runs'].get(tag, {}): continue
                kind = r['case'].split('-f')[0]; done = counts.setdefault(kind, {}).setdefault(tag, [0, 0]); done[0] += r['runs'][tag]['groups'][g]['passed']; done[1] += 1
    lines += ['', '| 算例类别 | ' + ' | '.join(tag.replace(':learned', ' 学习').replace(':gll', ' GLL') for tag in tags) + ' |', '|---|' + '---|' * len(tags)]
    for kind, by in counts.items(): lines.append(f'| {kind} | ' + ' | '.join(f'{by[tag][0]}/{by[tag][1]}' if tag in by else '—' for tag in tags) + ' |')
    lines += ['', '| 算例 | 细网格节点 | 细网格 s | ' + ' | '.join(f'{model} 节点 | s | 倍数' for model in MODELS) + ' |', '|---|---|---|' + '---|---|---|' * len(MODELS)]
    for r in rows:
        parts = []
        for model in MODELS:
            run = r['runs'].get(f'{model}:learned', {})
            parts.append(f"{run['nodes']} | {run['seconds']:.1f} | {r['fine']['seconds'] / run['seconds']:.0f}" if 'nodes' in run else '— | — | —')
        lines.append(f"| {r['case']} | {r['fine']['nodes']} | {r['fine']['seconds']:.0f} | " + ' | '.join(parts) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0]); sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('run'); a.add_argument('--output', type=Path, required=True); a.add_argument('--only', nargs='*')
    b = sub.add_parser('report'); b.add_argument('--results', type=Path, required=True)
    c = sub.add_parser('reference'); c.add_argument('--output', type=Path, required=True); c.add_argument('--case', required=True); c.add_argument('--h-mm', type=float, default=.8)
    d = sub.add_parser('table'); d.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run': run(args.output, args.only)
    elif args.command == 'table': print(table(args.results))
    elif args.command == 'reference': reference(args.case, args.h_mm, args.output)
    else: report(args.results)


if __name__ == '__main__': main()
