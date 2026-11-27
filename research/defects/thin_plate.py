"""A closed delamination at mid-depth of a thin plate, at low frequency: what the mesh needs.

    python research/defects/thin_plate.py fem 0.1 0.05 0.025 [case ...]     # the Q1 reference with exact contact (fem2d.py)
    python research/defects/thin_plate.py network                            # the frozen network on the variants of the mesh
    python research/defects/thin_plate.py analyse                            # → thin-plate-results.json

The case a user met (2026-11-27): a 200 × 200 × 2 mm plate driven continuously for 1 ms, with a closed delamination 10 mm
across at mid-depth. The defect mesher refines around a delamination front at ± h/4 across the delamination plane as
well; the mesh is a tensor grid, so those two grid lines cut the whole plate into four layers of elements through its
2 mm (17 nodes), the smallest node spacing falls from 0.345 mm to 0.065 mm and the time step with it: about seventeen
times the work of the plate without the defect, for a refinement the two layers of 1 mm hardly need.

Here the same situation in plane strain, where the independent reference exists: 120 × 2 mm, both ends free, a 45°
traction on the end x = 0 (S0 and A0), 0.1 MHz 5-cycle Hann, 1 MPa, 120 μs (the waves cross the plate and come back);
delamination x = 55–65 at 1 mm, closed (and open). Linear material: the nonlinearity is the contact alone. Receivers
on the top face before, above and behind the delamination and on the bottom face under it.
"""
import copy, json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import research.defects.plane_strain as base

OUT = HERE / 'out' / 'thin-plate'
LENGTH, THICKNESS, F_MHZ, CYCLES, DURATION, SAMPLE = 120., 2., .1, 5., 120., .1
RECEIVERS = [('top_30', 30., THICKNESS), ('top_60', 60., THICKNESS), ('top_90', 90., THICKNESS), ('bottom_60', 60., 0.)]
AT_DEFECT = ('top_60', 'bottom_60')
DELAMINATION = {'type': 'delamination', 'x_start_mm': 55., 'x_end_mm': 65., 'depth_mm': 1.}
CASES = {'T0-intact': [], 'T1-open-delamination': [DELAMINATION], 'T2-closed-delamination': [dict(DELAMINATION, closed=True)]}


def scene(defects, h, width=None):
    s = base.scene(defects, 1., width=width or h, sample_us=SAMPLE)
    s['geometry'].update(length_mm=LENGTH, thickness_mm=THICKNESS); s['source'].update(center_mm=[0., (width or h) / 2, THICKNESS / 2], frequency_mhz=F_MHZ, cycles=CYCLES)
    s['boundaries'] = [b for b in s['boundaries'] if b['type'] == 'symmetry']; s['acquisition']['duration_us'] = DURATION; s['numerics']['h_mm'] = h
    s['receivers'] = [{'name': n, 'point_mm': [x, (width or h) / 2, z], 'direction': [0., 0., 1.]} for n, x, z in RECEIVERS]
    return s


def fem_case(defects):
    splits = [{'axis': 'z', 'at': THICKNESS - d['depth_mm'], 'from': d['x_start_mm'], 'to': d['x_end_mm'], 'contact': d.get('closed', False)} for d in defects]
    return {'L': LENGTH, 'T': THICKNESS, 'rho': base.MATERIAL['rho_kg_m3'], 'lam': base.MATERIAL['lambda_pa'], 'mu': base.MATERIAL['mu_pa'], 'direction': [1., 1.], 'amplitude_mpa': 1.,
            'f_mhz': F_MHZ, 'cycles': CYCLES, 'duration_us': DURATION, 'sample_us': SAMPLE, 'absorber_mm': 0., 'splits': splits, 'receivers': RECEIVERS}


def fem(grids, cases):
    from research.defects.fem2d import simulate
    OUT.mkdir(parents=True, exist_ok=True)
    for h in grids:
        for name in cases:
            started = time.monotonic(); r = simulate(fem_case(CASES[name]), h)
            np.savez_compressed(OUT / f'{name}-fem{h:g}.npz', times_us=r['times_us'], probes_m=r['probes_m'], energy=r['energy'])
            info = {k: r[k] for k in ('dt_s', 'steps', 'nodes', 'elements', 'contact_pairs')}; info.update(h_mm=h, seconds=time.monotonic() - started)
            (OUT / f'{name}-fem{h:g}.json').write_text(json.dumps(info, indent=1) + '\n', encoding='utf-8'); print(name, 'fem', h, info, flush=True)


# (label, element size mm, FRONT_SPLIT_MIN): the rule up to 1.5.1 draws every refinement line (0), the new one leaves out a line
# through a layer that is no wider than twice the refinement distance (2).
VARIANTS = [('old-h2.5', 2.5, 0.), ('new-h2.5', 2.5, 2.), ('new-h4', 4., 2.), ('new-h5', 5., 2.), ('old-h1.25', 1.25, 0.), ('new-h1.25', 1.25, 2.)]


def network(cases, variants=VARIANTS):
    """The frozen network as the application runs it (validate_request → offline.worker → the record), on a strip one element
    wide (plane strain: both y faces are symmetry planes). Complete and linear controls: the linear one is the linear material
    with the contact, which is what the reference solves. The mesh rule and the largest element are set for the run."""
    import shutil
    from offline import automesh, worker
    from offline.domain import validate_request
    from offline.scan import receiver_series
    from spectral3d import feature_mesh
    OUT.mkdir(parents=True, exist_ok=True); kept = feature_mesh.FRONT_SPLIT_MIN, automesh.H_MAX
    try:
        for name in cases:
            for label, h, rule in variants:
                if not CASES[name] and rule != variants[0][2] and any(v[1] == h and v[2] != rule for v in variants): continue   # without a defect the rule changes nothing
                feature_mesh.FRONT_SPLIT_MIN = rule; automesh.H_MAX = max(kept[1], h)
                job = OUT / 'jobs' / f'{name}-{label}'; shutil.rmtree(job, ignore_errors=True); job.mkdir(parents=True)
                request = validate_request({'scene': scene(CASES[name], h), 'degree': 4, 'controls': 'full_linear', 'record': 'compact', 'time_stepping': 'auto'})
                (job / 'request.json').write_text(json.dumps(request)); (job / 'status.json').write_text(json.dumps({'job_id': job.name, 'phase': 'starting'}))
                started = time.monotonic(); worker.run(job); status = json.loads((job / 'status.json').read_text(encoding='utf-8'))
                if status.get('phase') != 'completed': raise RuntimeError(f'{name} {label}: {status.get("message")}')
                times, values, manifest = receiver_series(job / 'record', [r['point_mm'] for r in request['scene']['receivers']]); points = np.load(job / 'record' / 'mesh.npz')['points']
                z = np.unique(np.round(points[:, 2], 12)); plan = manifest['time_plan']
                np.savez_compressed(OUT / f'{name}-{label}.npz', times_us=np.asarray(times), probes_m=np.asarray(values['linear'])[:, :, [0, 2]])
                info = {'nodes': int(manifest['nodes']), 'h_mm': h, 'rule': rule, 'nodes_through_thickness': len(z), 'smallest_z_spacing_mm': float(np.diff(z).min() * 1e3), 'scheme': plan['scheme'],
                        'step_ns': plan['layer_step_us'] * 1e3, 'evaluations_per_control': plan['evaluations_per_control'], 'seconds': time.monotonic() - started,
                        'work_node_evaluations': int(manifest['nodes']) * plan['evaluations_per_control']}
                (OUT / f'{name}-{label}.json').write_text(json.dumps(info, indent=1, default=float) + '\n', encoding='utf-8'); shutil.rmtree(job, ignore_errors=True)
                print(name, label, info, flush=True)
    finally:
        feature_mesh.FRONT_SPLIT_MIN, automesh.H_MAX = kept


FEM_GRIDS = ('0.1', '0.05', '0.025')


def reference(name, receiver, component):
    """The finite-element trace extrapolated to zero element size (plane_strain.reference: the order is measured from three grids)."""
    grids = [g for g in FEM_GRIDS if (OUT / f'{name}-fem{g}.npz').exists()]; F = [np.load(OUT / f'{name}-fem{g}.npz')['probes_m'][:, receiver, component] for g in grids]
    order = float(np.clip(np.log2(np.linalg.norm(F[0] - F[1]) / np.linalg.norm(F[1] - F[2])), .5, 4.)) if len(F) == 3 else 2.
    E = F[-1] + (F[-1] - F[-2]) / (2 ** order - 1)
    return E, {'grids_mm': [float(g) for g in grids], 'order': order, 'finest_vs_extrapolated': float(np.linalg.norm(F[-1] - E) / np.linalg.norm(E))}


def spectrum_ratio(trace, k):
    """|X(k f)| / |X(f)| of a Hann-windowed trace."""
    n = 1 << 17; spec = np.abs(np.fft.rfft(trace * np.hanning(len(trace)), n)); freqs = np.fft.rfftfreq(n, SAMPLE); pick = lambda f: spec[np.argmin(np.abs(freqs - f))]
    return float(pick(k * F_MHZ) / pick(F_MHZ))


def analyse():
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b)); out = {'frequency_mhz': F_MHZ, 'plate_mm': [LENGTH, THICKNESS], 'cases': {}}
    for name in CASES:
        if not (OUT / f'{name}-fem{FEM_GRIDS[1]}.npz').exists(): continue
        row = {'reference': {}, 'variants': {}}
        refs = []
        for i, (receiver, _, _) in enumerate(RECEIVERS):
            trace, how = reference(name, i, 1); refs.append(trace); row['reference'][receiver] = dict(how, **{f'{k}f_over_f': spectrum_ratio(trace, k) for k in (2, 3)})
        for label, h, rule in VARIANTS:
            if not (OUT / f'{name}-{label}.npz').exists(): continue
            probes = np.load(OUT / f'{name}-{label}.npz')['probes_m']; info = json.loads((OUT / f'{name}-{label}.json').read_text(encoding='utf-8')); cells = {}
            for i, (receiver, _, _) in enumerate(RECEIVERS):
                trace = probes[:, i, 1]; low = lambda v: base.lowpass(v, SAMPLE, 3.5 * F_MHZ)
                cells[receiver] = {'waveform': rel(trace, refs[i]), 'waveform_to_3.5f': rel(low(trace), low(refs[i])),
                                   **{f'{k}f_over_f_vs_reference': spectrum_ratio(trace, k) / row['reference'][receiver][f'{k}f_over_f'] - 1 for k in (2, 3)}}
            # the harmonics are compared where the contact acts, above and below the delamination; elsewhere 2f and 3f are the burst's own side lobes
            near = [cells[r] for r in AT_DEFECT]
            row['variants'][label] = dict(info, receivers=cells, worst={'waveform': max(c['waveform'] for c in cells.values()), 'waveform_to_3.5f': max(c['waveform_to_3.5f'] for c in cells.values()),
                                                                        **{key: max((abs(c[key]) for c in near)) for key in ('2f_over_f_vs_reference', '3f_over_f_vs_reference')}})
        out['cases'][name] = row
        print(f'\n{name}：参考解 {row["reference"][RECEIVERS[0][0]]["grids_mm"]} mm，阶 {np.mean([r["order"] for r in row["reference"].values()]):.2f}，最细网格距外推 {max(r["finest_vs_extrapolated"] for r in row["reference"].values()):.2%}；'
              f'缺陷处（{"、".join(AT_DEFECT)}）2f/f = {" / ".join(f"{row['reference'][r]['2f_over_f']:.5f}" for r in AT_DEFECT)}，3f/f = {" / ".join(f"{row['reference'][r]['3f_over_f']:.5f}" for r in AT_DEFECT)}')
        print('| 网格 | 节点 | 板厚方向节点 | 时间步 ns | 每组控制的层数 | 计算量（相对） | 波形差 | 3.5 f 以下的波形差 | 缺陷处 2f/f 差 | 缺陷处 3f/f 差 |\n|---|---|---|---|---|---|---|---|---|---|')
        unit = min(v['work_node_evaluations'] for v in row['variants'].values()) if row['variants'] else 1
        for label, v in row['variants'].items():
            w = v['worst']; print(f"| {label} | {v['nodes']:,} | {v['nodes_through_thickness']} | {v['step_ns']:.2f} | {v['evaluations_per_control']:,} | {v['work_node_evaluations'] / unit:.1f} | {w['waveform']:.2%} | {w['waveform_to_3.5f']:.2%} | {w['2f_over_f_vs_reference']:.1%} | {w['3f_over_f_vs_reference']:.1%} |")
    (HERE / 'thin-plate-results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    if sys.argv[1] == 'fem':
        grids = [float(a) for a in sys.argv[2:] if a[0].isdigit()]; fem(grids, [a for a in sys.argv[2:] if not a[0].isdigit()] or list(CASES))
    elif sys.argv[1] == 'network': network(sys.argv[2:] or list(CASES))
    elif sys.argv[1] == 'analyse': analyse()
