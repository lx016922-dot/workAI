"""A circular delamination: the network on its stepped grid against an independent axisymmetric reference.

    python research/defects/circular_delamination.py fem 0.1 0.05 0.025 0.0125 [case ...]   # the axisymmetric Q1 reference (axisym.py)
    python research/defects/circular_delamination.py network [variant ...]            # the frozen network, three-dimensional
    python research/defects/circular_delamination.py analyse                          # → circular-delamination-results.json

The mesh of the network is a tensor grid, so the edge of a circular delamination is a staircase: the grid faces whose
centres lie in the circle, with the area of the circle (`spectral3d.feature_mesh`). How fine must the steps be?

The test is a problem whose exact reduction to two dimensions is known: a 100 × 100 × 2 mm aluminium plate, a normal
load with a Gaussian profile (σ 2 mm) at the centre of the top face, a delamination 10 mm across centred under it.
Everything is axisymmetric until the waves reach the plate edges, so until then the solution is that of the (r, z)
problem, which `axisym.py` solves with other elements, another time step and an exact contact condition. 0.2 MHz
2-cycle Hann, 16 μs (the fastest wave returns from the edge to r = 10 mm after 16.7 μs). The load pushes and pulls on
the layer above the delamination, so a closed one opens and claps.

Receivers on the top face at r = 0, 2.5, 5, 7.5, 10 mm along x and at the same radii in the direction (3, 4) / 5, which
is not a direction of the grid: their difference is what the staircase adds; and on the bottom face at r = 0 and 5.
"""
import copy, json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import research.defects.plane_strain as base

OUT = HERE / 'out' / 'circular-delamination'
SIDE, THICKNESS, F_MHZ, CYCLES, DURATION, SAMPLE, SIGMA, DIAMETER, AMPLITUDE, RADIUS_FEM = 100., 2., .2, 2., 16., .05, 2., 10., .01, 60.
RADII = (0., 2.5, 5., 7.5, 10.)
RECEIVERS = [(f'top_x_{r:g}', r, 0., THICKNESS) for r in RADII] + [(f'top_d_{r:g}', .6 * r, .8 * r, THICKNESS) for r in RADII[1:]] + [('bottom_x_0', 0., 0., 0.), ('bottom_x_5', 5., 0., 0.)]
AT_DEFECT = ('top_x_0', 'top_x_2.5', 'bottom_x_0')
circle = lambda depth, closed: [{'type': 'delamination', 'shape': 'circle', 'x_mm': SIDE / 2, 'y_mm': SIDE / 2, 'diameter_mm': DIAMETER, 'depth_mm': depth, 'closed': closed}]
CASES = {'C0-intact': [], 'C1-open-mid': circle(1., False), 'C2-closed-mid': circle(1., True), 'C3-closed-shallow': circle(.5, True)}
# (label, element size mm, CIRCLE_STEP): 2.07 mm is the automatic element of a 0.2 MHz record of 1 ms
# (label, element size mm, CIRCLE_STEP, FRONT_SPLIT_MIN, CIRCLE_BIAS). step-n: steps of about h / n along the edge. bias: the disc of faces
# larger than the circle by that fraction of a step (0: the same area). z4: lines across the delamination plane as well (four layers of elements
# through the 2 mm), which the mesh rule leaves out when the layers are thinner than twice the refinement distance.
VARIANTS = [('step-2', 2.07, .5, 2., 0.), ('step-4', 2.07, .25, 2., 0.), ('step-8', 2.07, .125, 2., 0.), ('z4-step-4', 2.07, .25, 0., 0.), ('fine-step-4', 1.035, .25, 2., 0.)]
BIASES = (.1, .15, .2)
BIASED = lambda bias: [(f'{label}-bias{bias:g}', h, step, rule, bias) for label, h, step, rule, _ in VARIANTS]


def scene(defects, h):
    s = base.scene(defects, AMPLITUDE, width=SIDE, sample_us=SAMPLE)
    s['geometry'].update(length_mm=SIDE, thickness_mm=THICKNESS)
    # 1 Pa-scale load: the material nonlinearity of the complete control is then below 1e-8 of the response, and the contact does not depend on the amplitude
    s['source'].update(face='zmax', center_mm=[SIDE / 2, SIDE / 2, THICKNESS], direction=[0., 0., -1.], frequency_mhz=F_MHZ, cycles=CYCLES, profile={'type': 'gaussian', 'sigma_mm': SIGMA})
    s['constitutive'] = 'murnaghan'; s['boundaries'] = []; s['acquisition']['duration_us'] = DURATION; s['numerics']['h_mm'] = h
    s['receivers'] = [{'name': n, 'point_mm': [SIDE / 2 + x, SIDE / 2 + y, z], 'direction': [0., 0., 1.]} for n, x, y, z in RECEIVERS]
    return s


def fem_case(defects):
    splits = [{'at': THICKNESS - d['depth_mm'], 'radius': d['diameter_mm'] / 2, 'contact': d.get('closed', False)} for d in defects]
    radial = sorted({(round(math.hypot(x, y), 9), z) for _, x, y, z in RECEIVERS})
    return {'R': RADIUS_FEM, 'T': THICKNESS, 'rho': base.MATERIAL['rho_kg_m3'], 'lam': base.MATERIAL['lambda_pa'], 'mu': base.MATERIAL['mu_pa'], 'amplitude_mpa': AMPLITUDE, 'sigma_mm': SIGMA,
            'f_mhz': F_MHZ, 'cycles': CYCLES, 'duration_us': DURATION, 'sample_us': SAMPLE, 'splits': splits, 'receivers': [(f'{r:g}@{z:g}', r, z) for r, z in radial]}, radial


def fem(grids, cases):
    from research.defects.axisym import simulate
    OUT.mkdir(parents=True, exist_ok=True)
    for h in grids:
        for name in cases:
            case, radial = fem_case(CASES[name]); started = time.monotonic(); r = simulate(case, h)
            np.savez_compressed(OUT / f'{name}-fem{h:g}.npz', times_us=r['times_us'], uz_m=r['probes_m'][:, :, 1], radial=np.array(radial), closed_pairs=r['closed_pairs'])
            info = {k: r[k] for k in ('dt_s', 'steps', 'nodes', 'elements', 'contact_pairs')}; info.update(h_mm=h, seconds=time.monotonic() - started)
            (OUT / f'{name}-fem{h:g}.json').write_text(json.dumps(info, indent=1) + '\n', encoding='utf-8'); print(name, 'fem', h, info, flush=True)


def network(cases, variants):
    """The frozen network as the application runs it (validate_request → offline.worker → the record): the complete control alone."""
    import shutil
    from offline import worker
    from offline.domain import validate_request
    from offline.scan import receiver_series
    from spectral3d import feature_mesh
    OUT.mkdir(parents=True, exist_ok=True); kept = feature_mesh.CIRCLE_STEP, feature_mesh.FRONT_SPLIT_MIN, feature_mesh.CIRCLE_BIAS
    try:
        for name in cases:
            for label, h, step, rule, bias in variants:
                if not CASES[name] and label != next(v[0] for v in variants if v[1] == h): continue   # without a defect the rules change nothing
                feature_mesh.CIRCLE_STEP, feature_mesh.FRONT_SPLIT_MIN, feature_mesh.CIRCLE_BIAS = step, rule, bias
                job = OUT / 'jobs' / f'{name}-{label}'; shutil.rmtree(job, ignore_errors=True); job.mkdir(parents=True)
                request = validate_request({'scene': scene(CASES[name], h), 'degree': 4, 'controls': 'full', 'record': 'compact', 'time_stepping': 'auto'})
                (job / 'request.json').write_text(json.dumps(request)); (job / 'status.json').write_text(json.dumps({'job_id': job.name, 'phase': 'starting'}))
                started = time.monotonic(); worker.run(job); status = json.loads((job / 'status.json').read_text(encoding='utf-8'))
                if status.get('phase') != 'completed': raise RuntimeError(f'{name} {label}: {status.get("message")}')
                times, values, manifest = receiver_series(job / 'record', [r['point_mm'] for r in request['scene']['receivers']]); mesh = np.load(job / 'record' / 'mesh.npz')
                x = np.unique(np.round(mesh['points'][:, 0], 12)); z = np.unique(np.round(mesh['points'][:, 2], 12)); plan = manifest['time_plan']
                area = float(mesh['contact_area'].sum() * 1e6) if 'contact_area' in mesh.files and len(mesh['contact_area']) else None
                np.savez_compressed(OUT / f'{name}-{label}.npz', times_us=np.asarray(times), uz_m=np.asarray(next(iter(values.values())))[:, :, 2])
                info = {'nodes': int(manifest['nodes']), 'h_mm': h, 'circle_step': step, 'front_split_min': rule, 'circle_bias': bias, 'faces': feature_mesh.circle_lattice(DIAMETER / 2, step * h)[:2] if CASES[name] else None, 'nodes_through_thickness': len(z), 'smallest_spacing_mm': float(min(np.diff(x).min(), np.diff(z).min()) * 1e3),
                        'contact_area_mm2': area, 'circle_area_mm2': math.pi * DIAMETER ** 2 / 4, 'scheme': plan['scheme'], 'step_ns': plan['layer_step_us'] * 1e3,
                        'evaluations_per_control': plan['evaluations_per_control'], 'seconds': time.monotonic() - started, 'work_node_evaluations': int(manifest['nodes']) * plan['evaluations_per_control']}
                (OUT / f'{name}-{label}.json').write_text(json.dumps(info, indent=1, default=float) + '\n', encoding='utf-8'); shutil.rmtree(job, ignore_errors=True)
                print(name, label, info, flush=True)
    finally:
        feature_mesh.CIRCLE_STEP, feature_mesh.FRONT_SPLIT_MIN, feature_mesh.CIRCLE_BIAS = kept


FEM_GRIDS = ('0.1', '0.05', '0.025', '0.0125')


def reference(name):
    """The axisymmetric traces of every receiver, extrapolated to zero element size (the order is measured from three grids)."""
    grids = [g for g in FEM_GRIDS if (OUT / f'{name}-fem{g}.npz').exists()][-3:]; data = [np.load(OUT / f'{name}-fem{g}.npz') for g in grids]
    radial = [tuple(row) for row in np.round(data[0]['radial'], 9)]; traces, how = [], []
    for _, x, y, z in RECEIVERS:
        k = radial.index((round(math.hypot(x, y), 9), z)); F = [d['uz_m'][:, k] for d in data]
        order = float(np.clip(np.log2(np.linalg.norm(F[0] - F[1]) / np.linalg.norm(F[1] - F[2])), .5, 4.)) if len(F) == 3 else 2.
        E = F[-1] + (F[-1] - F[-2]) / (2 ** order - 1); traces.append(E); how.append({'order': order, 'finest_vs_extrapolated': float(np.linalg.norm(F[-1] - E) / np.linalg.norm(E))})
    return traces, how, [float(g) for g in grids]


def spectrum_ratio(trace, k):
    """|X(k f)| / |X(f)| of a Hann-windowed trace."""
    n = 1 << 16; spec = np.abs(np.fft.rfft(trace * np.hanning(len(trace)), n)); freqs = np.fft.rfftfreq(n, SAMPLE); pick = lambda f: spec[np.argmin(np.abs(freqs - f))]
    return float(pick(k * F_MHZ) / pick(F_MHZ))


def time_stretch(trace, ref):
    """s − 1 of the s for which trace(t) is closest to ref(s t): positive when the computed motion runs ahead of the reference (too stiff)."""
    t = np.arange(len(ref)) * SAMPLE; scales = np.linspace(.9, 1.1, 801)
    return float(scales[int(np.argmin([np.linalg.norm(trace - np.interp(t * s, t, ref)) for s in scales]))] - 1)


def analyse():
    ALL = VARIANTS + [v for b in BIASES for v in BIASED(b)]
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b)); out = {'frequency_mhz': F_MHZ, 'plate_mm': [SIDE, SIDE, THICKNESS], 'diameter_mm': DIAMETER, 'cases': {}}
    names = [n for n, *_ in RECEIVERS]
    for name in CASES:
        if not (OUT / f'{name}-fem{FEM_GRIDS[1]}.npz').exists(): continue
        refs, how, grids = reference(name); row = {'reference': {'grids_mm': grids, 'order': float(np.mean([h['order'] for h in how])), 'finest_vs_extrapolated': max(h['finest_vs_extrapolated'] for h in how),
                                                                  **{f'{k}f_over_f': {r: spectrum_ratio(refs[names.index(r)], k) for r in AT_DEFECT} for k in (2, 3)}}, 'variants': {}}
        for label, h, step, rule, bias in ALL:
            if not (OUT / f'{name}-{label}.npz').exists(): continue
            uz = np.load(OUT / f'{name}-{label}.npz')['uz_m']; info = json.loads((OUT / f'{name}-{label}.json').read_text(encoding='utf-8'))
            cells = {r: rel(uz[:, i], refs[i]) for i, r in enumerate(names)}; stretch = {r: time_stretch(uz[:, names.index(r)], refs[names.index(r)]) for r in AT_DEFECT}
            pairs = {f'{r:g}': rel(uz[:, names.index(f'top_d_{r:g}')], uz[:, names.index(f'top_x_{r:g}')]) for r in RADII[1:]}   # the same radius in two directions
            harmonics = {f'{k}f_over_f_vs_reference': {r: spectrum_ratio(uz[:, names.index(r)], k) / row['reference'][f'{k}f_over_f'][r] - 1 for r in AT_DEFECT} for k in (2, 3)}
            row['variants'][label] = dict(info, waveform=cells, two_directions=pairs, time_stretch=stretch, **harmonics,
                                          worst={'waveform': max(cells.values()), 'waveform_over_defect': max(cells[r] for r in names if r.split('_')[-1] in ('0', '2.5')),
                                                 'two_directions': max(pairs.values()), **{k: max(abs(v) for v in harmonics[k].values()) for k in harmonics}})
        out['cases'][name] = row; ref = row['reference']
        print(f'\n{name}：参考解 {grids} mm，阶 {ref["order"]:.2f}，最细网格距外推 {ref["finest_vs_extrapolated"]:.2%}；缺陷处 2f/f = ' + ' / '.join(f'{v:.4f}' for v in ref['2f_over_f'].values()) + '，3f/f = ' + ' / '.join(f'{v:.4f}' for v in ref['3f_over_f'].values()))
        print('| 网格 | 节点 | 板厚方向节点 | 接触面积 mm² | 最小节点间距 mm | 时间步 ns | 计算量（相对） | 波形差（最大） | 缺陷上方的波形差 | 两个方向同半径的差 | 时间快慢（中心） | 2f/f 差 | 3f/f 差 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        unit = min(v['work_node_evaluations'] for v in row['variants'].values()) if row['variants'] else 1
        for label, v in row['variants'].items():
            w = v['worst']; area = f"{v['contact_area_mm2']:.2f}" if v['contact_area_mm2'] else '—'
            print(f"| {label} | {v['nodes']:,} | {v['nodes_through_thickness']} | {area} | {v['smallest_spacing_mm']:.3f} | {v['step_ns']:.2f} | {v['work_node_evaluations'] / unit:.1f} | {w['waveform']:.2%} | {w['waveform_over_defect']:.2%} | {w['two_directions']:.2%} | {v['time_stretch']['top_x_0']:+.2%} | {w['2f_over_f_vs_reference']:.1%} | {w['3f_over_f_vs_reference']:.1%} |")
    (HERE / 'circular-delamination-results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == 'fem':
        grids = [float(a) for a in args if a[0].isdigit()]; fem(grids, [a for a in args if not a[0].isdigit()] or list(CASES))
    elif mode == 'network':
        every = VARIANTS + [v for b in BIASES for v in BIASED(b)]; chosen = [v for v in every if v[0] in args] or VARIANTS[:2]; network([a for a in args if a in CASES] or list(CASES), chosen)
    else:
        analyse()
