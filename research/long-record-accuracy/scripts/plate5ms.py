"""5 ms plane-strain plate: the software as it runs today (automatic time layer, compact record) against a converged
2D spectral reference. The 1 ms study (plate.py, analyze_plate.py) extrapolated to 5 ms; this measures it.

Same plate as plate.py: 48 × 2 mm aluminium, free ends, uniform end-face traction along (1, 0, 1) (S0 and A0), 0.2 MHz
5-cycle Hann, 1 MPa; receivers u_z on the top at x = 24, 32, 40 mm and u_x at mid-thickness at x = 24, 32 mm. In 5 ms S0
crosses the plate about 280 times and A0 about 145 times (1000 carrier periods).

    python research/long-record-accuracy/scripts/plate5ms.py reference 0.75 linear     # also stvk, murnaghan; and h 0.5
    python research/long-record-accuracy/scripts/plate5ms.py network 2 | 1.5 | 1 | auto
    python research/long-record-accuracy/scripts/plate5ms.py analyse                    # → results/plate-5ms.json

Reference: `spectral2d` (velocity Verlet, degree 4, CFL 0.35, no network) at h 0.75 and 0.5 mm with the step budget lifted,
then Richardson extrapolation in Δt² as in analyze_plate.py; the two levels' difference is its self-check. About 35 and
75 min per law on one core. Network: the 3D plate with symmetry on both y faces (exactly plane strain), the record path of
the application (graphnet.rollout, time_plan 'auto', compact record, all three controls), at fixed h or with the
automatic mesh; receivers at 0.1 μs. Waveforms are written to reference/ (not in git).
"""
import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
from plate import plate, TOP, MID   # noqa: E402

REF = ROOT / 'reference'; REF.mkdir(exist_ok=True)
DURATION, REF_STEP, NET_STEP = 5000., .5, .1
LAWS = ('linear', 'stvk', 'murnaghan')
WINDOW = 50.
CHECKPOINT_FRAMES = 200   # 100 μs of record


def reference(h, law, stop_after_frames=None, cfl=None):
    """spectral2d's velocity-Verlet loop (spectral2d.solver.solve, undamped branch, same internal force, step and loads), with a
    checkpoint every CHECKPOINT_FRAMES output frames so a long reference survives an interrupted process. Without an
    interruption the result is bit for bit what solve() returns (checked on a 30 μs record)."""
    import math
    from spectral2d.run import prepared_scene
    from spectral2d.mesh import make_mesh, interpolation
    from spectral2d.loads import prepare_excitation, receiver_points
    from spectral2d.solver import internal_force
    from physics.signals import evaluate
    tag = f'{law}-h{h:g}' + (f'-cfl{cfl:g}' if cfl else '')   # cfl: the same mesh with a smaller time step (checks the Δt² extrapolation)
    path = REF / f'plate5ms-{tag}.npy'; checkpoint = REF / f'plate5ms-{tag}.checkpoint.npz'
    if path.exists(): return
    s = plate(h, '2d'); s['constitutive'] = law
    scene = prepared_scene(s, 4)   # validated as the 1 ms scene (the scene rules stop at 1 ms), then extended
    scene['acquisition']['duration_us'] = DURATION; assert scene['acquisition']['sample_step_us'] == REF_STEP
    mesh = make_mesh(scene, 4, h); excitation = prepare_excitation(mesh, scene); receivers = receiver_points(scene)
    assert excitation['alpha'] is None, 'free ends: no absorbing layer'
    material = scene['material']; mass = mesh['mass']; cfl = cfl or scene['numerics']['cfl']
    constants = np.array([material[k] for k in ['lambda_pa', 'mu_pa', 'l_pa', 'm_pa', 'n_pa']]); code = {'linear': 0, 'stvk': 1, 'murnaghan': 2}[law]
    speed = math.sqrt((material['lambda_pa'] + 2 * material['mu_pa']) / material['rho_kg_m3'])
    sample = REF_STEP * 1e-6; subdivisions = math.ceil(sample / (cfl * mesh['min_node_spacing'] / (speed * 1.05))); dt = sample / subdivisions
    frames = round(DURATION / REF_STEP) + 1; steps = (frames - 1) * subdivisions
    loads = excitation['loads']; constrained = excitation['constrained']
    times = np.arange(steps + 1) * dt * 1e6; values = [evaluate(src, times) for _, src in loads]
    taps = [interpolation(mesh, rx['point_mm']) for rx in receivers]
    if checkpoint.exists():
        with np.load(checkpoint) as z:
            u, v, acc, probes, previous_ext = z['u'], z['v'], z['acc'], z['probes'], z['previous_ext']; first = int(z['step']) + 1; work = float(z['work']); spent = float(z['seconds'])
        print('reference', law, h, 'resumes at', f'{(first - 1) * dt * 1e6:.1f} μs', flush=True)
    else:
        u = np.zeros((len(mass), 2)); v = u.copy(); acc = u.copy(); previous_ext = u.copy(); probes = np.zeros((frames, len(taps), 2)); first = 1; work = 0.; spent = 0.
    started = time.time(); stored = 0.
    for step in range(first, steps + 1):
        next_u = u + dt * v + .5 * dt * dt * acc; next_u[constrained] = 0
        fint, energy, strain, jac = internal_force(next_u, mesh['elements'], mesh['derivative'], mesh['size'], mesh['area_weight'], constants, code)
        if not np.isfinite(energy) or strain > .001 or jac <= 0: raise ValueError(f'Elastic guard failed: strain={strain}, J={jac}')
        ext = loads[0][0] * values[0][step]
        for k in range(1, len(loads)): ext = ext + loads[k][0] * values[k][step]
        next_acc = (ext - fint) / mass[:, None]; next_acc[constrained] = 0
        next_v = v + .5 * dt * (acc + next_acc); next_v[constrained] = 0
        work += float(np.sum(.5 * (ext + previous_ext) * (next_u - u))); u, v, acc = next_u, next_v, next_acc; previous_ext = ext
        if step % subdivisions == 0:
            frame = step // subdivisions
            for i, (ids, w) in enumerate(taps): probes[frame, i] = w @ u[ids]
            stored = float(np.sum(.5 * mass[:, None] * v * v)) + energy
            if frame % CHECKPOINT_FRAMES == 0 and step < steps:
                np.savez(checkpoint.with_suffix('.tmp.npz'), u=u, v=v, acc=acc, probes=probes, previous_ext=previous_ext, step=step, work=work, seconds=spent + time.time() - started)
                checkpoint.with_suffix('.tmp.npz').replace(checkpoint)
                if stop_after_frames and frame >= stop_after_frames: return
    np.save(path, probes); checkpoint.unlink(missing_ok=True)
    (REF / f'plate5ms-{tag}.json').write_text(json.dumps({'h_mm': h, 'law': law, 'cfl': cfl, 'nodes': int(len(mass)), 'seconds': spent + time.time() - started, 'frames': frames,
                                                              'internal_step_ns': dt * 1e9, 'final_energy_balance': abs(stored - work) / max(abs(work), 1e-300)}, indent=1) + '\n')
    print('reference', law, h, f'{spent + time.time() - started:.0f} s', flush=True)


def network(h):
    from offline.domain import validate_request
    from spectral3d.feature_mesh import make_mesh
    from graphnet.model import load
    from graphnet.rollout import read_record, rollout
    auto = h == 'auto'
    if (REF / f"plate5ms-network-{'auto' if auto else f'h{float(h):g}'}.npz").exists(): return
    s = plate(1. if auto else float(h)); s['acquisition'] = {'duration_us': DURATION, 'sample_step_us': NET_STEP}
    request = validate_request({'scene': s, 'mesh': 'auto'} if auto else {'scene': s, 'degree': 4}); scene = request['scene']
    mesh = make_mesh(scene, request['degree'], scene['numerics']['h_mm']); model = load(request['degree'], long_record=True); started = time.monotonic()
    name = 'auto' if auto else f'h{float(h):g}'
    # what the application does for this request: compact record, automatic time layer. The record directory persists, so
    # an interrupted run resumes from its last chunk (graphnet.rollout checkpoints), as a paused job does in the application.
    directory = REF / 'records' / name
    manifest = rollout(scene, mesh, model, directory, {'case': f'plate5ms-{h}'}, time_plan='auto', record='compact', controls=LAWS, flush=True)
    record = read_record(directory)
    np.savez_compressed(REF / f'plate5ms-network-{name}.npz', times_us=record['times_us'], probes_m=record['probes'])
    energies = record['energies']
    balance = [float(np.abs(e[:, 1] + e[:, 2] + e[:, 4] - e[:, 3]).max() / np.abs(e[:, 3]).max()) for e in energies]
    info = {'h_mm': float(scene['numerics']['h_mm']), 'degree': int(request['degree']), 'nodes': int(len(mesh['points'])), 'mesh': request.get('mesh_report'),
            'time_layer': {k: manifest['time_plan'].get(k) for k in ('scheme', 'layer_step_us', 'evaluations_per_control', 'phase_error_bound_rad', 'stability_number')},
            'seconds': float(manifest.get('elapsed_s') or time.monotonic() - started), 'energy_balance': balance}
    (ROOT / 'results' / f'plate5ms-network-{name}.json').write_text(json.dumps(info, ensure_ascii=False, indent=1, default=float) + '\n', encoding='utf-8')
    print('network', name, json.dumps({k: info[k] for k in ('h_mm', 'nodes', 'time_layer', 'seconds')}, default=float), flush=True)
    shutil.rmtree(directory)


def component(u):
    """u_z at the top points, u_x at the mid-plane points (arrays carry the in-plane components last: 2D (x, z), 3D (x, y, z))."""
    return np.concatenate([u[:, :len(TOP), -1], u[:, len(TOP):, 0]], 1)


def responses(d):
    return {'linear': d['linear'], 'geometric': d['stvk'] - d['linear'], 'material': d['murnaghan'] - d['stvk'], 'full': d['murnaghan']}


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def compare(a, b, t):
    edges = np.arange(0, t[-1] + 1e-9, WINDOW)
    return {'window_rel_l2': [rel(a[(t >= lo) & (t < lo + WINDOW)], b[(t >= lo) & (t < lo + WINDOW)]) for lo in edges[:-1]],
            'cumulative_rel_l2': {f'{T:g}': rel(a[t <= T], b[t <= T]) for T in (1000., 2000., 3000., 4000., 5000.)},
            'last_50us_rel_l2': rel(a[t >= DURATION - WINDOW], b[t >= DURATION - WINDOW])}


def analyse():
    from analyze_plate import extrapolated, time_step
    levels = {h: responses({law: component(np.load(REF / f'plate5ms-{law}-h{h:g}.npy')) for law in LAWS}) for h in (.75, .5)}
    t = np.arange(len(levels[.5]['linear'])) * REF_STEP
    dt = {h: time_step(h) for h in levels}; ref = extrapolated(levels[.5], levels[.75], dt[.5], dt[.75])
    out = {'duration_us': DURATION, 'reference_dt_ns': {f'h{h:g}': dt[h] * 1e9 for h in dt},
           'reference_self_check': {k: compare(levels[.75][k], levels[.5][k], t) for k in ref},
           'reference_fine_vs_extrapolated': {k: compare(levels[.5][k], ref[k], t)['cumulative_rel_l2'] for k in ref},
           'windows_end_us': (np.arange(1, len(t) * REF_STEP / WINDOW + 1e-9) * WINDOW).tolist(), 'network': {}}
    half = REF / 'plate5ms-linear-h0.75-cfl0.175.npy'
    if half.exists():
        # The same h 0.75 mesh with half the time step: extrapolating in Δt alone must give the same reference if the
        # difference between the two mesh levels is the time step's (the assumption of the extrapolation above).
        dt_half = json.loads(half.with_suffix('.json').read_text())['internal_step_ns'] * 1e-9; r2 = (dt_half / dt[.75]) ** 2
        fine_step = component(np.load(half)); base = levels[.75]['linear']; time_only = fine_step + (fine_step - base) * r2 / (1 - r2)
        out['reference_time_step_check'] = {'dt_ns': {'h0.75': dt[.75] * 1e9, 'h0.75_half': dt_half * 1e9},
                                            'h0.75_vs_half_step': compare(base, fine_step, t)['cumulative_rel_l2'],
                                            'time_extrapolated_h0.75_vs_reference': compare(time_only, ref['linear'], t)}
        check = out['reference_time_step_check']['time_extrapolated_h0.75_vs_reference']
        print('reference check (linear): time-only extrapolation on h 0.75 vs the reference', {T: f'{v:.1e}' for T, v in check['cumulative_rel_l2'].items()}, f"last 50 μs {check['last_50us_rel_l2']:.1e}")
    every = round(REF_STEP / NET_STEP)
    for path in sorted(REF.glob('plate5ms-network-*.npz')):
        name = path.stem.split('-')[-1]
        with np.load(path) as z: probes = z['probes_m'][::every]
        assert len(probes) == len(t)
        sim = responses({law: component(probes[:, :, i]) for i, law in enumerate(LAWS)})
        info = json.loads((ROOT / 'results' / f'plate5ms-network-{name}.json').read_text(encoding='utf-8'))
        out['network'][name] = {'run': info, 'against_reference': {k: compare(sim[k], ref[k], t) for k in sim},
                                'against_fine_reference_level': {k: compare(sim[k], levels[.5][k], t)['cumulative_rel_l2'] for k in sim}}
        print(name, info['h_mm'], {k: {T: f'{v:.2e}' for T, v in c['cumulative_rel_l2'].items()} | {'last50': f"{c['last_50us_rel_l2']:.2e}"}
                                   for k, c in out['network'][name]['against_reference'].items()}, flush=True)
    print('reference h0.75 vs h0.5', {k: {T: f'{v:.1e}' for T, v in c['cumulative_rel_l2'].items()} for k, c in out['reference_self_check'].items()})
    (ROOT / 'results' / 'plate-5ms.json').write_text(json.dumps(out, indent=1) + '\n')
    figures(out, t, ref, levels)


def figures(out, t, ref, levels):
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed: no figures'); return
    ends = np.array(out['windows_end_us'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, key, title in ((axes[0], 'full', 'complete response'), (axes[1], 'material', 'material-nonlinearity increment')):
        for name, row in out['network'].items():
            ax.semilogy(ends / 1000, row['against_reference'][key]['window_rel_l2'], lw=1.2, label=f"WorkAI {name} (h {row['run']['h_mm']:g} mm)")
        if key == 'full' and 'reference_time_step_check' in out:   # the reference's own uncertainty: its two independent extrapolations (linear response)
            ax.semilogy(ends / 1000, out['reference_time_step_check']['time_extrapolated_h0.75_vs_reference']['window_rel_l2'], color='0.4', lw=1, ls=':',
                        label='reference uncertainty (two extrapolations)')
        ax.set_xlabel('t (ms)'); ax.set_ylabel('relative L2 in each 50 μs window'); ax.set_title(f'5 ms plate: {title}', fontsize=10); ax.grid(alpha=.3, which='both')
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ROOT / 'figures' / 'fig5-plate-5ms-error-vs-time.png', dpi=110); plt.close(fig)
    late = t >= DURATION - 100.
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    every = round(REF_STEP / NET_STEP)
    for ax, key, label in ((axes[0], 'full', 'complete response u_z, top x = 24 mm (nm)'), (axes[1], 'material', 'material increment u_z, top x = 24 mm (nm)')):
        ax.plot(t[late], ref[key][late, 0] * 1e9, color='k', lw=1.6, label='reference (extrapolated)')
        for name in out['network']:
            with np.load(REF / f'plate5ms-network-{name}.npz') as z: probes = z['probes_m'][::every]
            sim = responses({law: component(probes[:, :, i]) for i, law in enumerate(LAWS)})
            ax.plot(t[late], sim[key][late, 0] * 1e9, lw=1, ls='--', label=f'WorkAI {name}')
        ax.set_ylabel(label, fontsize=9); ax.grid(alpha=.3)
    axes[1].set_xlabel('t (μs)'); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ROOT / 'figures' / 'fig6-plate-5ms-late-waveform.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    if sys.argv[1] == 'reference': reference(float(sys.argv[2]), sys.argv[3], cfl=float(sys.argv[4]) if len(sys.argv) > 4 else None)
    elif sys.argv[1] == 'network': network(sys.argv[2])
    else: analyse()
