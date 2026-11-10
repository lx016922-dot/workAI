"""Error of 1 ms plate records (network, three meshes) against the 2D spectral reference (h 0.5 mm; h 0.75 mm bounds its error)."""
import json, os, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'literature-verification' / 'scripts'))
from harness import series  # noqa: E402
from plate import points, TOP, MID  # noqa: E402

HERE = Path(__file__).resolve().parents[1]; REF = HERE / 'reference'
JOBS = Path(os.environ.get('WORKAI_VERIFY_JOBS') or Path(tempfile.gettempdir()) / 'workai-literature-verification')
STEP, EVERY, WINDOW = .5, 5, 50.   # reference samples every 0.5 μs = every 5th network sample


def component(u):
    """Receiver components: u_z at the top points, u_x at the mid-plane points (2D arrays carry x, z)."""
    return np.concatenate([u[:, :len(TOP), -1], u[:, len(TOP):, 0]], 1)


def rel(a, b): return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def responses(d): return {'linear': d['linear'], 'geometric': d['stvk'] - d['linear'], 'material': d['murnaghan'] - d['stvk'], 'full': d['murnaghan']}


def reference(h): return responses({law: component(np.load(REF / f'plate-{law}-h{h:g}.npy')) for law in ('linear', 'stvk', 'murnaghan')})


def time_step(h):
    """The reference's internal step (velocity Verlet; spectral2d.solver picks it from CFL, the smallest node spacing and the sample step)."""
    import math
    from plate import plate
    from spectral2d.run import prepared_scene
    from spectral2d.mesh import make_mesh
    s = prepared_scene(plate(h, '2d'), 4); m = make_mesh(s, 4, h); mat = s['material']; c = math.sqrt((mat['lambda_pa'] + 2 * mat['mu_pa']) / mat['rho_kg_m3'])
    sample = s['acquisition']['sample_step_us'] * 1e-6; return sample / math.ceil(sample / (s['numerics']['cfl'] * m['min_node_spacing'] / (c * 1.05)))


def extrapolated(fine, coarse, dt_fine, dt_coarse):
    """Richardson extrapolation in the time step: velocity Verlet errors scale with Δt² (its spatial error is far smaller here)."""
    r2 = (dt_fine / dt_coarse) ** 2
    return {k: fine[k] + (fine[k] - coarse[k]) * r2 / (1 - r2) for k in fine}


def compare(a, b, t):
    edges = np.arange(0, t[-1] + 1e-9, WINDOW)
    return {'window_rel_l2': [rel(a[(t >= lo) & (t < lo + WINDOW)], b[(t >= lo) & (t < lo + WINDOW)]) for lo in edges[:-1]],
            'cumulative_rel_l2': {f'{T:g}': rel(a[t <= T], b[t <= T]) for T in (100., 250., 500., 1000.)}}


if __name__ == '__main__':
    fine, ref_coarse = reference(.5), reference(.75); t = np.arange(len(fine['linear'])) * STEP
    dt_fine, dt_coarse = time_step(.5), time_step(.75); ref = extrapolated(fine, ref_coarse, dt_fine, dt_coarse)
    # the comparison reference is the extrapolated one; the two levels' own difference is kept as the self-check
    out_dt = {'reference_dt_ns': {'h0.5': dt_fine * 1e9, 'h0.75': dt_coarse * 1e9}, 'reference_fine_vs_extrapolated': {k: compare(fine[k], ref[k], t)['cumulative_rel_l2'] for k in ref}}
    out = {**out_dt, 'reference_self_check': {k: compare(ref_coarse[k], fine[k], t) for k in ref}, 'windows_us': (np.arange(1, len(t) * STEP / WINDOW + 1e-9) * WINDOW).tolist()}
    late = t >= 900.; out['excerpt'] = {'t_us': t[late].tolist(), 'reference_full_top24': ref['full'][late, 0].tolist(), 'reference_material_top24': ref['material'][late, 0].tolist()}
    print('reference h0.75 vs h0.5', {k: {T: f'{v:.1e}' for T, v in c['cumulative_rel_l2'].items()} for k, c in out['reference_self_check'].items()})
    for h in sys.argv[1:] or ('2', '1.5', '1'):
        job = JOBS / f'P-h{h}'
        if not (job / 'key').exists(): print('skip', h); continue
        width = float(h); pts = points(width)
        sim = {}
        for part, law in (('linear', 'linear'), ('stvk', 'stvk'), ('full', 'murnaghan')):
            u = series(job, part, pts)[1][::EVERY]; sim[law] = np.concatenate([u[:, :len(TOP), 2], u[:, len(TOP):, 0]], 1)
        sim = responses(sim); out[f'h{h}'] = {k: compare(sim[k], ref[k], t) for k in sim}
        out[f'h{h}_vs_fine_reference'] = {k: compare(sim[k], fine[k], t)['cumulative_rel_l2'] for k in sim}
        out['excerpt'][f'h{h}_full_top24'] = sim['full'][late, 0].tolist(); out['excerpt'][f'h{h}_material_top24'] = sim['material'][late, 0].tolist()
        print(f'network h {h} mm', {k: {T: f'{v:.1e}' for T, v in c['cumulative_rel_l2'].items()} for k, c in out[f'h{h}'].items()})
    path = HERE / 'results' / 'plate-1ms.json'; old = json.loads(path.read_text()) if path.exists() else {}
    old.update(out); path.write_text(json.dumps(old, indent=1))
