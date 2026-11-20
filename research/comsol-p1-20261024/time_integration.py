"""COMSOL's time integration on the software's mesh: is the difference between the software and COMSOL its time step?

    python research/comsol-p1-20261024/time_integration.py run        # about an hour on 4 cores
    python research/comsol-p1-20261024/time_integration.py analyse    # write software-levels-20261120/time-integration.json

`software_levels.py` shows that the difference between the software and COMSOL in the second harmonic of the nonlinear
increment does not change when the software's mesh is refined. This script tests the obvious remaining cause without
COMSOL: the P1 runs integrate in time with generalized-α at ρ∞ = 1 and a step of 0.01 μs, a second-order scheme without
damping; the software's time layer keeps its phase error below 0.01 rad over the record by construction.

The repository's implicit reference solver (`spectral3d.implicit`: average-acceleration Newmark, the full hyperelastic
force converged in every step) is the same family of scheme — second order, no damping, the same leading period
error (ωΔt)²/12. It is run on the mesh of the spectral labels (GLL degree 4, 1.5 mm) with 5 and 10 steps per output
sample, i.e. Δt = 0.01 μs (COMSOL's base runs) and 0.005 μs (its dt2 runs), and for case A also 20 (0.0025 μs).

* If the Δt = 0.01 μs result is much closer to COMSOL than the software is, the difference was the time step.
* The Δt = 0.005 μs result is then what COMSOL's dt2 run should give, and the difference between the two is a
  prediction of `comsol_base_against_refined` in `compare_remaining.py`.

This is not the software's computation (which is explicit, on the frozen network); it is a numerical experiment with
the reference solver, used only to attribute the difference.
"""
import json, os, sys, tempfile, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from software_levels import BASELINES, CASES, JOBS, LEVELS, OUT, load, request

RUNS = JOBS / 'implicit'; DEGREE, H = LEVELS[0]
PLAN = [(cid, law, sub) for cid in CASES for sub in (5, 10) for law in ('linear', 'murnaghan')] + [('A-f0.40-T1.6', law, 20) for law in ('linear', 'murnaghan')]


def one(job):
    cid, law, sub = job; path = RUNS / f'{cid}-{law}-{sub}.npz'
    if path.exists(): return job, 0.
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    from spectral3d.feature_mesh import make_mesh
    from spectral3d.implicit import solve
    from spectral3d.loads import prepare_excitation
    from spectral3d.run import prepared_scene
    raw = request(cid, DEGREE, H)['scene']; raw['constitutive'] = law; scene = prepared_scene(raw, DEGREE); mesh = make_mesh(scene, DEGREE, scene['numerics']['h_mm'])
    started = time.monotonic(); out = solve(scene, mesh, substeps=sub, excitation=prepare_excitation(mesh, scene)); d = out['diagnostics']
    np.savez(path.with_suffix('.tmp.npz'), probes_m=out['probes_m'], times_us=out['times_us'], internal_step_us=d['internal_step_us'], nodes=d['nodes'], elapsed_s=time.monotonic() - started)
    path.with_suffix('.tmp.npz').replace(path); return job, time.monotonic() - started


def run():
    from concurrent.futures import ProcessPoolExecutor
    RUNS.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        for job, seconds in pool.map(one, sorted(PLAN, key=lambda j: -j[2])): print(*job, f'{seconds:.0f} s', flush=True)   # the longest first


def analyse():
    from coarsegraph.comsol import load_case, projected, read_csv
    from coarsegraph.evaluate import case_metrics
    out = {'schema': 'workai.comsol.p1.time_integration.v1', 'mesh': {'degree': DEGREE, 'h_mm': H}, 'scheme': 'average-acceleration Newmark (spectral3d.implicit), full hyperelastic force converged in every step',
           'comsol': 'generalized-α, ρ∞ = 1, Δt = 0.01 μs (base runs)', 'metric': 'coarsegraph.evaluate.case_metrics; the second argument of each comparison is the reference', 'cases': {}}
    keys = ('worst_increment_a2', 'increment_a2_vector_relative', 'increment_l2', 'worst_waveform_l2', 'record_l2')
    for cid in CASES:
        case = load_case(cid); f = case['sources'][0]['frequency_mhz']; t, CL = read_csv(BASELINES / f'{cid}_linear.csv', case); _, CN = read_csv(BASELINES / f'{cid}_murnaghan.csv', case); CL, CN = projected(case, CL), projected(case, CN)
        _, values, _, _ = load(cid, DEGREE, H); SL, SN = projected(case, values['linear']), projected(case, values['murnaghan'])
        _, finest, _, _ = load(cid, *LEVELS[-1]); FL, FN = projected(case, finest['linear']), projected(case, finest['murnaghan'])
        pick = lambda m: {k: m[k] for k in keys}; rows = []; got = {}
        for sub in sorted({s for c, _, s in PLAN if c == cid}):
            files = [RUNS / f'{cid}-{law}-{sub}.npz' for law in ('linear', 'murnaghan')]
            if not all(p.exists() for p in files): continue
            L, N = (projected(case, np.load(p)['probes_m']) for p in files); got[sub] = (L, N); meta = np.load(files[1])
            rows.append({'steps_per_sample': sub, 'time_step_us': float(meta['internal_step_us']), 'nodes': int(meta['nodes']), 'elapsed_s_murnaghan': float(meta['elapsed_s']),
                         'against_comsol': pick(case_metrics(t, f, N, L, CN, CN - CL)), 'against_software_same_mesh': pick(case_metrics(t, f, N, L, SN, SN - SL))})
        entry = {'frequency_mhz': f, 'software_against_comsol': {'same_mesh': pick(case_metrics(t, f, SN, SL, CN, CN - CL)), 'finest_mesh': pick(case_metrics(t, f, FN, FL, CN, CN - CL))}, 'implicit': rows}
        if 5 in got and 10 in got: entry['predicted_comsol_base_against_dt2'] = pick(case_metrics(t, f, got[5][1], got[5][0], got[10][1], got[10][1] - got[10][0]))
        out['cases'][cid] = entry; show = lambda m: f"{m['worst_increment_a2']:.3%} / {m['increment_a2_vector_relative']:.3%}"
        print(cid, '| software against COMSOL (worst / vector):', show(entry['software_against_comsol']['same_mesh']), '| implicit against COMSOL:', '  '.join(f"Δt {r['time_step_us']:g}: {show(r['against_comsol'])}" for r in rows),
              '| implicit against the software:', '  '.join(f"Δt {r['time_step_us']:g}: {show(r['against_software_same_mesh'])}" for r in rows), flush=True)
    (OUT / 'time-integration.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    if sys.argv[1:] == ['run']: run()
    elif sys.argv[1:] == ['analyse']: analyse()
    else: raise SystemExit(__doc__)
