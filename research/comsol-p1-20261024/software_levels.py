"""The three P1 cases in the desktop software on a sequence of finer meshes, against the COMSOL baselines that exist.

    python research/comsol-p1-20261024/software_levels.py run        # compute (about 20 minutes on 4 cores)
    python research/comsol-p1-20261024/software_levels.py analyse    # write software-levels-20261120/levels.json and predictions/

Why: 15 of the 24 planned COMSOL runs are missing (StVK, and the time-step and mesh variants of COMSOL itself). Their
purpose is to say how much of the difference between the software and COMSOL in the second harmonic of the nonlinear
increment (0.06 %–2.7 % at the worst receiver) is COMSOL's own discretisation error. The same question can be put from
the software's side without COMSOL: refine the software's mesh and see whether the difference to COMSOL changes.

* A level raises the degree of the elements, then reduces their size (the sequence of the workbench's convergence
  check, one level further): (4, 1.5 mm) — the mesh of the spectral labels — then (5, 1.5), (5, 1.125), (5, 0.84375).
  The automatic time layer follows the mesh. All three constitutive controls are computed.
* Every level is compared with COMSOL (`coarsegraph.evaluate.case_metrics`, COMSOL as reference, the same metric as
  `compare_baselines.py`) and with the finest level of the software itself.
* The finest level is written in the delivery format (`predictions/<case>_<control>.csv`) with its StVK response:
  a prediction of the COMSOL runs that do not exist yet, fixed before they are made (`compare_remaining.py`).

Runs go through `offline.worker` exactly as the workbench runs them (explicit mesh, compact record: the receivers).
"""
import copy, hashlib, json, os, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent; OUT = HERE / 'software-levels-20261120'; BASELINES = HERE / 'serendipity-baselines-20261109' / 'data'
JOBS = Path(os.environ.get('WORKAI_P1_JOBS') or Path(tempfile.gettempdir()) / 'workai-p1-software-levels')
CASES = ('A-f0.40-T1.6', 'B-f0.40-T1.6', 'C-f0.25-T2.0')
LEVELS = ((4, 1.5), (5, 1.5), (5, 1.125), (5, .84375))
LAWS = ('linear', 'stvk', 'murnaghan')


def request(cid, degree, h):
    from coarsegraph.comsol import load_case
    case = load_case(cid); s = copy.deepcopy(case['scene_v2']); s['receiver_lines'] = []; s['numerics']['h_mm'] = h; s['name'] = f'{cid} · {degree} 阶 · {h:g} mm'
    s.pop('dimension', None); s.setdefault('constitutive', 'murnaghan')   # the case file's scene as the workbench stores it
    s['receivers'] = [{'name': r['name'], 'point_mm': r['point_mm'], 'direction': r['direction']} for r in case['output']['receivers']]   # the 13 receivers of the delivery, in its order
    return {'scene': s, 'degree': degree, 'controls': 'all', 'record': 'compact'}


def run(cid, degree, h):
    from offline.domain import validate_request
    job = JOBS / f'{cid}-p{degree}-h{h:g}'; checked = validate_request(request(cid, degree, h)); key = hashlib.sha256(json.dumps(checked, sort_keys=True).encode()).hexdigest(); status = job / 'status.json'
    if status.exists() and json.loads(status.read_text(encoding='utf-8')).get('phase') == 'completed' and (job / 'key').exists() and (job / 'key').read_text() == key: return job
    import shutil
    shutil.rmtree(job, ignore_errors=True); job.mkdir(parents=True)
    (job / 'request.json').write_text(json.dumps(checked), encoding='utf-8'); status.write_text(json.dumps({'job_id': job.name, 'name': checked['scene']['name'], 'phase': 'starting'}), encoding='utf-8')
    started = time.monotonic(); done = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'offline.worker', '--job-dir', str(job)], cwd=REPO, capture_output=True, text=True)
    state = json.loads(status.read_text(encoding='utf-8'))
    if done.returncode or state.get('phase') != 'completed': raise RuntimeError(f"{job.name}: {state.get('message')} {done.stderr[-1500:]}")
    (job / 'key').write_text(key); (job / 'elapsed_s').write_text(f'{time.monotonic() - started:.1f}')
    return job


def load(cid, degree, h):
    """(times μs, {control: frames × 13 receivers × 3 in m}, manifest of the record) of one finished level."""
    from coarsegraph.comsol import load_case
    from offline.scan import receiver_series
    job = JOBS / f'{cid}-p{degree}-h{h:g}'; points = [r['point_mm'] for r in load_case(cid)['output']['receivers']]
    times, values, m = receiver_series(job / 'record', points); return times, values, m, float((job / 'elapsed_s').read_text())


def analyse():
    from coarsegraph.comsol import export_csv, load_case, projected, read_csv
    from coarsegraph.evaluate import case_metrics, compare, vector_relative
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    out = {'schema': 'workai.comsol.p1.software_levels.v1', 'levels': [{'degree': p, 'h_mm': h} for p, h in LEVELS], 'reference': 'COMSOL serendipity baselines of 2026-11-09 (linear and murnaghan)',
           'metric': 'coarsegraph.evaluate.case_metrics, 10 % relative reporting floor, 0–30 μs, displacement along each receiver\'s direction', 'cases': {}}
    (OUT / 'predictions').mkdir(parents=True, exist_ok=True)
    for cid in CASES:
        case = load_case(cid); f = case['sources'][0]['frequency_mhz']; t, comsol_L = read_csv(BASELINES / f'{cid}_linear.csv', case); _, comsol_N = read_csv(BASELINES / f'{cid}_murnaghan.csv', case)
        CL, CN = projected(case, comsol_L), projected(case, comsol_N); loaded = [load(cid, p, h) for p, h in LEVELS]; finest = loaded[-1]
        assert all(np.allclose(l[0], t) for l in loaded), 'the records are not on the time grid of the delivery'
        P = lambda values, law: projected(case, values[law]); rows = []
        for (p, h), (_, values, m, elapsed) in zip(LEVELS, loaded):
            L, S, N = P(values, 'linear'), P(values, 'stvk'), P(values, 'murnaghan'); fL, fS, fN = (P(finest[1], law) for law in LAWS); plan = m.get('time_plan') or {}
            change = lambda a, b: {'worst_a2': compare(t, a, b, f, ('a2',))['a2']['worst'], 'a2_vector_relative': vector_relative(t, a, b, f, 2), 'l2': float(np.linalg.norm(a - b) / np.linalg.norm(b))}
            rows.append({'degree': p, 'h_mm': h, 'nodes': m['nodes'], 'layer_step_us': plan.get('layer_step_us'), 'scheme': plan.get('scheme'), 'elapsed_s': elapsed,
                         'against_comsol': case_metrics(t, f, N, L, CN, CN - CL),
                         'against_finest_level': {'record_l2': float(np.linalg.norm(N - fN) / np.linalg.norm(fN)), 'increment': change(N - L, fN - fL), 'geometric_increment': change(S - L, fS - fL), 'material_increment': change(N - S, fN - fS)}})
        gap = rows[-1]['against_comsol']; own = rows[-2]['against_finest_level']['increment']
        out['cases'][cid] = {'frequency_mhz': f, 'levels': rows, 'comsol_csv_sha256': {'linear': digest(BASELINES / f'{cid}_linear.csv'), 'murnaghan': digest(BASELINES / f'{cid}_murnaghan.csv')},
                             'summary': {'difference_to_comsol_worst_increment_a2': [r['against_comsol']['worst_increment_a2'] for r in rows],
                                         'difference_to_comsol_increment_a2_vector': [r['against_comsol']['increment_a2_vector_relative'] for r in rows],
                                         'change_to_finest_worst_increment_a2': [r['against_finest_level']['increment']['worst_a2'] for r in rows],
                                         'last_refinement_over_gap': own['worst_a2'] / gap['worst_increment_a2'] if gap['worst_increment_a2'] > 0 else None}}
        files = {}
        for law in LAWS:
            path = OUT / 'predictions' / f'{cid}_{law}.csv'; export_csv(case, t, finest[1][law], path, comment=f'WorkAI desktop software, frozen network, degree {LEVELS[-1][0]}, h {LEVELS[-1][1]:g} mm, automatic time layer; written before the COMSOL {law} run of tiers 2-4 existed'); files[law] = digest(path)
        out['cases'][cid]['prediction_csv_sha256'] = files
        s = out['cases'][cid]['summary']
        print(cid, '| worst increment 2f against COMSOL per level:', ' '.join(f'{v:.4%}' for v in s['difference_to_comsol_worst_increment_a2']), '| vector:', ' '.join(f'{v:.4%}' for v in s['difference_to_comsol_increment_a2_vector']),
              '| against the finest level:', ' '.join(f'{v:.2e}' for v in s['change_to_finest_worst_increment_a2']), '| last refinement / gap:', f"{s['last_refinement_over_gap']:.3g}", flush=True)
    (OUT / 'levels.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    if sys.argv[1:] == ['run']:
        for cid in CASES:
            for p, h in LEVELS: job = run(cid, p, h); print(job.name, (job / 'elapsed_s').read_text(), 's', flush=True)
    elif sys.argv[1:] == ['analyse']: analyse()
    else: raise SystemExit(__doc__)
