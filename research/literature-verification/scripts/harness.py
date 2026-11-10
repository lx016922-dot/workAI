"""Run scenes exactly as the offline workbench does: validate_request -> offline.worker (subprocess) -> offline.recording.traces."""
import copy, hashlib, json, os, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent; RESULTS = HERE.parent / 'results'
# Job records are large (0.5–2 GB each): keep them outside the repository.
JOBS = Path(os.environ.get('WORKAI_VERIFY_JOBS') or Path(tempfile.gettempdir()) / 'workai-literature-verification')
from offline.domain import validate_request
from offline.recording import traces

BASE = json.loads((REPO / 'coarsegraph/specs/labels-b-end-axial.json').read_text(encoding='utf-8'))['base_scene']


def scene(length, width, thickness, h, duration, step=.05, boundaries=(), source=None, sources=(), material=None, receivers=None, name='verification'):
    s = copy.deepcopy(BASE); s['name'] = name
    s['geometry'].update(length_mm=float(length), width_mm=float(width), thickness_mm=float(thickness))
    s['boundaries'] = list(boundaries); s['receiver_lines'] = []
    s['source'] = dict(s['source'], center_mm=[0., width / 2, thickness / 2], **(source or {}))
    if sources: s['sources'] = list(sources)
    if material: s['material'] = material
    s['acquisition'] = {'duration_us': float(duration), 'sample_step_us': float(step)}
    s['numerics'] = {'h_mm': float(h), 'cfl': .3, 'max_elements': 40000}
    s['receivers'] = receivers or [{'name': 'R', 'point_mm': [length / 2, width / 2, thickness / 2], 'direction': [1., 0., 0.]}]
    return s


def run(name, s, degree=4):
    """Run once; reuse a completed job with the identical request."""
    job = JOBS / name; request = validate_request({'scene': s, 'degree': degree})
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    status = job / 'status.json'
    if status.exists() and json.loads(status.read_text()).get('phase') == 'completed' and (job / 'key').exists() and (job / 'key').read_text() == key:
        return job
    job.mkdir(parents=True, exist_ok=True)
    for p in job.iterdir():
        if p.is_file(): p.unlink()
    import shutil; shutil.rmtree(job / 'record', ignore_errors=True)
    (job / 'request.json').write_text(json.dumps(request)); status.write_text(json.dumps({'job_id': name, 'phase': 'starting'}))
    started = time.monotonic()
    done = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'offline.worker', '--job-dir', str(job)], cwd=REPO, capture_output=True, text=True)
    st = json.loads(status.read_text())
    if done.returncode or st.get('phase') != 'completed': raise RuntimeError(f'{name}: {st.get("message")} {done.stderr[-1500:]}')
    (job / 'key').write_text(key); (job / 'elapsed_s').write_text(f'{time.monotonic() - started:.1f}')
    return job


def series(job, part, points):
    """(times_us, values[T, n, 3]) at arbitrary points, read the way the result page reads them (8 points per call)."""
    out = []; times = None
    for i in range(0, len(points), 8):
        t = traces(job / 'record', part, [list(map(float, p)) for p in points[i:i + 8]])
        times = np.asarray(t['times_us']); out += [np.asarray(p['xyz_m']) for p in t['points']]
    return times, np.stack(out, 1)
