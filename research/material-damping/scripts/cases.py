"""Cases of the material-damping verification, run exactly as the desktop software runs them (the harness of
research/literature-verification: validate_request → offline.worker → the record).

    WORKAI_VERIFY_JOBS=<dir> python research/material-damping/scripts/cases.py [name ...]

* B-*: a bar with symmetry on its four lateral faces — a plane longitudinal wave, the one-dimensional medium that has
  an exact solution with Rayleigh damping (damped_bar.py). 0.5 MHz, 5 cycles, end-face traction.
* P-*: S0 in a 1 mm plate (plane strain by symmetry on both y faces), 0.2 MHz.
"""
import hashlib, json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'literature-verification' / 'scripts'))
from harness import JOBS, REPO, scene


def run(name, s, degree=4, time_stepping='auto', controls='all'):
    """One job with a compact record (the receivers at every sample); a finished job with the same request is reused."""
    from offline.domain import validate_request
    job = JOBS / name; request = validate_request({'scene': s, 'degree': degree, 'controls': controls, 'record': 'compact', 'time_stepping': time_stepping})
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(); status = job / 'status.json'
    if status.exists() and json.loads(status.read_text()).get('phase') == 'completed' and (job / 'key').exists() and (job / 'key').read_text() == key: return job
    import shutil
    pending = job / 'key.pending'; before = 0.
    if pending.exists() and pending.read_text() == key and (job / 'record' / 'recording.json').exists():   # the same request, interrupted: the worker continues from its checkpoint
        before = float(json.loads(status.read_text()).get('elapsed_s') or 0.)
    else:
        shutil.rmtree(job, ignore_errors=True); job.mkdir(parents=True); pending.write_text(key)
    (job / 'request.json').write_text(json.dumps(request)); status.write_text(json.dumps({'job_id': name, 'phase': 'starting'})); started = time.monotonic() - before
    done = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'offline.worker', '--job-dir', str(job)], cwd=REPO, capture_output=True, text=True); st = json.loads(status.read_text())
    if done.returncode or st.get('phase') != 'completed': raise RuntimeError(f'{name}: {st.get("message")} {done.stderr[-1500:]}')
    (job / 'key').write_text(key); (job / 'elapsed_s').write_text(f'{time.monotonic() - started:.1f}'); return job


def load(name):
    """(times μs, {control: frames × receivers} along each receiver's direction, manifest of the record)."""
    import numpy as np
    from offline.scan import receiver_series
    request = json.loads((JOBS / name / 'request.json').read_text()); rx = request['scene']['receivers']
    times, values, manifest = receiver_series(JOBS / name / 'record', [r['point_mm'] for r in rx]); d = np.array([r['direction'] for r in rx])
    return np.asarray(times), {law: np.einsum('trc,rc->tr', np.asarray(v), d) for law, v in values.items()}, manifest

LATERAL = [{'type': 'symmetry', 'face': f} for f in ('ymin', 'ymax', 'zmin', 'zmax')]
X = [20., 40., 60., 80., 100., 120.]
# decay rates chosen so that the fundamental loses about 40 % over 100 mm: γ(0.5 MHz) = 32 000 1/s
ALPHA, BETA = 64000., 6.5e-9
DAMPING = {'none': None, 'mass': {'mass_per_s': ALPHA, 'stiffness_s': 0.}, 'stiffness': {'mass_per_s': 0., 'stiffness_s': BETA},
           'both': {'mass_per_s': ALPHA / 2, 'stiffness_s': BETA / 2}}


def bar(damping, amplitude=10., f=.5, cycles=5., h=.8, duration=40.):
    s = scene(200., .8, .8, h, duration, boundaries=LATERAL, source={'face': 'xmin', 'direction': [1., 0., 0.], 'amplitude_mpa': amplitude, 'frequency_mhz': f, 'cycles': cycles},
              receivers=[{'name': f'x{int(x)}', 'point_mm': [x, .4, .4], 'direction': [1., 0., 0.]} for x in X], name='damped bar')
    if DAMPING[damping]: s['damping'] = DAMPING[damping]
    return s


def plate(damping, f=.2, length=400., duration=110.):
    """S0 in a 1 mm plate: the upper half (mid-plane symmetry), plane strain, in-plane end traction, absorbing far end."""
    s = scene(length, 1., .5, 1., duration, boundaries=[{'type': 'symmetry', 'face': f_} for f_ in ('ymin', 'ymax', 'zmin')] + [{'type': 'absorbing', 'face': 'xmax', 'length_mm': 60., 'ramp': 'quadratic', 'attenuation_db': 60.}],
              source={'face': 'xmin', 'direction': [1., 0., 0.], 'amplitude_mpa': 5., 'frequency_mhz': f, 'cycles': 10.},
              receivers=[{'name': f'x{int(x)}', 'point_mm': [x, .5, .5], 'direction': [1., 0., 0.]} for x in range(40, 321, 40)], name='damped plate')
    if DAMPING[damping]: s['damping'] = DAMPING[damping]
    return s


CASES = {**{f'B-{name}': (lambda name=name: bar(name)) for name in DAMPING},
         'B-both-5MPa': lambda: bar('both', 5.), 'B-both-fine': lambda: bar('both', h=.5),
         'B-both-20cycles': lambda: bar('both', cycles=20., duration=64.), 'B-none-20cycles': lambda: bar('none', cycles=20., duration=64.),
         **{f'P-{name}': (lambda name=name: plate(name)) for name in ('none', 'both')}}

if __name__ == '__main__':
    for name in sys.argv[1:] or CASES:
        job = run(name, CASES[name]()); print(name, 'done', (job / 'elapsed_s').read_text(), flush=True)
