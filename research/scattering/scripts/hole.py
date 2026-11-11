"""S0 scattering from a through hole: network vs the plane-stress analytical solution (hole_theory.py).

Plate 48 × 48 × 1 mm, hole radius 4 mm at the centre, symmetry on both y faces (the incident wave is then an exact plane
S0 wave), uniform axial end traction (0.25 MHz, 2-cycle Hann, ka = 1.16). The mesh exceeds the worker's 30 000-node
display budget, so the record is computed by calling the same network rollout the worker calls (graphnet.rollout with
the validated scene and the long-record model) and read with the same offline.recording.traces.
Usage: hole.py run hole|plate
"""
import copy, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'literature-verification' / 'scripts'))
from harness import REPO  # noqa: E402

JOBS = Path(os.environ.get('WORKAI_VERIFY_JOBS') or Path(tempfile.gettempdir()) / 'workai-literature-verification')
L = W = 48.; T = 1.; X0 = Y0 = 24.; A = 4.; F = .25; H = 2.5
BASE = json.loads((REPO / 'coarsegraph/specs/labels-b-end-axial.json').read_text(encoding='utf-8'))['base_scene']
ANGLES = (180., 135., 90., 45., 0.)


def receivers():
    """Hole surface (r = a + 0.02 mm) at five angles and r = 2a at three, mid-plane; θ from +x (θ = 180° faces the source)."""
    pts = [(A + .02, th) for th in ANGLES] + [(2 * A, th) for th in (180., 90., 0.)]
    return [(r, th, [X0 + r * np.cos(np.radians(th)), Y0 + r * np.sin(np.radians(th)), T / 2]) for r, th in pts]


def scene(kind):
    s = copy.deepcopy(BASE); s['name'] = f'hole scattering {kind}'; s['receiver_lines'] = []
    s['geometry'].update(kind=kind, length_mm=L, width_mm=W, thickness_mm=T, hole_radius_mm=A, hole_x_mm=X0, hole_y_mm=Y0)
    s['source'] = {'face': 'xmin', 'center_mm': [0., W / 2, T / 2], 'radius_mm': 0., 'direction': [1., 0., 0.], 'amplitude_mpa': 1., 'frequency_mhz': F,
                   'cycles': 2., 'delay_us': 0., 'phase_deg': 0., 'signal': {'type': 'tone_burst', 'window': 'hann'}}
    s['boundaries'] = [{'type': 'symmetry', 'face': 'ymin'}, {'type': 'symmetry', 'face': 'ymax'}]
    s['acquisition'] = {'duration_us': 20., 'sample_step_us': .05}; s['numerics'] = {'h_mm': H, 'cfl': .3, 'max_elements': 40000}
    s['receivers'] = [{'name': f'r{r:g}t{th:g}', 'point_mm': [float(v) for v in p], 'direction': [1., 0., 0.]} for r, th, p in receivers()]
    return s


def run(kind):
    """The worker's network path without its display budget: validate_request → make_mesh → graphnet.rollout."""
    from offline.domain import validate_request
    from spectral3d.mesh import make_mesh
    from graphnet.model import load
    from graphnet.rollout import rollout
    job = JOBS / f'H-{kind}'; record = job / 'record'
    if (record / 'recording.json').exists() and json.loads((record / 'recording.json').read_text())['phase'] == 'completed': return job
    job.mkdir(parents=True, exist_ok=True)
    request = validate_request({'scene': scene(kind), 'degree': 4}); s = request['scene']; mesh = make_mesh(s, 4, s['numerics']['h_mm'])
    network = load(4, long_record=True); started = time.time()
    identity = {'weights_sha256': network['weights_sha256'], 'time_weights_sha256': network['time_weights_sha256'], 'input_sha256': 'verification'}
    print(kind, 'nodes', len(mesh['points']), flush=True)
    rollout(s, mesh, network, record, identity, callback=lambda m: print(kind, f"{m['completed_time_us']:.2f} μs", flush=True))
    (job / 'elapsed_s').write_text(f'{time.time() - started:.1f}'); return job


if __name__ == '__main__':
    job = run(sys.argv[2]); print('done', job)
