"""Guided waves in a pipe and their reflection at the free pipe end: network runs.

Aluminium pipe, outer radius 3 mm, wall 1 mm, length 60 mm, both ends free. The xmin end face carries a uniform
traction: axial (excites the axisymmetric L(0,m) modes, circumferential order n = 0) or along y (excites the flexural
F(1,m) modes, n = 1). The mesh exceeds the worker's 30 000-node display budget, so the record is computed by calling
the same network rollout the worker calls (graphnet.rollout with the validated scene and the long-record model) and read
with the same offline.recording.traces. The independent reference is pipe_reference.py (2.5D, one circumferential order).
Usage: pipe.py run CASE H
"""
import copy, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'literature-verification' / 'scripts'))
from harness import REPO  # noqa: E402

JOBS = Path(os.environ.get('WORKAI_VERIFY_JOBS') or Path(tempfile.gettempdir()) / 'workai-literature-verification')
RO, RI, LENGTH = 3., 2., 60.
BASE = json.loads((REPO / 'coarsegraph/specs/labels-b-end-axial.json').read_text(encoding='utf-8'))['base_scene']
# case: (direction, circumferential order, frequency MHz, cycles)
CASES = {'axial': ([1., 0., 0.], 0, .5, 3.), 'transverse': ([0., 1., 0.], 1, .25, 3.)}
DURATION, STEP = 40., .05
STATIONS = (15., 30., 45.)
ANGLES = (0., 90., 180., 270.)
RADII = (RO - .02, (RO + RI) / 2)


def receivers():
    """(x, r, θ) and the point; θ from +y towards +z, as in the pipe mesh (y = r cos θ, z = r sin θ)."""
    return [((x, r, th), [x, r * np.cos(np.radians(th)), r * np.sin(np.radians(th))]) for x in STATIONS for r in RADII for th in ANGLES]


LINE_X = np.arange(2., 58.001, .25)   # outer-surface line at θ = 0 for frequency–wavenumber analysis


def line_points():
    return [((x, RO - .02, 0.), [x, RO - .02, 0.]) for x in LINE_X]


def scene(case, h):
    direction, _, f, cycles = CASES[case]
    s = copy.deepcopy(BASE); s['name'] = f'pipe {case}'; s['receiver_lines'] = []
    s['geometry'].update(kind='pipe', length_mm=LENGTH, outer_radius_mm=RO, inner_radius_mm=RI)
    s['source'] = {'face': 'xmin', 'center_mm': [0., 0., 0.], 'radius_mm': 0., 'direction': direction, 'amplitude_mpa': 1., 'frequency_mhz': f,
                   'cycles': cycles, 'delay_us': 0., 'phase_deg': 0., 'signal': {'type': 'tone_burst', 'window': 'hann'}}
    s['boundaries'] = []; s['boundary'] = {'fixed_face': 'none'}
    s['acquisition'] = {'duration_us': DURATION, 'sample_step_us': STEP}; s['numerics'] = {'h_mm': float(h), 'cfl': .3, 'max_elements': 40000}
    s['receivers'] = [{'name': f'x{p[0]:g}r{p[1]:g}t{p[2]:g}', 'point_mm': [float(v) for v in q], 'direction': [1., 0., 0.]} for p, q in receivers()]
    return s


def run(case, h, stop_after_chunks=None):
    """The worker's network path without its display budget: validate_request → make_mesh → graphnet.rollout."""
    from offline.domain import validate_request
    from spectral3d.mesh import make_mesh
    from graphnet.model import load
    from graphnet.rollout import rollout
    job = JOBS / f'P-{case}-h{h:g}'; record = job / 'record'
    if (record / 'recording.json').exists() and json.loads((record / 'recording.json').read_text())['phase'] == 'completed': return job
    job.mkdir(parents=True, exist_ok=True)
    request = validate_request({'scene': scene(case, h), 'degree': 4}); s = request['scene']; mesh = make_mesh(s, 4, s['numerics']['h_mm'])
    network = load(4, long_record=True); started = time.time()
    identity = {'weights_sha256': network['weights_sha256'], 'time_weights_sha256': network['time_weights_sha256'], 'input_sha256': 'verification'}
    print(case, h, 'nodes', len(mesh['points']), flush=True)
    m = rollout(s, mesh, network, record, identity, callback=lambda m: print(case, h, f"{m['completed_time_us']:.2f} μs {m['elapsed_s']:.0f} s", flush=True),
                stop_after_chunks=stop_after_chunks)
    if m['phase'] == 'completed': (job / 'elapsed_s').write_text(f"{m['elapsed_s']:.1f}")
    return job


if __name__ == '__main__':
    job = run(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else None); print('done', job)
