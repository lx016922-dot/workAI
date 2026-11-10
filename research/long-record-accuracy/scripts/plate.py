"""1 ms plane-strain plate (48 × 2 mm, free ends, 0.2 MHz): workbench network vs a converged 2D spectral reference.

The load is a uniform end-face traction along (1, 0, 1): its x part excites S0, its z part A0, and it is uniform across the
width, so the 3D run is exactly plane strain (a round Gaussian patch would vary across the 3D width).

The network runs the 3D plate with symmetry on both y faces (plane strain; the project's tests show this equals the 2D
solver to 1e-10). The reference is `spectral2d` (separate code path: velocity Verlet, no network) at h 0.75 and 0.5 mm, degree 4,
CFL 0.35; their difference bounds the reference's own error. Usage: plate.py network h | plate.py reference h
"""
import copy, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'literature-verification' / 'scripts'))
from harness import REPO, run  # noqa: E402

REF = Path(__file__).resolve().parents[1] / 'reference'; REF.mkdir(exist_ok=True)
TOP = [24., 32., 40.]; MID = [24., 32.]
BASE = json.loads((REPO / 'coarsegraph/specs/labels-a-top-gaussian.json').read_text(encoding='utf-8'))['base_scene']


def points(width):
    return [[x, width / 2, 2.] for x in TOP] + [[x, width / 2, 1.] for x in MID]


def plate(h, dimension='3d'):
    s = copy.deepcopy(BASE); s['name'] = '1 ms plate'; s['receiver_lines'] = []
    s['source'] = {'face': 'xmin', 'center_mm': [0., 1., 1.], 'radius_mm': 0., 'direction': [1., 0., 1.], 'amplitude_mpa': 1., 'frequency_mhz': .2,
                   'cycles': 5., 'delay_us': 0., 'phase_deg': 0., 'signal': {'type': 'tone_burst', 'window': 'hann'}}
    s['acquisition'] = {'duration_us': 1000., 'sample_step_us': .1}; s['numerics'] = {'h_mm': float(h), 'cfl': .3, 'max_elements': 40000}
    if dimension == '3d':
        s['geometry']['width_mm'] = float(h); s['source']['center_mm'] = [0., h / 2, 1.]
        s['boundaries'] = [{'type': 'symmetry', 'face': 'ymin'}, {'type': 'symmetry', 'face': 'ymax'}]
    else:
        # the 2D solver's scene core allows ≤ 2000 sample intervals: sample every 0.5 μs (10 per carrier period; the internal step is unchanged)
        s['dimension'] = '2d_plane_strain'; s['boundaries'] = []; s['numerics']['cfl'] = .35; s['acquisition']['sample_step_us'] = .5
    width = s['geometry']['width_mm']
    s['receivers'] = [{'name': f'top{int(x)}', 'point_mm': p, 'direction': [0., 0., 1.]} for x, p in zip(TOP, points(width)[:3])] + \
                     [{'name': f'mid{int(x)}', 'point_mm': p, 'direction': [1., 0., 0.]} for x, p in zip(MID, points(width)[3:])]
    return s


def reference(h):
    from spectral2d.run import simulate
    for law in ('linear', 'stvk', 'murnaghan'):
        path = REF / f'plate-{law}-h{h:g}.npy'
        if path.exists(): continue
        s = plate(h, '2d'); s['constitutive'] = law; started = time.time(); r = simulate(s, degree=4)
        np.save(path, np.asarray(r['computed']['probes_m'])); print('reference', law, h, f'{time.time() - started:.0f} s', flush=True)


if __name__ == '__main__':
    kind, h = sys.argv[1], float(sys.argv[2])
    if kind == 'network': job = run(f'P-h{h:g}', plate(h)); print('network', h, 'done', (job / 'elapsed_s').read_text(), flush=True)
    else: reference(h)
