"""S0 scattering from a surface-breaking notch (plane strain): workbench network vs the 2D spectral reference.

Plate 110 × 2 mm, S0 from a uniform axial end traction (0.3 MHz, 3-cycle Hann), absorbing layer over the last 30 mm.
The network runs the 3D plate with symmetry on both y faces (plane strain, notch through the width); the reference is
`spectral2d` at h 0.5 and 0.25 mm (separate code path). A plate without notch gives the incident field for both.
Usage: notch.py network NAME [h] | notch.py reference NAME
"""
import copy, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'literature-verification' / 'scripts'))
from harness import REPO, run  # noqa: E402

REF = Path(__file__).resolve().parents[1] / 'reference'; REF.mkdir(exist_ok=True)
LENGTH, THICK, NOTCH_X, F, CYCLES, DURATION = 110., 2., 60., .3, 3., 45.
XS = (30., 75.)   # reflection side, transmission side
# name: (kind, width mm, depth mm)
CASES = {'plate': ('plate', 1., .8), 'd0.4-w1': ('notch', 1., .4), 'd0.8-w1': ('notch', 1., .8), 'd1.2-w1': ('notch', 1., 1.2),
         'd0.8-w2': ('notch', 2., .8), 'd0.8-w4': ('notch', 4., .8), 'd0.8-w6': ('notch', 6., .8), 'd0.8-w9': ('notch', 9., .8)}
BASE = json.loads((REPO / 'coarsegraph/specs/labels-b-end-axial.json').read_text(encoding='utf-8'))['base_scene']


def points(width):
    """Top and bottom surface at each x (u_x of both separates S0 and A0)."""
    return [[x, width / 2, z] for x in XS for z in (THICK, 0.)]


def scene(name, h, dimension='3d'):
    kind, w, d = CASES[name]; s = copy.deepcopy(BASE); s['name'] = f'notch {name}'; s['receiver_lines'] = []
    width = h if dimension == '3d' else 2.
    s['geometry'].update(kind=kind, length_mm=LENGTH, width_mm=width, thickness_mm=THICK, notch_x_mm=NOTCH_X, notch_width_mm=w, notch_depth_mm=d)
    s['source'] = {'face': 'xmin', 'center_mm': [0., width / 2, THICK / 2], 'radius_mm': 0., 'direction': [1., 0., 0.], 'amplitude_mpa': 1., 'frequency_mhz': F,
                   'cycles': CYCLES, 'delay_us': 0., 'phase_deg': 0., 'signal': {'type': 'tone_burst', 'window': 'hann'}}
    s['boundaries'] = [{'type': 'absorbing', 'face': 'xmax', 'length_mm': 30., 'ramp': 'cubic', 'attenuation_db': 60.}]
    if dimension == '3d': s['boundaries'] += [{'type': 'symmetry', 'face': 'ymin'}, {'type': 'symmetry', 'face': 'ymax'}]
    else: s['dimension'] = '2d_plane_strain'
    s['acquisition'] = {'duration_us': DURATION, 'sample_step_us': .05}; s['numerics'] = {'h_mm': float(h), 'cfl': .3, 'max_elements': 40000}
    s['receivers'] = [{'name': f'x{int(p[0])}{"top" if p[2] else "bot"}', 'point_mm': p, 'direction': [1., 0., 0.]} for p in points(width)]
    return s


def reference(name):
    from spectral2d.run import simulate
    for h in (.5, .25):
        path = REF / f'notch-{name}-h{h:g}.npy'
        if path.exists(): continue
        s = scene(name, h, '2d'); s['constitutive'] = 'linear'; started = time.time(); r = simulate(s, degree=4)
        np.save(path, np.asarray(r['computed']['probes_m'])); print('reference', name, h, f'{time.time() - started:.0f} s', flush=True)


if __name__ == '__main__':
    kind, name = sys.argv[1], sys.argv[2]
    if kind == 'network':
        h = float(sys.argv[3]) if len(sys.argv) > 3 else 1.
        job = run(f'N-{name}-h{h:g}', scene(name, h)); print('network', name, h, 'done', (job / 'elapsed_s').read_text(), flush=True)
    else: reference(name)
