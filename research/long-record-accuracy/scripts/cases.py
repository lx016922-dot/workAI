"""1 ms records of a free–free bar (plane longitudinal waves, lateral symmetry faces) through the workbench path.

The bar has an exact linear solution (multiple reflections, method of images) and a cheap independent 1D
reference for the three constitutive controls, so the error of a 1 ms record can be measured, not estimated.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'literature-verification' / 'scripts'))
from harness import scene, run  # noqa: E402  (same request -> worker -> record path as the workbench)

LENGTH, X = 40., [10., 20., 30.]
LATERAL = [{'type': 'symmetry', 'face': f} for f in ('ymin', 'ymax', 'zmin', 'zmax')]


def bar(h, f, step, amplitude=1., duration=1000., degree_note=4):
    return scene(LENGTH, h, h, h, duration, step=step, boundaries=LATERAL,
                 source={'face': 'xmin', 'direction': [1., 0., 0.], 'amplitude_mpa': amplitude, 'frequency_mhz': f, 'cycles': 5.},
                 receivers=[{'name': f'x{int(x)}', 'point_mm': [x, h / 2, h / 2], 'direction': [1., 0., 0.]} for x in X])


# name: (h_mm, f_MHz, sample step μs); nodes per 2f longitudinal wavelength at degree 4 in brackets
CASES = {'T1-f025-h2.5': (2.5, .25, .1),   # [20]
         'T2-f025-h1.5': (1.5, .25, .1),   # [34]
         'T3-f050-h1.5': (1.5, .5, .05),   # [17]
         'T4-f050-h0.8': (.8, .5, .05)}    # [32]

if __name__ == '__main__':
    for name in sys.argv[1:] or CASES:
        h, f, step = CASES[name]; job = run(name, bar(h, f, step)); print(name, 'done', (job / 'elapsed_s').read_text(), flush=True)
