"""Plane longitudinal waves: a bar with symmetry (zero normal displacement) on all four lateral faces is a 1D uniaxial-strain medium."""
import copy, sys
from harness import scene, run
from physics.materials import presets

LATERAL = [{'type': 'symmetry', 'face': f} for f in ('ymin', 'ymax', 'zmin', 'zmax')]
X = [20., 40., 60., 80., 100., 120.]


def bar(amplitude=10., material=None, f=.5, length=200., duration=40., xs=X):
    return scene(length, .8, .8, .8, duration, boundaries=LATERAL, material=material,
                 source={'face': 'xmin', 'direction': [1., 0., 0.], 'amplitude_mpa': amplitude, 'frequency_mhz': f, 'cycles': 5.},
                 receivers=[{'name': f'x{int(x)}', 'point_mm': [x, .4, .4], 'direction': [1., 0., 0.]} for x in xs])


def cfrp(angle, a1=0.):
    m = copy.deepcopy(presets()['cfrp_ud']['material']); m['fiber_angle_deg'] = float(angle); m['parameters']['a1'] = float(a1); return m


CASES = {
    'L1-al-10MPa': lambda: bar(10.),
    'L1-al-5MPa': lambda: bar(5.),
    'L1-cfrp-0deg-a1': lambda: bar(10., cfrp(0., -200.), f=.4, length=150., duration=28., xs=[20., 40., 60., 80.]),
    'L1-cfrp-90deg': lambda: bar(2., cfrp(90.), f=.4, length=150., duration=60., xs=[20., 40., 60., 80.]),
    # 10 MPa across the fibres: incident strain 8.5e-4, doubled near the free end -> the strain guard must stop the job with a readable message
    'G1-cfrp-90deg-10MPa': lambda: bar(10., cfrp(90.), f=.4, length=40., duration=30., xs=[20.]),
}

if __name__ == '__main__':
    for name in sys.argv[1:] or CASES:
        job = run(name, CASES[name]()); print(name, 'done', (job / 'elapsed_s').read_text(), flush=True)
