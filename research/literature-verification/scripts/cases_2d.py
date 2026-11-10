"""Lamb waves (plane strain through y-symmetry faces) and absorbing-layer echoes, run through the workbench worker."""
import sys
from harness import scene, run

PLANE = [{'type': 'symmetry', 'face': 'ymin'}, {'type': 'symmetry', 'face': 'ymax'}]
LATERAL = PLANE + [{'type': 'symmetry', 'face': 'zmin'}, {'type': 'symmetry', 'face': 'zmax'}]


def absorber(length): return {'type': 'absorbing', 'face': 'xmax', 'length_mm': float(length), 'ramp': 'cubic', 'attenuation_db': 60.}


def lamb(direction, duration):
    # End-face traction: axial (x) is symmetric through the thickness -> S0 only; transverse (z) is antisymmetric -> A0 only (below the A1/S1 cut-offs).
    return scene(160., 1., 2., 1., duration, boundaries=PLANE + [absorber(30.)],
                 source={'face': 'xmin', 'direction': direction, 'amplitude_mpa': 1., 'frequency_mhz': .3, 'cycles': 5.},
                 receivers=[{'name': 'mid', 'point_mm': [60., .5, 1.], 'direction': [1., 0., 0.]}])


def echo(length, absorbing):
    return scene(length, .8, .8, 1., 45., boundaries=LATERAL + ([absorber(absorbing)] if absorbing else []),
                 source={'face': 'xmin', 'direction': [1., 0., 0.], 'amplitude_mpa': 1., 'frequency_mhz': .5, 'cycles': 5.},
                 receivers=[{'name': 'x40', 'point_mm': [40., .4, .4], 'direction': [1., 0., 0.]}])


def shg_s0():
    # S0 at 0.4 MHz in a 2 mm plate: S0(f) -> S0(2f) with phase mismatch dk = k(2f) - 2k(f) (de Lima & Hamilton 2003; Muller et al. 2010)
    return scene(200., 1.25, 2., 1.25, 60., boundaries=PLANE + [absorber(30.)],
                 source={'face': 'xmin', 'direction': [1., 0., 0.], 'amplitude_mpa': 10., 'frequency_mhz': .4, 'cycles': 5.},
                 receivers=[{'name': 'mid', 'point_mm': [80., .625, 1.], 'direction': [1., 0., 0.]}])


CASES = {'N2-S0-0.4': shg_s0, 'L2-S0': lambda: lamb([1., 0., 0.], 50.), 'L2-A0': lambda: lamb([0., 0., 1.], 80.),
         'B1-ref': lambda: echo(280., 0), 'B1-free': lambda: echo(100., 0), 'B1-abs6': lambda: echo(100., 6.4), 'B1-abs13': lambda: echo(100., 13.), 'B1-abs26': lambda: echo(100., 26.), 'B1-abs39': lambda: echo(100., 39.)}

if __name__ == '__main__':
    for name in sys.argv[1:] or CASES:
        job = run(name, CASES[name]()); print(name, 'done', (job / 'elapsed_s').read_text(), flush=True)
