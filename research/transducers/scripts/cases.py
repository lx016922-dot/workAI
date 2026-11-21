"""Cases of the transducer verification, run as the desktop software runs them.

    WORKAI_VERIFY_JOBS=<dir> python research/transducers/scripts/cases.py [name ...]

A 2 mm aluminium plate, 0.5 MHz (fd = 1 MHz·mm: A0, S0 and SH0 propagate), plane strain by symmetry on both y faces,
absorbing layers (40 mm) at both ends. Elements of 1 mm: only the response at f is used, which that resolves with
18 nodes per A0 wavelength. The transducer sits on the top face at x = 120 mm; sixteen receivers on the top face on
each side of it read the waves that leave towards +x and towards −x. Only the complete control is computed; at 1 kPa
it is the linear response.

* W-<v>: a wedge with a 32 mm footprint and trace velocity v. A mode of wavenumber k is driven in proportion to the
  spectrum of the footprint at k, L·sinc((k − k_t)L/2) for a uniform one, whatever the mode's shape — so the amplitude
  of one mode at two trace velocities is in the ratio of two sinc values (analyze.py).
* P-<length>: a bonded wafer (strip) of that length, edge zone 0.5 mm. Its two edge forces drive a mode in proportion
  to sin(k(a − e/2))·sinc(ke/2) with a the half length and e the edge zone.
"""
import importlib.util, sys
from pathlib import Path
# the harness of the damping verification (one job with a compact record), loaded by path: both directories have a cases.py
_spec = importlib.util.spec_from_file_location('damping_cases', Path(__file__).resolve().parents[2] / 'material-damping' / 'scripts' / 'cases.py')
_harness = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_harness)
JOBS, load, run, scene = _harness.JOBS, _harness.load, _harness.run, _harness.scene

F = .5; THICKNESS = 2.; LENGTH = 240.; CENTRE = 120.; FOOTPRINT = 32.; EDGE = .5; H = 1.
RIGHT = [CENTRE + 30. + 2. * i for i in range(16)]; LEFT = [CENTRE - 30. - 2. * i for i in range(16)]


def plate(profile, direction, h=H):
    ends = [{'type': 'absorbing', 'face': face, 'length_mm': 40., 'ramp': 'quadratic', 'attenuation_db': 60.} for face in ('xmin', 'xmax')]
    s = scene(LENGTH, h, THICKNESS, h, 70., boundaries=[{'type': 'symmetry', 'face': 'ymin'}, {'type': 'symmetry', 'face': 'ymax'}] + ends,
              source={'face': 'zmax', 'radius_mm': 0., 'direction': direction, 'amplitude_mpa': 1e-3, 'frequency_mhz': F, 'cycles': 10., 'delay_us': 0., 'profile': profile},
              receivers=[{'name': f'{side}{i}', 'point_mm': [x, h / 2, THICKNESS], 'direction': [0., 0., 1.]} for side, xs in (('R', RIGHT), ('L', LEFT)) for i, x in enumerate(xs)], name='transducer')
    s['source']['center_mm'] = [CENTRE, h / 2, THICKNESS]
    return s


def wedge(velocity, apodization='uniform', h=H):
    return plate({'type': 'wedge', 'along': 'x', 'length_mm': FOOTPRINT, 'width_mm': 0., 'trace_velocity_m_s': float(velocity), 'reverse': False, 'apodization': apodization}, [0., 0., -1.], h)


def patch(length, h=H):
    return plate({'type': 'patch', 'shape': 'strip', 'length_mm': float(length), 'width_mm': 0., 'edge_mm': EDGE}, [1., 0., 0.], h)


def modes():
    """Wavenumbers (rad/m) and velocities of the modes of the plate at F, from the software's own plate theory (offline/guided.py, checked against the Rayleigh–Lamb equations)."""
    from offline.guided import describe
    d = describe(plate({'type': 'uniform'}, [0., 0., -1.]))
    return {m['name']: {'k': m['wavenumber_rad_mm'] * 1e3, 'cp': m['phase_velocity_m_s'], 'cg': m['group_velocity_m_s']} for m in d['modes']['f']}


def wedge_velocities():
    """Trace velocities that put S0, then A0, on the peak of the footprint's spectrum, on its first zero and half-way down its main lobe."""
    import math
    m = modes(); w = 2 * math.pi * F * 1e6; out = {}
    for name in ('S0', 'A0'):
        k = m[name]['k']
        for label, shift in (('peak', 0.), ('half', math.pi / (FOOTPRINT * 1e-3)), ('zero', 2 * math.pi / (FOOTPRINT * 1e-3))):
            out[f'{name}-{label}'] = w / (k - shift)
    return out


def patch_lengths():
    """Wafer lengths that put S0 on a zero and on a maximum of sin(k(a − e/2)), and the same for A0."""
    import math
    m = modes(); out = {}
    for name in ('S0', 'A0'):
        k = m[name]['k']
        out[f'{name}-zero'] = 2 * (math.pi / k * 1e3 + EDGE / 2); out[f'{name}-peak'] = 2 * (1.5 * math.pi / k * 1e3 + EDGE / 2)
    return out


def cases():
    out = {f'W-{label}': (lambda v=v: wedge(v)) for label, v in wedge_velocities().items()}
    out['W-S0-peak-hann'] = lambda: wedge(wedge_velocities()['S0-peak'], 'hann'); out['W-S0-peak-fine'] = lambda: wedge(wedge_velocities()['S0-peak'], h=.7)
    out['P-S0-peak-fine'] = lambda: patch(patch_lengths()['S0-peak'], h=.7)
    out.update({f'P-{label}': (lambda length=length: patch(length)) for label, length in patch_lengths().items()})
    return out


if __name__ == '__main__':
    table = cases()
    for name in sys.argv[1:] or table:
        job = run(name, table[name](), controls='full'); print(name, 'done', (job / 'elapsed_s').read_text(), flush=True)
