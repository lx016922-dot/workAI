"""The synchronous Lamb mode pair S1(f) → S2(2f) at fd = cT/√(1 − cT²/cL²) = 3.5585 MHz·mm, run through the workbench worker.

An aluminium plate in plane strain (y faces are symmetry planes), modelled as its upper half: the mid-plane
(z = 0 of the model) is a symmetry plane, so only symmetric modes exist; the top face is free. Both ends absorb.

Both modes have no out-of-plane displacement at the surface at this point (p = 0, sin(qh) = 0), so the plate is
driven and observed in-plane, and the receivers take u_x on the top face. Two excitations:

* N3, a comb of tangential tractions on the top face with the pitch of the S1 wavelength: it selects the wavenumber
  k = ω/cL, so the wave train at the receivers is almost pure S1, as in an experiment with a comb or wedge
  transducer. The source is ten wavelengths long.
* N4, a uniform axial traction on the end face x = 0: every symmetric mode is excited (S0, S1, S2), and they are
  separated afterwards by wavenumber in the complete field. The source is at one position, so the distance a part
  of the wave train has travelled is its x.

The workbench accepts element sizes of 0.8–2.5 mm, and the second harmonic needs about three elements over the
half thickness, so the plate here is 8 mm thick (0.4448 MHz). Elasticity without dissipation has no length scale:
with every length and time multiplied by s and the same traction, displacements multiply by s and strains are
unchanged, so this is the 2 mm plate at 1.779 MHz in other units. N3-2mm is that plate itself (elements of
0.333 mm), run through relaxed_worker.py, which differs from the workbench's worker only in the size rule.

    python cases_s1s2.py N3-e3 N4-e3      # 3 elements over the half thickness (2f resolution 9.6 in the workbench's measure)
    python cases_s1s2.py N3-e4 N4-e4      # 4 elements (12.9)
    python cases_s1s2.py N3-half          # half the amplitude on the first mesh
    python cases_s1s2.py N3-2mm           # the 2 mm plate at 1.779 MHz, outside the workbench's element-size rule
"""
import math, sys
from harness import scene, run

CL, CT = 6391.54534145318, 3109.126351029605
FD = CT / math.sqrt(1 - (CT / CL) ** 2)               # Hz·m: 3558.5
HALF = 4.0                                             # mm: half of the 8 mm plate
TEETH, CYCLES = 10, 20.
# Lengths in units of the half thickness: plate, absorbing layers, comb centre; receivers past the comb and before the absorber
LENGTH, ABSORB, COMB_CENTRE = 140., 12., 31.
RECEIVERS = [55. + 2.2 * i for i in range(32)]


def frequency_mhz(half): return FD / (2 * half * 1e-3) * 1e-6


def pitch_mm(half): return CL / frequency_mhz(half) * 1e-3   # the S1 wavelength at cp = cL: 3.5922 half thicknesses


def s1s2(elements, amplitude=2., half=HALF):
    h = half / elements; width = h; f = frequency_mhz(half); pitch = pitch_mm(half)
    boundaries = [{'type': 'symmetry', 'face': 'ymin'}, {'type': 'symmetry', 'face': 'ymax'}, {'type': 'symmetry', 'face': 'zmin'},
                  {'type': 'absorbing', 'face': 'xmin', 'length_mm': ABSORB * half, 'ramp': 'cubic', 'attenuation_db': 60.},
                  {'type': 'absorbing', 'face': 'xmax', 'length_mm': ABSORB * half, 'ramp': 'cubic', 'attenuation_db': 60.}]
    source = {'face': 'zmax', 'direction': [1., 0., 0.], 'amplitude_mpa': amplitude, 'frequency_mhz': f, 'cycles': CYCLES,
              'signal': {'type': 'tone_burst', 'window': 'hann'},
              'profile': {'type': 'comb', 'pitch_mm': pitch, 'teeth': TEETH, 'tooth_width_mm': pitch / 2, 'along': 'x', 'alternating_sign': False}}
    receivers = [{'name': f'x{x:.1f}', 'point_mm': [x * half, width / 2, half], 'direction': [1., 0., 0.]} for x in RECEIVERS]
    s = scene(LENGTH * half, width, half, h, 52. * half, .02 * half, boundaries=boundaries, source=source, receivers=receivers, name='S1–S2 同步模态对')
    s['source']['center_mm'] = [COMB_CENTRE * half, width / 2, half]   # the comb's centre on the top face
    return s


END_RECEIVERS = [20. + 3.3 * i for i in range(32)]     # N4: 20 … 122.3 half thicknesses from the loaded end


def endface(elements, amplitude=2., half=HALF):
    h = half / elements; width = h
    boundaries = [{'type': 'symmetry', 'face': 'ymin'}, {'type': 'symmetry', 'face': 'ymax'}, {'type': 'symmetry', 'face': 'zmin'},
                  {'type': 'absorbing', 'face': 'xmax', 'length_mm': ABSORB * half, 'ramp': 'cubic', 'attenuation_db': 60.}]
    source = {'face': 'xmin', 'direction': [1., 0., 0.], 'amplitude_mpa': amplitude, 'frequency_mhz': frequency_mhz(half), 'cycles': CYCLES, 'signal': {'type': 'tone_burst', 'window': 'hann'}}
    receivers = [{'name': f'x{x:.1f}', 'point_mm': [x * half, width / 2, half], 'direction': [1., 0., 0.]} for x in END_RECEIVERS]
    # 40 half thicknesses of time: the complete fields kept at 1/2 and 3/4 of the record have the S1 train between the end and the absorbing layer
    return scene(LENGTH * half, width, half, h, 40. * half, .02 * half, boundaries=boundaries, source=source, receivers=receivers, name='S1–S2 同步模态对 · 端面激励')


CASES = {'N3-e3': lambda: s1s2(3), 'N3-e4': lambda: s1s2(4), 'N3-e5': lambda: s1s2(5), 'N3-half': lambda: s1s2(3, amplitude=1.),
         'N4-e3': lambda: endface(3), 'N4-e4': lambda: endface(4), 'N4-half': lambda: endface(3, amplitude=1.),
         'N3-2mm': lambda: s1s2(3, half=1.)}
HALVES = {'N3-2mm': 1.}                                # half thickness (mm) of the cases that are not the 8 mm plate

if __name__ == '__main__':
    for name in sys.argv[1:] or CASES:
        job = run(name, CASES[name](), relaxed=name in HALVES); print(name, 'done', (job / 'elapsed_s').read_text(), flush=True)
