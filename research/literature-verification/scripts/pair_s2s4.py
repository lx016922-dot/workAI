"""A second synchronous pair: S2(f) → S4(2f) at fd = 2 cT/√(1 − cT²/cL²) = 7.117 MHz·mm, with the analysis of the first pair.

    WORKAI_VERIFY_JOBS=<dir> python pair_s2s4.py run        # about an hour on 4 cores
    WORKAI_VERIFY_JOBS=<dir> python pair_s2s4.py analyse    # → results/results-s2s4.json

The family of points where a symmetric mode has the phase velocity cL (sin(qh) = 0) gives one synchronous pair for
every order n: S_n at f and S_2n at 2f, both with the group velocity of the family. Section 9 of the README checks
n = 1. This is n = 2: the same 8 mm plate at twice the frequency (0.8896 MHz), so twice the modes propagate — nine
symmetric and antisymmetric Lamb modes at f, eighteen at 2f — and the pair has to be found among them.

The model is that of N3 (cases_s1s2.py) with every length along the plate halved: the upper half of the plate in
plane strain, a comb of tangential tractions on the top face with the pitch of the S2 wavelength (7.18 mm), ten teeth,
a 20-cycle burst, 32 receivers on the top face. Six elements over the half thickness keep the resolution per
wavelength of N3-e3. The record is compact (the receivers at every sample).

The analysis is `analyze_s1s2.analyse` with the theory of order 2 (`theory_shg_lamb.pair(order=2)`): phase and group
velocity of the primary mode, and the growth rate κ₂ of the second harmonic at the receivers for the three
increments, against the single frequency and against the theory summed over the spectrum of the burst.
"""
import importlib.util, json, math, sys
from functools import partial
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location('damping_cases', HERE.parents[1] / 'material-damping' / 'scripts' / 'cases.py'); _harness = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_harness)
JOBS, run, scene, load = _harness.JOBS, _harness.run, _harness.scene, _harness.load
import cases_s1s2 as first

ORDER = 2; HALF = first.HALF; NAME = 'N5-s2s4'
LENGTH, ABSORB, COMB_CENTRE = 70., 6., 15.5                       # half thicknesses: half of N3's lengths
RECEIVERS = [27.5 + 1.1 * i for i in range(32)]


def frequency_mhz(half): return ORDER * first.frequency_mhz(half)


def plate(elements=6, amplitude=2., half=HALF):
    h = half / elements; width = h; f = frequency_mhz(half); pitch = first.CL / f * 1e-3
    boundaries = [{'type': 'symmetry', 'face': 'ymin'}, {'type': 'symmetry', 'face': 'ymax'}, {'type': 'symmetry', 'face': 'zmin'},
                  {'type': 'absorbing', 'face': 'xmin', 'length_mm': ABSORB * half, 'ramp': 'cubic', 'attenuation_db': 60.}, {'type': 'absorbing', 'face': 'xmax', 'length_mm': ABSORB * half, 'ramp': 'cubic', 'attenuation_db': 60.}]
    source = {'face': 'zmax', 'direction': [1., 0., 0.], 'amplitude_mpa': amplitude, 'frequency_mhz': f, 'cycles': first.CYCLES, 'signal': {'type': 'tone_burst', 'window': 'hann'},
              'profile': {'type': 'comb', 'pitch_mm': pitch, 'teeth': first.TEETH, 'tooth_width_mm': pitch / 2, 'along': 'x', 'alternating_sign': False}}
    receivers = [{'name': f'x{x:.1f}', 'point_mm': [x * half, width / 2, half], 'direction': [1., 0., 0.]} for x in RECEIVERS]
    s = scene(LENGTH * half, width, half, h, 26. * half, .01 * half, boundaries=boundaries, source=source, receivers=receivers, name='S2–S4 同步模态对')
    s['source']['center_mm'] = [COMB_CENTRE * half, width / 2, half]
    return s


def analyse():
    import analyze_s1s2 as A
    from theory_shg_lamb import burst_table, pair
    # the analysis of the first pair, with the theory, the frequency and the case of this one
    A.pair = partial(pair, order=ORDER); A.burst_table = partial(burst_table, order=ORDER); A.frequency_mhz = frequency_mhz; A.CASES = {NAME: plate}; A.JOBS = JOBS / 'order-2'
    t, values, manifest = load(NAME); result, _ = A.analyse(NAME, HALF, data=(t, values['linear'], values['stvk'], values['murnaghan']))
    theory = pair(2 * HALF * 1e-3, **A.MATERIAL, order=ORDER)
    result.update(order=ORDER, nodes=manifest['nodes'], elapsed_s=float((JOBS / NAME / 'elapsed_s').read_text()), layer_step_us=manifest['layer_step_us'],
                  theory={'f_mhz': theory['f_hz'] * 1e-6, 'group_1': theory['group_1'], 'group_2': theory['group_2'], 'surface_uz_over_ux_1': theory['surface_uz_over_ux_1'], 'surface_uz_over_ux_2': theory['surface_uz_over_ux_2'],
                          'growth_over_k_squared': {part: abs(v) / theory['k_rad_m'] ** 2 for part, v in theory['growth'].items()}})
    path = HERE.parent / 'results' / 'results-s2s4.json'; path.write_text(json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')
    print('f', round(result['f_mhz'], 5), 'MHz, fd', round(result['fd_mhz_mm'], 4), '| phase velocity', f"{result['phase_velocity']['difference']:+.3e}", '| group velocity vs burst theory', f"{result['group_velocity']['difference_from_burst']:+.3%}",
          'vs single frequency', f"{result['group_velocity']['difference']:+.3%}")
    for part, p in result['parts'].items():
        print(f"  {part:9s} κ₂/k² software {p['slope_over_k_squared']:.4f}  theory {p['theory_over_k_squared']:.4f}  vs single frequency {p['magnitude_difference']:+.2%} ({p['phase_difference_deg']:+.1f}°)"
              f"  vs burst theory {p['magnitude_difference_from_burst']:+.2%} ({p['phase_difference_from_burst_deg']:+.1f}°)  fit residual {p['fit_residual']:.3f}")


if __name__ == '__main__':
    if sys.argv[1:] == ['run']: job = run(NAME, plate()); print(NAME, 'done', (job / 'elapsed_s').read_text(), flush=True)
    elif sys.argv[1:] == ['analyse']: analyse()
    else: raise SystemExit(__doc__)
