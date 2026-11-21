"""A second synchronous pair, of another kind: an antisymmetric primary with a symmetric second harmonic.

    WORKAI_VERIFY_JOBS=<dir> python pair_a_s.py run        # about an hour on 4 cores
    WORKAI_VERIFY_JOBS=<dir> python pair_a_s.py analyse    # → results/results-pair-a-s.json

Section 9 checks S1(f) → S2(2f), a symmetric primary on the cp = cL family (every pair of that family is the same field
stacked through the thickness, so S2 → S4 would test nothing new). The second harmonic of an antisymmetric mode is
symmetric too, and such pairs are synchronous where two branches meet the condition by themselves. `theory_pair.py`
finds one in the software's aluminium at fd = 5.033 MHz·mm: phase velocity 8273 m/s, both group velocities 1856 m/s,
κ₂/k² = 0.367 (geometric +0.107, material −0.474). The coupling runs through different terms than for a symmetric
primary — an antisymmetric field has its shear and its bending where S1 has compression — and the modes are found
among nine propagating at f and nineteen at 2f.

The model: the whole 8 mm thickness (no mid-plane symmetry: both parities must exist), plane strain, 0.6291 MHz. Two
combs of tangential tractions with the pitch of the primary wavelength (13.15 mm), eight teeth, one on the top face and
one on the bottom face in the opposite direction: an antisymmetric drive at the wavenumber of the pair. 32 receivers of
uₓ on the top face. Elements of 1 mm (9.9 nodes per shear wavelength at 2f).

The analysis is `analyze_s1s2.analyse` with the theory of this pair: the phase and group velocity of the primary, and
the growth rate κ₂ of the second harmonic at the receivers for the three increments, against the single frequency and
against the theory summed over the spectrum of the burst.
"""
import importlib.util, json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location('damping_cases', HERE.parents[1] / 'material-damping' / 'scripts' / 'cases.py'); _harness = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_harness)
JOBS, run, scene, load = _harness.JOBS, _harness.run, _harness.scene, _harness.load
from theory_pair import GUESS, MATERIAL, synchronous, table

THICKNESS = 8.; H = 1.; NAME = 'N7-a-s'; TEETH, CYCLES = 8, 20.
LENGTH, ABSORB, COMB_CENTRE = 400., 40., 100.
RECEIVERS = [172. + 5.5 * i for i in range(32)]


def point():
    """The pair: (frequency in MHz, phase velocity in m/s, the theory's result)."""
    r = synchronous(THICKNESS * 1e-3, **MATERIAL, **GUESS); return r['f_hz'] * 1e-6, r['phase_velocity'], r


def plate(amplitude=2.):
    f, cp, _ = point(); pitch = cp / f * 1e-3; width = H
    boundaries = [{'type': 'symmetry', 'face': 'ymin'}, {'type': 'symmetry', 'face': 'ymax'}] + [{'type': 'absorbing', 'face': face, 'length_mm': ABSORB, 'ramp': 'cubic', 'attenuation_db': 60.} for face in ('xmin', 'xmax')]
    comb = {'type': 'comb', 'pitch_mm': pitch, 'teeth': TEETH, 'tooth_width_mm': pitch / 2, 'along': 'x', 'alternating_sign': False}
    drive = lambda face, sign, z: {'face': face, 'center_mm': [COMB_CENTRE, width / 2, z], 'radius_mm': 0., 'direction': [sign, 0., 0.], 'amplitude_mpa': amplitude, 'frequency_mhz': f, 'cycles': CYCLES, 'delay_us': 0., 'phase_deg': 0.,
                                   'signal': {'type': 'tone_burst', 'window': 'hann'}, 'profile': comb}
    receivers = [{'name': f'x{x:g}', 'point_mm': [x, width / 2, THICKNESS], 'direction': [1., 0., 0.]} for x in RECEIVERS]
    top = drive('zmax', 1., THICKNESS); source = {k: v for k, v in top.items() if k != 'center_mm'}
    s = scene(LENGTH, width, THICKNESS, H, 150., .05, boundaries=boundaries, source=source, sources=[dict(drive('zmin', -1., 0.), name='bottom comb')], receivers=receivers, name='A → S 同步模态对')
    s['source']['center_mm'] = top['center_mm']
    return s


def analyse():
    import analyze_s1s2 as A
    f, cp, theory = point()
    # the analysis of the first pair with the frequency, the wavenumber and the theory of this one
    A.pair = lambda thickness, **material: synchronous(thickness, **material, **GUESS); A.burst_table = lambda thickness, span=.16, count=17, **material: table(thickness, **material, **GUESS, span=span, count=count)
    A.frequency_mhz = lambda half: f; A.CL = cp; A.CASES = {NAME: plate}; A.JOBS = JOBS / 'pair-a-s'
    t, values, manifest = load(NAME); result, _ = A.analyse(NAME, THICKNESS / 2, data=(t, values['linear'], values['stvk'], values['murnaghan']))
    result.update(pair='antisymmetric primary → symmetric second harmonic', nodes=manifest['nodes'], elapsed_s=float((JOBS / NAME / 'elapsed_s').read_text()), layer_step_us=manifest['layer_step_us'],
                  theory={'f_mhz': f, 'fd_mhz_mm': f * THICKNESS, 'phase_velocity': cp, 'group_1': theory['group_1'], 'group_2': theory['group_2'], 'surface_uz_over_ux_1': theory['surface_uz_over_ux_1'],
                          'surface_uz_over_ux_2': theory['surface_uz_over_ux_2'], 'growth_over_k_squared': {part: abs(v) / theory['k_rad_m'] ** 2 for part, v in theory['growth'].items()}})
    try:   # the names the software's own plate theory gives the two modes
        from offline.domain import validate_request
        from offline.guided import describe
        d = describe(validate_request({'scene': plate(), 'degree': 4})['scene']); nearest = lambda key: min(d['modes'][key], key=lambda mode: abs(mode['phase_velocity_m_s'] - cp))
        result['software_theory'] = {key: {k: nearest(key)[k] for k in ('name', 'phase_velocity_m_s', 'group_velocity_m_s')} for key in ('f', '2f')}; result['modes_propagating'] = {key: len(d['modes'][key]) for key in ('f', '2f')}
    except Exception as error: result['software_theory'] = {'error': str(error)}
    (HERE.parent / 'results' / 'results-pair-a-s.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')
    print('f', round(result['f_mhz'], 5), 'MHz, fd', round(result['fd_mhz_mm'], 4), '| modes', result.get('software_theory'), '| phase velocity', f"{result['phase_velocity']['difference']:+.3e}",
          '| group velocity vs burst theory', f"{result['group_velocity']['difference_from_burst']:+.3%}", 'vs single frequency', f"{result['group_velocity']['difference']:+.3%}")
    for part, p in result['parts'].items():
        print(f"  {part:9s} κ₂/k² software {p['slope_over_k_squared']:.4f}  theory {p['theory_over_k_squared']:.4f}  vs single frequency {p['magnitude_difference']:+.2%} ({p['phase_difference_deg']:+.1f}°)"
              f"  vs burst theory {p['magnitude_difference_from_burst']:+.2%} ({p['phase_difference_from_burst_deg']:+.1f}°)  fit residual {p['fit_residual']:.3f}")


if __name__ == '__main__':
    if sys.argv[1:] == ['run']: job = run(NAME, plate()); print(NAME, 'done', (job / 'elapsed_s').read_text(), flush=True)
    elif sys.argv[1:] == ['analyse']: analyse()
    else: raise SystemExit(__doc__)
