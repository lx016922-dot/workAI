"""Collinear mixing of two plane longitudinal waves, analysed with the software's own mixing analysis (offline/nonlinear.py).

    WORKAI_VERIFY_JOBS=<dir> python research/wave-mixing/mixing_bar.py run
    WORKAI_VERIFY_JOBS=<dir> python research/wave-mixing/mixing_bar.py analyse     # writes results.json

A bar with symmetry on its four lateral faces carries plane longitudinal waves (uniaxial strain), for which the second-order
solution is closed: with u₁ = A_a cos θ_a + A_b cos θ_b and ρ u_tt = C₁₁ u_xx (1 − β u_x), β = −(3 + C₁₁₁/C₁₁),

    u₂ = β x (∂ₜu₁)² / (4c²)   ⇒   A₊ = A₋ = β k_a k_b A_a A_b x / 4,   A(2f_a) = β k_a² A_a² x / 8.

So A±/(A_a A_b) grows along the bar with the slope β k_a k_b / 4, and the sum and difference components are phase matched
(no dispersion). The three controls give the geometric (β = −3), material (β = −C₁₁₁/C₁₁) and complete values.

The source is one end-face traction with a two-tone signal, 0.4 and 0.5 MHz, 40 μs long.
"""
import importlib.util, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('damping_cases', HERE.parent / 'material-damping' / 'scripts' / 'cases.py'); _harness = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_harness)
JOBS, run, scene = _harness.JOBS, _harness.run, _harness.scene

FA, FB = .4, .5; X = [20. + 10. * i for i in range(11)]; NAME = 'M-bar'
LATERAL = [{'type': 'symmetry', 'face': f} for f in ('ymin', 'ymax', 'zmin', 'zmax')]


def bar(amplitude=10.):
    signal = {'type': 'multi_tone', 'duration_us': 40., 'window': 'hann', 'components': [{'frequency_mhz': FA, 'weight': 1., 'phase_deg': 0.}, {'frequency_mhz': FB, 'weight': 1., 'phase_deg': 0.}]}
    return scene(300., .8, .8, .8, 72., boundaries=LATERAL, source={'face': 'xmin', 'direction': [1., 0., 0.], 'amplitude_mpa': amplitude, 'frequency_mhz': FB, 'cycles': 1., 'signal': signal},
                 receivers=[{'name': f'x{int(x)}', 'point_mm': [x, .4, .4], 'direction': [1., 0., 0.]} for x in X], name='two-tone bar')


def analyse():
    from offline.nonlinear import mixing
    request = json.loads((JOBS / NAME / 'request.json').read_text()); m = request['scene']['material']; c11 = m['lambda_pa'] + 2 * m['mu_pa']; c = math.sqrt(c11 / m['rho_kg_m3'])
    beta = -(3 + 2 * (m['l_pa'] + 2 * m['m_pa']) / c11); ka, kb = (2 * math.pi * f * 1e6 / c for f in (FA, FB)); theory = abs(beta) * ka * kb / 4
    out = mixing(JOBS / NAME, {'points': {'mode': 'receivers'}, 'direction': [1., 0., 0.]}); fit = out['fit']
    rows = [r for r in out['rows'] if r['usable']]; x = np.array([r['distance_mm'] for r in rows]) * 1e-3 + X[0] * 1e-3
    # the second harmonics of the two tones, from the same rows: A(2f)/A² against the distance has the slope β k²/8
    second = {key: float(np.polyfit(x, [r[f'2{key}_m'] / r[f'{key}_m'] ** 2 for r in rows], 1)[0]) for key in ('a', 'b')}
    result = {'frequencies_mhz': out['frequencies_mhz'], 'window': out['window'], 'mixed_basis': out['mixed_basis'], 'beta': beta, 'speed_m_s': c, 'points': len(rows), 'notes': out['notes'],
              'theory_slope_per_m2': theory, 'sum': dict(fit['sum'], against_theory=fit['sum']['slope_per_m2'] / theory - 1), 'difference': dict(fit['difference'], against_theory=fit['difference']['slope_per_m2'] / theory - 1),
              'second_harmonic': {key: {'slope_per_m2': second[key], 'theory_per_m2': abs(beta) * k * k / 8, 'against_theory': second[key] / (abs(beta) * k * k / 8) - 1} for key, k in (('a', ka), ('b', kb))},
              'wavenumbers_rad_mm': {key: fit['waves'][key]['wavenumber_rad_mm'] for key in fit['waves']}, 'theory_wavenumbers_rad_mm': {'a': ka * 1e-3, 'b': kb * 1e-3, 'sum': (ka + kb) * 1e-3, 'difference': (kb - ka) * 1e-3},
              'elapsed_s': float((JOBS / NAME / 'elapsed_s').read_text())}
    (HERE / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')
    print('theory slope', f'{theory:.5e}', '| sum', f"{fit['sum']['slope_per_m2']:.5e} ({result['sum']['against_theory']:+.3%}, R² {fit['sum']['r_squared']:.5f})",
          '| difference', f"{fit['difference']['slope_per_m2']:.5e} ({result['difference']['against_theory']:+.3%}, R² {fit['difference']['r_squared']:.5f})",
          '| 2f_a', f"{result['second_harmonic']['a']['against_theory']:+.3%}", '2f_b', f"{result['second_harmonic']['b']['against_theory']:+.3%}", '| k', result['wavenumbers_rad_mm'], result['theory_wavenumbers_rad_mm'], out['notes'])


if __name__ == '__main__':
    if sys.argv[1:] == ['run']: job = run(NAME, bar()); print(NAME, 'done', (job / 'elapsed_s').read_text(), flush=True)
    elif sys.argv[1:] == ['analyse']: analyse()
    else: raise SystemExit(__doc__)
