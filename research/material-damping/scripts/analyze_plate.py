"""S0 in a damped plate, analysed with the software's own accumulation analysis; writes results/plate.json.

    WORKAI_VERIFY_JOBS=<dir> python research/material-damping/scripts/analyze_plate.py

* The attenuation of S0 at f: A₁ of the damped plate over A₁ of the undamped one, receiver by receiver, against
  exp(−γ(f) x / c_g) with the group velocity of S0 from the plate theory (offline/guided.py).
* The accumulation analysis of the damped plate reports A₂/A₁² with the attenuation taken out; its slope is set
  against the slope of the undamped plate, which is what it should give back.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from cases import JOBS

RESULTS = HERE.parent / 'results'


def main():
    from offline.nonlinear import accumulation
    body = {'points': {'mode': 'receivers'}, 'direction': [1., 0., 0.]}; plain = accumulation(JOBS / 'P-none', body); damped = accumulation(JOBS / 'P-both', body)
    rows = lambda a: [r for r in a['rows'] if r['usable']]; p, d = rows(plain), rows(damped); n = min(len(p), len(d)); x = np.array([r['distance_mm'] for r in p[:n]]) * 1e-3
    t = damped['fit']['attenuation']; ratio = np.array([r['a1_m'] for r in d[:n]]) / np.array([r['a1_m'] for r in p[:n]]); measured = -float(np.polyfit(x, np.log(ratio), 1)[0])
    out = {'frequency_mhz': damped['frequency_mhz'], 'damping': json.loads((JOBS / 'P-both' / 'request.json').read_text())['scene']['damping'], 'mode': t.get('mode_f'), 'mode_2f': t.get('mode_2f'), 'points': n,
           'attenuation_np_m': {'theory_f': t['a1_np_m'], 'theory_2f': t['a2_np_m'], 'a1_ratio_fit': measured, 'difference': measured / t['a1_np_m'] - 1, 'a1_of_damped_plate_alone': t['a1_measured_np_m']},
           'group_velocity_m_s': {'measured_undamped': plain['group']['velocity_m_s'], 'measured_damped': damped['group']['velocity_m_s'], 'theory': plain['fit']['f']['theory']['group_velocity_m_s'] if plain['fit']['f']['theory'] else None},
           'ratio_slope_per_m2': {'undamped': plain['fit']['ratio']['slope_per_m2'], 'damped_uncorrected': damped['fit']['ratio']['slope_per_m2'], 'damped_corrected': t['corrected']['slope_per_m2'],
                                  'corrected_against_undamped': t['corrected']['slope_per_m2'] / plain['fit']['ratio']['slope_per_m2'] - 1, 'uncorrected_against_undamped': damped['fit']['ratio']['slope_per_m2'] / plain['fit']['ratio']['slope_per_m2'] - 1,
                                  'r_squared': {'undamped': plain['fit']['ratio']['r_squared'], 'damped_uncorrected': damped['fit']['ratio']['r_squared'], 'damped_corrected': t['corrected']['r_squared']}},
           'wavenumber_f_against_theory': {'undamped': plain['fit']['f']['theory']['difference'] if plain['fit']['f']['theory'] else None, 'damped': damped['fit']['f']['theory']['difference'] if damped['fit']['f']['theory'] else None},
           'notes': {'undamped': plain['notes'], 'damped': damped['notes']}, 'elapsed_s': {name: float((JOBS / name / 'elapsed_s').read_text()) for name in ('P-none', 'P-both')}}
    # the same with the group velocities of the plate theory (the receivers are too far apart for the analysis to recognise the mode by its wavenumber)
    from offline.guided import describe
    from physics.damping import decay_per_s
    scene = json.loads((JOBS / 'P-both' / 'request.json').read_text())['scene']; modes = describe(scene)['modes']; s0 = {key: next(m for m in modes[key] if m['name'] == 'S0') for key in ('f', '2f')}
    f = scene['source']['frequency_mhz'] * 1e6; exact = decay_per_s(scene['damping'], f) / s0['f']['group_velocity_m_s']
    out['plate_theory'] = {'group_velocity_f_m_s': s0['f']['group_velocity_m_s'], 'group_velocity_2f_m_s': s0['2f']['group_velocity_m_s'], 'attenuation_f_np_m': exact,
                           'attenuation_2f_np_m': decay_per_s(scene['damping'], 2 * f) / s0['2f']['group_velocity_m_s'], 'a1_ratio_fit_against_it': measured / exact - 1,
                           'measured_group_velocity_against_it': plain['group']['velocity_m_s'] / s0['f']['group_velocity_m_s'] - 1}
    out['rows'] = {'source_offset_mm': t['source_offset_mm'], 'undamped_fit': plain['fit']['ratio'],
                   'undamped': [{'distance_mm': r['distance_mm'], 'ratio_per_m': r['ratio_per_m'], 'a1_m': r['a1_m']} for r in p],
                   'damped': [{'distance_mm': r['distance_mm'], 'ratio_per_m': r['ratio_per_m'], 'ratio_corrected_per_m': r.get('ratio_corrected_per_m'), 'a1_m': r['a1_m']} for r in d]}
    RESULTS.mkdir(exist_ok=True); (RESULTS / 'plate.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('attenuation_np_m', 'plate_theory', 'ratio_slope_per_m2')}, ensure_ascii=False, indent=1))


if __name__ == '__main__': main()
