"""Compact pass/fail summary of every check (reads results/*.json; writes results/summary.json)."""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / 'results'


def load(name):
    p = RESULTS / name; return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


def main():
    m, r1, r2, s = load('materials.json'), load('results-1d.json'), load('results-2d.json'), load('results-shg.json'); out = {}
    if m:
        out['M1_murnaghan_toe_max_rel'] = max(m['M1_murnaghan_brugger']['max_rel_diff_exact'], m['M1_murnaghan_brugger']['max_rel_diff_custom_W'])
        out['M2_ti_engineering_max_rel'] = max(r['max_rel_diff'] for r in m['M2_TI_engineering'])
        out['M3_christoffel_max_rel'] = max(r['max_rel_diff'] for r in m['M3_TI_christoffel'])
    for name, r in r1.items():
        row = {'c_rel_diff': r['c_rel_diff'], 'particle_velocity_rel_L2': r['particle_velocity_rel_L2_vs_sigma_over_rho_c'], 'beta_theory': r['beta_theory']}
        for part in ('geometric', 'material', 'total'):
            fits = [p[part]['beta_fit'] for p in r['pairs']]; row[f'beta_fit_{part}'] = [min(fits), max(fits)]
            row[f'max_waveform_residual_{part}'] = max(p[part]['waveform_residual'] for p in r['pairs'])
        ratios = [g['A2_m'] / g['A2_textbook_m'] for g in r['growth']]; row['A2_over_textbook'] = [min(ratios), max(ratios)]
        out[name] = row
    for name in ('L2-S0', 'L2-A0'):
        if name in r2: out[name] = {k: r2[name][k] for k in ('band_mhz', 'max_abs_cp_rel_diff', 'max_abs_cg_rel_diff_inner')}
    if 'echo' in r2: out['B1_echo_db'] = {k: {'layer_mm': v['layer_mm'], 'layer_wavelengths': v['layer_mm'] / v['wavelength_mm'], 'reflection_db': v['reflection_db']} for k, v in r2['echo'].items() if isinstance(v, dict)}
    if s: out['N2_s0_shg'] = {'dk_theory_rad_m': s['dk_theory_rad_m'], **{f'dk_fit_{p}': v['dk_fit_rad_m'] for p, v in s['parts'].items()}, 'fit_residual_total': s['parts']['total']['fit_residual']}
    (RESULTS / 'summary.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__': main()
