"""Software side of the defect COMSOL runs (docs/comsol/defects): the frozen network on the same scenes.

    python research/defects/run_cases.py run D0-intact D1-open-crack D4-closed-crack-1MPa ...   # writes research/defects/out/<case>.npz
    python research/defects/run_cases.py analyse                                                 # writes research/defects/results.json

Each run computes the linear control (no material nonlinearity), so a defect's scattering and a closed defect's
contact nonlinearity are seen on their own. Receiver traces (u_z at the top, u_x at mid-thickness) are stored.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
CASES = REPO / 'docs' / 'comsol' / 'defects' / 'cases'
OUT = HERE / 'out'


def run(names):
    from offline.domain import validate_request
    from graphnet.run import simulate
    OUT.mkdir(exist_ok=True)
    for name in names:
        request = validate_request(json.loads((CASES / f'{name}.json').read_text(encoding='utf-8')))
        started = time.monotonic(); r = simulate(request['scene'], degree=request['degree'], laws='scene'); d = r['computed']['diagnostics']
        np.savez_compressed(OUT / f'{name}.npz', times_us=r['computed']['times_us'], probes_m=r['computed']['probes_m'], names=np.array([x['name'] for x in r['receivers']]))
        info = {'case': name, 'nodes': len(r['mesh']['points']), 'h_mm': request['scene']['numerics']['h_mm'], 'steps': d['internal_steps'], 'seconds': time.monotonic() - started,
                'energy_balance': d['external_work_balance_relative'], 'contact': d.get('contact'), 'mesh': request['mesh_report']}
        (OUT / f'{name}.json').write_text(json.dumps(info, ensure_ascii=False, indent=1, default=float) + '\n', encoding='utf-8')
        print(json.dumps({k: info[k] for k in ('case', 'nodes', 'steps', 'seconds', 'energy_balance')}), flush=True)


def spectrum(trace, dt_us, f_mhz):
    n = 1 << 16; spec = np.fft.rfft(trace * np.hanning(len(trace)), n); freqs = np.fft.rfftfreq(n, dt_us)
    return lambda f: float(np.abs(spec[np.argmin(np.abs(freqs - f))]))


def analyse():
    data = {p.stem: np.load(p) for p in sorted(OUT.glob('*.npz'))}
    ref = data['D0-intact']; names = [str(x) for x in ref['names']]; t = ref['times_us']; dt = t[1] - t[0]; f = .3
    rows = {}
    for case, d in data.items():
        if case == 'D0-intact': continue
        row = {}
        for i, name in enumerate(names):
            comp = 2 if name.startswith('top') else 0
            base, here = ref['probes_m'][:, i, comp], d['probes_m'][:, i, comp]
            s0, s1, sd = spectrum(base, dt, f), spectrum(here, dt, f), spectrum(here - base, dt, f)
            row[name] = {'f_ratio_to_intact': s1(f) / s0(f), 'scattered_f_ratio': sd(f) / s0(f),
                         'harmonic_2f_to_f': s1(2 * f) / s1(f), 'harmonic_3f_to_f': s1(3 * f) / s1(f), 'intact_2f_to_f': s0(2 * f) / s0(f)}
        rows[case] = row
    if 'D4-closed-crack-1MPa' in data and 'D5-closed-crack-2MPa' in data:
        a, b = data['D4-closed-crack-1MPa']['probes_m'], data['D5-closed-crack-2MPa']['probes_m']
        rows['D5/D4 homogeneity'] = {'max_relative_difference_of_D5_minus_2D4': float(np.max(np.abs(b - 2 * a)) / np.max(np.abs(b)))}
    (HERE / 'results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'run': run(sys.argv[2:])
    else: analyse()
