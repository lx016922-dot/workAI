"""1 ms references for the bar cases: three controls × two resolutions (element 0.5 and 1 mm, degree 8) + the exact linear solution."""
import json, math, sys, time
from pathlib import Path
import numpy as np
from reference1d import solve, exact_linear

OUT = Path(__file__).resolve().parents[1] / 'reference'; OUT.mkdir(exist_ok=True)
RHO, LAM, MU, L_, M_ = 2700., 58.1e9, 26.1e9, -143e9, -297e9   # the aluminium card (docs/aluminum-material.md)
C11 = LAM + 2 * MU; C111 = 2 * L_ + 4 * M_; C = math.sqrt(C11 / RHO); LENGTH = 40e-3; PROBES = [10e-3, 20e-3, 30e-3]

for f, sample in ((.25e6, .1e-6), (.5e6, .05e-6)):
    t = np.arange(int(round(1e-3 / sample)) + 1) * sample; tag = f'f{int(f / 1e3):03d}k'
    np.save(OUT / f'{tag}-exact-linear.npy', exact_linear(np.array(PROBES), t, LENGTH, C, RHO, 1e6, f, 5))
    for element in (.5e-3, 1e-3):
        for law, name in enumerate(('linear', 'stvk', 'murnaghan')):
            started = time.time(); u = solve(law, LENGTH, RHO, C11, C111, 1e6, f, 5, 1e-3, sample, PROBES, element=element, steps_per_sample=int(round(sample / .5e-9)))
            np.save(OUT / f'{tag}-{name}-e{element * 1e3:g}.npy', u); print(tag, name, element, f'{time.time() - started:.0f} s', flush=True)
(OUT / 'meta.json').write_text(json.dumps({'rho': RHO, 'C11': C11, 'C111': C111, 'c': C, 'length_m': LENGTH, 'probes_m': PROBES, 'sigma_pa': 1e6, 'cycles': 5,
                                         'dt_s': .5e-9, 'degree': 8, 'note': 'u[t, probe]; reference written for this check, independent of the software'}))
