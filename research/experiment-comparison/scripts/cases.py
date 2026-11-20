"""The experiment of Ding, Zhao, Hu, Liu, Zhang and Deng, "Experimental and Numerical Study of Nonlinear Lamb Waves of a
Low-Frequency S0 Mode in Plates with Quadratic Nonlinearity", Materials 11 (2018) 2096, doi:10.3390/ma11112096 (CC BY 4.0),
as models of the software.

The paper sends a 10-cycle Hann tone burst of the S0 mode along an AL-6061-T6 sheet (wedge transducer), receives the
out-of-plane surface motion at a series of distances and reports, for four combinations of frequency and thickness,
the distance at which the acoustic nonlinearity parameter A2/A1² is largest (its Table 2, read with its Figures 7
and 8), from the experiment and from its own plane-strain finite-element simulation (ABAQUS/Explicit).

A model here is the plane-strain strip of that simulation: 1.6 mm wide (one element) between two symmetry faces, the burst as
an in-plane traction on the end face (which excites S0 only), an absorbing far end, the material constants of the
paper's Table 1 as a Murnaghan material of the software. Every model runs exactly as the workbench runs it:
automatic mesh and time layer, `offline.worker`, a complete record.
"""
import copy, hashlib, json, os, subprocess, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent; RESULTS = HERE.parent / 'results'; FIGURES = HERE.parent / 'figures'
# Records are 1–3 GB each: keep them outside the repository.
JOBS = Path(os.environ.get('WORKAI_EXPERIMENT_JOBS') or Path(tempfile.gettempdir()) / 'workai-experiment-comparison')

PAPER = {'citation': 'X. Ding, Y. Zhao, N. Hu, Y. Liu, J. Zhang, M. Deng, Materials 11 (2018) 2096', 'doi': '10.3390/ma11112096', 'license': 'CC BY 4.0'}
# Table 1 of the paper (AL-6061-T6): density, Lamé constants and Murnaghan third-order constants.
MATERIAL = {'model': 'murnaghan', 'name': 'AL-6061-T6 (Ding 2018)', 'rho_kg_m3': 2704., 'lambda_pa': 5.11e10, 'mu_pa': 2.63e10, 'l_pa': -2.82e11, 'm_pa': -3.39e11, 'n_pa': -4.16e11}
CYCLES = 10.
# Table 2 of the paper: distance of the largest A2/A1² (mm) from its simulation and its experiment; `measured_mm` is the
# range of distances with experimental points in its Figure 8, `simulated_mm` the range of its simulation in Figure 7
# (case d: to 950 mm there, 940 mm here because a model of the software is at most 1000 mm long), `note` what Figure 8 shows.
# The experimental points of Figure 8 are 25 mm apart; the positions of the smallest and largest point of case b are read
# from that figure.
CASES = {
    'a-300kHz-2.0mm': {'frequency_mhz': .30, 'thickness_mm': 2.0, 'paper_simulation_mm': 180., 'experiment_mm': 175., 'measured_mm': [100., 225.], 'simulated_mm': [20., 380.], 'reach_mm': 420.,
                       'note': '实验点在 175 mm 处最大，两侧各有下降的点'},
    'b-300kHz-2.5mm': {'frequency_mhz': .30, 'thickness_mm': 2.5, 'paper_simulation_mm': 100., 'experiment_mm': 125., 'experiment_minimum_mm': 175., 'experiment_maximum_mm': 300., 'measured_mm': [150., 350.], 'simulated_mm': [5., 205.], 'reach_mm': 420.,
                       'note': '实验从 150 mm 起测：175 mm 处最小、300 mm 处最大，表中的 125 mm 是这两处的间距（半个拍长），第一个极大在测量范围之外'},
    'c-240kHz-2.5mm': {'frequency_mhz': .24, 'thickness_mm': 2.5, 'paper_simulation_mm': 240., 'experiment_mm': 250., 'measured_mm': [100., 325.], 'simulated_mm': [20., 380.], 'reach_mm': 420.,
                       'note': '实验点在 250–275 mm 处最大，两侧各有下降的点'},
    'd-200kHz-2.0mm': {'frequency_mhz': .20, 'thickness_mm': 2.0, 'paper_simulation_mm': 800., 'experiment_mm': 700., 'measured_mm': [100., 700.], 'simulated_mm': [50., 940.], 'reach_mm': 940.,
                       'note': '实验曲线到最远的测点 700 mm 仍在上升：700 mm 是测量范围的终点，只能当作下限'},
}
# The same model as case a with a 20-cycle burst (the longest the software takes): half the bandwidth, to see what the
# bandwidth does to the distance.
EXTRA = {'a-300kHz-2.0mm-20cycles': dict(CASES['a-300kHz-2.0mm'], cycles=20.)}


def scene(case):
    """The plane-strain strip of one case, as a version-2 scene of the workbench."""
    from offline.domain import templates
    f, d, reach, cycles = case['frequency_mhz'], case['thickness_mm'], case['reach_mm'], case.get('cycles', CYCLES)
    absorber = 40.; length = reach + 10. + absorber; width = 1.6; speed = 5.36   # mm/μs, S0 at these frequency-thickness products
    s = templates()['plate']; s['schema'] = 'workai.hyperelastic.solid.v2'; s.pop('signal', None); s['name'] = f'Ding 2018 · {f * 1e3:g} kHz · {d:g} mm' + (f' · {cycles:g} 周期' if cycles != CYCLES else '')
    s['material'] = dict(MATERIAL); s['geometry'].update(length_mm=length, width_mm=width, thickness_mm=d)
    s['source'].update(face='xmin', center_mm=[0., width / 2, d / 2], radius_mm=0., direction=[1., 0., 0.], amplitude_mpa=2., frequency_mhz=f, cycles=cycles, delay_us=0., phase_deg=0., signal={'type': 'tone_burst', 'window': 'hann'})
    s['sources'] = []; s['boundary'] = {'fixed_face': 'none'}
    s['boundaries'] = [{'type': 'symmetry', 'face': 'ymin'}, {'type': 'symmetry', 'face': 'ymax'}, {'type': 'absorbing', 'face': 'xmax', 'length_mm': absorber, 'ramp': 'cubic', 'attenuation_db': 60.}]
    duration = math_ceil((reach + 10.) / speed + cycles / f * 1.15 + 10., 5.)
    s['acquisition'] = {'duration_us': duration, 'sample_step_us': .2}
    s['receivers'] = [{'name': f'{x:g} mm', 'point_mm': [float(x), width / 2, d], 'direction': [0., 0., 1.]} for x in range(100, int(min(reach, 700.)) + 1, 100)]
    return s


def math_ceil(value, step):
    import math
    return float(math.ceil(value / step) * step)


def request(case):
    return {'scene': scene(case), 'mesh': 'auto', 'controls': 'full_linear', 'record': 'full'}


def run(name, case):
    """Compute one case with the workbench's worker; a finished job with the identical request is reused."""
    from offline.domain import validate_request
    job = JOBS / name; checked = validate_request(request(case)); key = hashlib.sha256(json.dumps(checked, sort_keys=True).encode()).hexdigest(); status = job / 'status.json'
    if status.exists() and json.loads(status.read_text(encoding='utf-8')).get('phase') == 'completed' and (job / 'key').exists() and (job / 'key').read_text() == key: return job
    if not ((job / 'key.pending').exists() and (job / 'key.pending').read_text() == key and (job / 'record').exists()):
        import shutil
        shutil.rmtree(job, ignore_errors=True); job.mkdir(parents=True); (job / 'key.pending').write_text(key)
    (job / 'request.json').write_text(json.dumps(checked), encoding='utf-8'); status.write_text(json.dumps({'job_id': name, 'name': checked['scene']['name'], 'phase': 'starting'}), encoding='utf-8')
    started = time.monotonic(); done = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'offline.worker', '--job-dir', str(job)], cwd=REPO, capture_output=True, text=True)
    state = json.loads(status.read_text(encoding='utf-8'))
    if done.returncode or state.get('phase') != 'completed': raise RuntimeError(f"{name}: {state.get('message')} {done.stderr[-1500:]}")
    (job / 'key').write_text(key); (job / 'elapsed_s').write_text(f'{time.monotonic() - started:.1f}')
    return job


if __name__ == '__main__':
    from offline.preview import preview
    wanted = sys.argv[1:] or list(CASES) + list(EXTRA)
    if wanted == ['check']:
        for name, case in {**CASES, **EXTRA}.items():
            p = preview(request(case)); m = p.get('mesh') or {}; e = p.get('estimate') or {}
            print(name, p['ok'], p['errors'], 'nodes', m.get('nodes'), 'h', (m.get('auto') or {}).get('h_mm'), 'duration', request(case)['scene']['acquisition']['duration_us'], 'seconds', round(e.get('seconds') or 0), 'disk GB', round((e.get('disk_bytes') or 0) / 1e9, 2), 'record', e.get('record'), flush=True)
    else:
        for name in wanted:
            started = time.monotonic(); job = run(name, {**CASES, **EXTRA}[name]); print(name, 'completed', (job / 'elapsed_s').read_text(), 's', flush=True)
