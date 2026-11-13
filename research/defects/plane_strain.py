"""Defects in plane strain: the network against an independent 2D finite-element reference (research/defects/fem2d.py).

    python research/defects/plane_strain.py network            # the frozen network on the 3D strip (both y faces symmetric = plane strain)
    python research/defects/plane_strain.py fem 0.05 0.025     # the Q1 reference on two grids (mm)
    python research/defects/plane_strain.py fem 0.0125 PS1-open-crack PS2-open-delamination PS4-closed-crack-1MPa PS6-closed-delamination
    python research/defects/plane_strain.py refine PS1-open-crack 0.79   # the network on a finer mesh than the automatic one
    python research/defects/plane_strain.py analyse            # → plane-strain-results.json

60 × 1.6 mm aluminium plate; uniform traction on the x = 0 end at 45° (x and z, so S0 and A0 are both excited),
0.3 MHz 5-cycle Hann, 1 MPa; 10 mm absorbing layer at the x = 60 end (cubic, 60 dB); 35 μs record. Defects:
crack x = 30, 0.8 mm deep from the top (open / closed); delamination x = 25–35 at 0.5 mm below the top (open /
closed). Linear material, so the closed defects' nonlinearity is the contact alone.
"""
import copy
import json
import math
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
OUT = HERE / 'out' / 'plane-strain'
MATERIAL = {'rho_kg_m3': 2700., 'lambda_pa': 58.1e9, 'mu_pa': 26.1e9, 'l_pa': -143e9, 'm_pa': -297e9, 'n_pa': -345e9}
RECEIVERS = [('top_20', 20., 1.6), ('top_40', 40., 1.6), ('mid_20', 20., .8), ('mid_40', 40., .8)]
CRACK = {'type': 'crack', 'x_mm': 30., 'depth_mm': .8, 'from': 'top'}
DELAMINATION = {'type': 'delamination', 'x_start_mm': 25., 'x_end_mm': 35., 'depth_mm': .5}
CASES = {
    'PS0-intact': ([], 1.), 'PS1-open-crack': ([CRACK], 1.), 'PS2-open-delamination': ([DELAMINATION], 1.),
    'PS4-closed-crack-1MPa': ([dict(CRACK, closed=True)], 1.), 'PS5-closed-crack-2MPa': ([dict(CRACK, closed=True)], 2.),
    'PS6-closed-delamination': ([dict(DELAMINATION, closed=True)], 1.),
}
W = 1.5


def scene(defects, amplitude, width=W, sample_us=.05):
    s = {'schema': 'workai.hyperelastic.solid.v2', 'name': 'plane strain', 'constitutive': 'linear', 'material': dict(MATERIAL),
         'geometry': {'kind': 'plate', 'length_mm': 60., 'width_mm': width, 'thickness_mm': 1.6, 'outer_radius_mm': 3., 'inner_radius_mm': 2., 'hole_radius_mm': .5,
                      'hole_x_mm': 6., 'hole_y_mm': width / 2, 'notch_x_mm': 6., 'notch_width_mm': 1., 'notch_depth_mm': .6},
         'source': {'face': 'xmin', 'center_mm': [0., width / 2, .8], 'radius_mm': 0., 'direction': [1., 0., 1.], 'amplitude_mpa': amplitude, 'frequency_mhz': .3, 'cycles': 5.,
                    'delay_us': 0., 'phase_deg': 0., 'signal': {'type': 'tone_burst', 'window': 'hann'}},
         'sources': [], 'boundary': {'fixed_face': 'none'},
         'boundaries': [{'type': 'symmetry', 'face': 'ymin'}, {'type': 'symmetry', 'face': 'ymax'},
                        {'type': 'absorbing', 'face': 'xmax', 'length_mm': 10., 'ramp': 'cubic', 'attenuation_db': 60.}],
         'acquisition': {'duration_us': 35., 'sample_step_us': sample_us}, 'numerics': {'h_mm': 1., 'cfl': .3, 'max_elements': 40000},
         'receivers': [{'name': n, 'point_mm': [x, width / 2, z], 'direction': [1., 0., 0.]} for n, x, z in RECEIVERS]}
    if defects: s['defects'] = copy.deepcopy(defects)
    return s


def fem_case(defects, amplitude):
    splits = []
    for d in defects:
        if d['type'] == 'crack': splits.append({'axis': 'x', 'at': d['x_mm'], 'from': 1.6 - d['depth_mm'], 'to': 1.6, 'contact': d.get('closed', False)})
        else: splits.append({'axis': 'z', 'at': 1.6 - d['depth_mm'], 'from': d['x_start_mm'], 'to': d['x_end_mm'], 'contact': d.get('closed', False)})
    return {'L': 60., 'T': 1.6, 'rho': MATERIAL['rho_kg_m3'], 'lam': MATERIAL['lambda_pa'], 'mu': MATERIAL['mu_pa'], 'direction': [1., 1.], 'amplitude_mpa': amplitude,
            'f_mhz': .3, 'cycles': 5., 'duration_us': 35., 'sample_us': .05, 'absorber_mm': 10., 'splits': splits, 'receivers': RECEIVERS}


def network(cases=None, tag='network'):
    from offline.domain import validate_request
    from graphnet.run import simulate
    OUT.mkdir(parents=True, exist_ok=True)
    for name in cases or CASES:
        defects, amplitude = CASES[name]
        request = validate_request({'scene': scene(defects, amplitude), 'mesh': 'auto'}); started = time.monotonic()
        r = simulate(request['scene'], degree=request['degree'], laws='scene'); d = r['computed']['diagnostics']
        p = r['computed']['probes_m']   # frames × receivers × 3 → keep x and z
        np.savez_compressed(OUT / f'{name}-{tag}.npz', times_us=r['computed']['times_us'], probes_m=p[:, :, [0, 2]])
        info = {'nodes': len(r['mesh']['points']), 'h_mm': request['scene']['numerics']['h_mm'], 'steps': d['internal_steps'], 'seconds': time.monotonic() - started,
                'energy_balance': d['external_work_balance_relative'], 'contact': d.get('contact'), 'mesh': request['mesh_report']}
        (OUT / f'{name}-{tag}.json').write_text(json.dumps(info, indent=1, default=float) + '\n', encoding='utf-8')
        print(name, 'network', {k: info[k] for k in ('nodes', 'h_mm', 'steps', 'seconds', 'energy_balance')}, flush=True)


def refine(cases, grids):
    """The network on finer meshes than the automatic one (element size h, degree 4). The fields do not vary across the strip,
    so one element across it is exact; the strip is made one element wide and the sampling coarsened to keep the field store
    under the network's 512 MiB."""
    from offline.domain import _bounded_network_scene   # the workbench limits h to 0.8–2.5 mm; the scene rules themselves do not
    from graphnet.model import load
    from graphnet.run import simulate
    for name in cases:
        defects, amplitude = CASES[name]
        for h in grids:
            sample = .05
            while True:
                s = scene(defects, amplitude, width=h, sample_us=sample); s['numerics']['h_mm'] = h
                checked = _bounded_network_scene(s, 4, load(4, long_record=True)['material'])
                try:
                    started = time.monotonic(); r = simulate(checked, degree=4, laws='scene'); break
                except ValueError as error:
                    if '场存储' not in str(error) or sample >= .2: raise
                    sample = round(sample + .05, 2)
            d = r['computed']['diagnostics']
            np.savez_compressed(OUT / f'{name}-network-h{h:g}.npz', times_us=r['computed']['times_us'], probes_m=r['computed']['probes_m'][:, :, [0, 2]])
            info = {'nodes': len(r['mesh']['points']), 'h_mm': h, 'width_mm': h, 'sample_us': sample, 'steps': d['internal_steps'],
                    'seconds': time.monotonic() - started, 'energy_balance': d['external_work_balance_relative'], 'contact': d.get('contact')}
            (OUT / f'{name}-network-h{h:g}.json').write_text(json.dumps(info, indent=1, default=float) + '\n', encoding='utf-8')
            print(name, 'network h', h, {k: info[k] for k in ('nodes', 'sample_us', 'steps', 'seconds')}, flush=True)


def fem(grids, cases=None):
    from research.defects.fem2d import simulate
    OUT.mkdir(parents=True, exist_ok=True)
    for h in grids:
        for name in cases or CASES:
            defects, amplitude = CASES[name]
            started = time.monotonic(); r = simulate(fem_case(defects, amplitude), h)
            np.savez_compressed(OUT / f'{name}-fem{h:g}.npz', times_us=r['times_us'], probes_m=r['probes_m'], energy=r['energy'])
            info = {k: r[k] for k in ('dt_s', 'steps', 'nodes', 'elements', 'contact_pairs')}; info.update(h_mm=h, seconds=time.monotonic() - started)
            (OUT / f'{name}-fem{h:g}.json').write_text(json.dumps(info, indent=1) + '\n', encoding='utf-8')
            print(name, 'fem', h, info, flush=True)


def harmonic(trace, dt_us, f):
    n = 1 << 16; spec = np.abs(np.fft.rfft(trace * np.hanning(len(trace)), n)); freqs = np.fft.rfftfreq(n, dt_us)
    return float(spec[np.argmin(np.abs(freqs - 2 * f))] / spec[np.argmin(np.abs(freqs - f))])


FEM_GRIDS = ('0.05', '0.025', '0.0125')


def reference(name, receiver, component):
    """The finite-element trace extrapolated to zero element size, and how it was obtained.

    With three grids the order is measured from the trace itself (p = log2(|F1 − F2| / |F2 − F3|)) and Richardson
    extrapolation uses it; a crack tip gives p ≈ 1 (r^1/2 singularity), a smooth field p ≈ 2. With two grids p = 2
    is assumed (used only for the intact plate). The 2 MPa closed crack is twice the 1 MPa one: the reference
    is exactly homogeneous (checked below), so its fine-grid run is not repeated."""
    if name == 'PS5-closed-crack-2MPa':
        trace, info = reference('PS4-closed-crack-1MPa', receiver, component); return 2 * trace, dict(info, scaled_from='PS4-closed-crack-1MPa × 2')
    grids = [g for g in FEM_GRIDS if (OUT / f'{name}-fem{g}.npz').exists()]
    F = [np.load(OUT / f'{name}-fem{g}.npz')['probes_m'][:, receiver, component] for g in grids]
    if len(F) == 3:
        order = float(np.clip(np.log2(np.linalg.norm(F[0] - F[1]) / np.linalg.norm(F[1] - F[2])), .5, 4.))
    else:
        order = 2.
    E = F[-1] + (F[-1] - F[-2]) / (2 ** order - 1)
    return E, {'grids_mm': [float(g) for g in grids], 'order': order, 'finest_vs_extrapolated': float(np.linalg.norm(F[-1] - E) / np.linalg.norm(E))}


def lowpass(trace, dt_us, top_mhz):
    """The trace without its content above top_mhz (raised-cosine roll-off over the last 10 %)."""
    spec = np.fft.rfft(trace); freqs = np.fft.rfftfreq(len(trace), dt_us)
    gain = np.clip((top_mhz - freqs) / (.1 * top_mhz), 0., 1.); gain = .5 - .5 * np.cos(np.pi * gain)
    return np.fft.irfft(spec * gain, len(trace))


def analyse():
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    load = lambda name, kind: np.load(OUT / f'{name}-{kind}.npz')['probes_m']
    info = lambda name, kind: json.loads((OUT / f'{name}-{kind}.json').read_text(encoding='utf-8'))
    components = {'top': 1, 'mid': 0}   # top: u_z, mid-plane: u_x
    rows = {}
    for name, (defects, amplitude) in CASES.items():
        row = {'network_runs': {kind: {k: info(name, kind).get(k) for k in ('nodes', 'steps', 'seconds')}
                                for kind in ('network', 'network-v1.1.0')}, 'receivers': {}}
        top = info(name, 'network')['mesh']['band_top_mhz']
        for i, (rx, _, _) in enumerate(RECEIVERS):
            c = components[rx.split('_')[0]]
            ref, how = reference(name, i, c); base, _ = reference('PS0-intact', i, c); base = amplitude * base
            entry = {'component': 'u' + 'xz'[c], 'reference': how, 'harmonic_2f_reference': harmonic(ref, .05, .3), 'mesh_band_top_mhz': top}
            for kind, key in (('network', 'now'), ('network-v1.1.0', 'v1.1.0')):
                a = load(name, kind)[:, i, c]
                entry[f'network_{key}_vs_reference'] = rel(a, ref); entry[f'harmonic_2f_network_{key}'] = harmonic(a, .05, .3)
                # within the band the automatic mesh is sized for (2 × the −30 dB band edge): the clapping of a closed
                # defect also radiates 4f, 5f, … that lie above it
                entry[f'network_{key}_vs_reference_in_mesh_band'] = rel(lowpass(a, .05, top), lowpass(ref, .05, top))
                if defects:
                    entry[f'scattered_network_{key}_vs_reference'] = float(np.linalg.norm((a - amplitude * load('PS0-intact', 'network')[:, i, c]) - (ref - base)) / np.linalg.norm(ref - base))
            if defects: entry['scattered_to_incident'] = float(np.linalg.norm(ref - base) / np.linalg.norm(base))
            for kind in ('network-h1', 'network-h0.8', 'network-h0.79'):
                if (OUT / f'{name}-{kind}.npz').exists(): entry[f'{kind}_vs_reference'] = rel(load(name, kind)[:, i, c], ref)
            row['receivers'][rx] = entry
        rows[name] = row
    rows['homogeneity_max_PS5_minus_2PS4'] = {kind: float(np.max(np.abs(load('PS5-closed-crack-2MPa', kind) - 2 * load('PS4-closed-crack-1MPa', kind))) / np.max(np.abs(load('PS5-closed-crack-2MPa', kind))))
                                             for kind in ('network', 'fem0.025')}
    (HERE / 'plane-strain-results.json').write_text(json.dumps(rows, indent=1) + '\n', encoding='utf-8')
    return rows


if __name__ == '__main__':
    if sys.argv[1] == 'network': network(sys.argv[2].split(',') if len(sys.argv) > 2 else None)
    elif sys.argv[1] == 'fem': fem([float(x) for x in sys.argv[2:] if x[0].isdigit()], [x for x in sys.argv[2:] if not x[0].isdigit()])
    elif sys.argv[1] == 'refine': refine(sys.argv[2].split(','), [float(x) for x in sys.argv[3:]])
    else: print(json.dumps(analyse(), indent=1))
