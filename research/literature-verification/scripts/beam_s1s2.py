"""S1 → S2 in three dimensions: a beam of finite width against the plane-strain plate and scalar diffraction.

    WORKAI_VERIFY_JOBS=<dir> python beam_s1s2.py run        # the 3-D plate (about two hours on 4 cores) and its plane-strain twin
    WORKAI_VERIFY_JOBS=<dir> python beam_s1s2.py analyse    # → results/results-beam.json

Section 9 checks the synchronous pair S1(f) → S2(2f) in plane strain: the model is one element wide. Here the same
8 mm plate (its upper half, 0.4448 MHz) is a three-dimensional body: the end face x = 0 is loaded by an axial traction
over a strip of half width b = 30 mm (4.2 wavelengths of S1 across the whole aperture), the plane y = 0 is a symmetry
plane (the beam axis), the far side and the far end absorb. The beam spreads, and its second harmonic is generated in
a spreading beam.

The reference is not a second three-dimensional solver. In an isotropic plate a Lamb mode spreads in the plane like a
scalar wave with the wavenumber of the mode, so for one frequency

* the fundamental on the axis is the Fresnel field of a slit: the angular spectrum of the aperture, each plane-wave
  component advanced with k_x = √(k² − k_y²);
* the second harmonic obeys the same propagation at 2k with the source κ₂·A₁² (the pair is phase matched, and the
  coupling of two components a few degrees apart is taken as that of the collinear pair): integrated step by step along x;
* what is read is u_x, and the in-plane motion of a component is along its own direction: every component counts with
  cos θ = k_x/k.

Both are compared as ratios to the plane-strain twin — the same plate, burst and mesh, one element wide, where
nothing spreads — so the mode shapes, the excitation of S1 by the end face and most of the burst's bandwidth drop out:

    D₁(x) = |u₁|₃D / |u₁|₂D            against |A₁(x, 0)| of the slit,
    D₂(x) = (u₂/u₁²)₃D / (u₂/u₁²)₂D    against A₂(x, 0) / (κ₂ x A₁(x, 0)²).

u₁ is the linear field on the top face limited to wavenumbers around k, u₂ the increment (complete − linear) around 2k,
both from the complete field the record keeps at four instants (as in `analyze_s1s2.snapshot`).

What the reference leaves out, and so what the comparison cannot resolve (README, section 11): how the excitation by
the end face and the coupling depend on the angle (the waves from the edges of the strip reach the axis at
atan(b/x): 27° at 60 mm, 13° at 130 mm), and the 20-cycle burst (the edge waves arrive up to 2 μs late, in which the
envelope changes by up to 15 %).

    python beam_s1s2.py check      # the reference alone: finer steps, a wider window, and the wide strip against the plane wave
"""
import importlib.util, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location('damping_cases', HERE.parents[1] / 'material-damping' / 'scripts' / 'cases.py'); _harness = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_harness)
JOBS, run, scene = _harness.JOBS, _harness.run, _harness.scene
import cases_s1s2 as first

HALF = first.HALF; ELEMENTS = 3; H = HALF / ELEMENTS
LENGTH, WIDTH, APERTURE = 180., 80., 30.          # mm: plate length, half width of the model, half width of the loaded strip
END_LAYER, SIDE_LAYER = 40., 24.                   # absorbing layers
DURATION = 60.                                     # μs: the record keeps the complete field at 15, 30, 45 and 60 μs
NAMES = {'3d': 'N6-beam', '2d': 'N6-plane'}


def plate(wide, amplitude=2.):
    f = first.frequency_mhz(HALF); width = WIDTH if wide else H
    boundaries = [{'type': 'symmetry', 'face': 'ymin'}, {'type': 'symmetry', 'face': 'zmin'}, {'type': 'absorbing', 'face': 'xmax', 'length_mm': END_LAYER, 'ramp': 'cubic', 'attenuation_db': 60.},
                  {'type': 'absorbing', 'face': 'ymax', 'length_mm': SIDE_LAYER, 'ramp': 'cubic', 'attenuation_db': 60.} if wide else {'type': 'symmetry', 'face': 'ymax'}]
    source = {'face': 'xmin', 'direction': [1., 0., 0.], 'amplitude_mpa': amplitude, 'frequency_mhz': f, 'cycles': first.CYCLES, 'signal': {'type': 'tone_burst', 'window': 'hann'}}
    if wide: source['profile'] = {'type': 'comb', 'pitch_mm': 4 * APERTURE, 'teeth': 1, 'tooth_width_mm': 2 * APERTURE, 'along': 'y', 'alternating_sign': False}   # one tooth: the strip |y| ≤ b
    receivers = [{'name': f'x{x:g}', 'point_mm': [x, 0. if wide else width / 2, HALF], 'direction': [1., 0., 0.]} for x in (40., 80., 120., 160.)]
    s = scene(LENGTH, width, HALF, H, DURATION, .02 * HALF, boundaries=boundaries, source=source, receivers=receivers, name='S1–S2 有限宽声束' if wide else 'S1–S2 平面应变对照')
    s['source']['center_mm'] = [0., 0. if wide else width / 2, HALF / 2]
    return s


def slit(x, k, b, kappa=1., span=40., points=4096, step=2.5e-4):
    """Diffraction of a slit of half width b (m) on the axis, for unit amplitude in the aperture: (A₁, A₂, the same two as the
    x components that are read). A₂ is generated with the coupling `kappa` (per metre) at twice the wavenumber. `x` increasing, in m."""
    L = span * b; y = (np.arange(points) - points // 2) * (L / points); ky = 2 * math.pi * np.fft.fftfreq(points, L / points)
    taper = np.where(np.abs(y) < .42 * L, 1., np.cos(np.clip((np.abs(y) - .42 * L) / (.08 * L), 0., 1.) * math.pi / 2) ** 2)   # the edges of the window absorb what reaches them
    advance = lambda q: np.exp(1j * (np.sqrt(np.maximum(q * q - ky * ky, 0.)) - q) * step) * (np.abs(ky) < q)
    along = lambda field, q: np.fft.ifft(np.fft.fft(field) * np.sqrt(np.maximum(1 - (ky / q) ** 2, 0.)))[points // 2]             # Σ cos θ over the components
    P1, P2 = advance(k), advance(2 * k); a1 = (np.abs(y) <= b).astype(complex); a2 = np.zeros(points, dtype=complex); out = []; at = 0.; centre = points // 2
    for target in x:
        while at < target - 1e-12:
            # one step: half of the source of this step before the propagation, half (with the advanced fundamental) after it
            a2 = np.fft.ifft(np.fft.fft(a2 + .5 * step * kappa * a1 * a1) * P2) * taper
            a1 = np.fft.ifft(np.fft.fft(a1) * P1) * taper
            a2 = a2 + .5 * step * kappa * a1 * a1; at += step
        out.append((a1[centre], a2[centre], along(a1, k), along(a2, 2 * k)))
    return np.array(out).T


def check():
    """The reference against itself: half the step with twice the window and the points; a strip 30 times as wide against the plane wave."""
    f = first.frequency_mhz(HALF) * 1e6; k = 2 * math.pi * f / first.CL; b = APERTURE * 1e-3; x = np.linspace(.04, .13, 10); coarse = slit(x, k, b); fine = slit(x, k, b, span=80., points=16384, step=1.25e-4)
    wide = slit(x, k, 30 * b, span=8.); out = {'refined': {name: float(np.abs(fine[i] / coarse[i] - 1).max()) for i, name in enumerate(('a1', 'a2', 'a1_x', 'a2_x'))},
                                             'plane_wave': {'a1': float(np.abs(np.abs(wide[2]) - 1).max()), 'a2': float(np.abs(np.abs(wide[3] / (x * wide[2] ** 2)) - 1).max())}}
    print(json.dumps(out, indent=1)); return out


def axis_fields(name, index, k):
    """(x grid in m, u₁ around k, u₂ around 2k) on the top face along the beam axis, from the complete field at one stored instant."""
    import analyze_s1s2 as A
    record = JOBS / name / 'record'; manifest = json.loads((record / 'recording.json').read_text(encoding='utf-8')); points = np.load(record / 'mesh.npz')['points']
    frame = manifest['snapshot_frames'][index]; controls = manifest['controls']
    for row in manifest['chunks']:
        with np.load(record / row['file']) as z:
            if frame in z['snapshot_frames']: fields = dict(zip(controls, z['snapshots'][:, list(z['snapshot_frames']).index(frame)])); break
    lines = A.lines_of(points); ids = lines[max(lines)]; x = points[ids, 0]; grid = np.linspace(x[0], x[-1], 8192, endpoint=False); dx = grid[1] - grid[0]
    along = lambda values, centre: A.band(A.resample(x, values[ids, 0], grid), dx, centre)
    return grid, along(fields['linear'], k), along(fields['murnaghan'], 2 * k) - along(fields['linear'], 2 * k), frame * manifest['sample_step_us'], manifest


FAR = 80.   # mm: beyond it the edge waves reach the axis within 21°


def analyse():
    f = first.frequency_mhz(HALF) * 1e6; k = 2 * math.pi * f / first.CL; b = APERTURE * 1e-3
    out = {'frequency_mhz': f * 1e-6, 'wavelength_mm': 2e3 * math.pi / k, 'aperture_half_width_mm': APERTURE, 'rayleigh_distance_mm': (2 * b) ** 2 * k / (8 * math.pi) * 1e3, 'far_mm': FAR, 'reference': check(), 'instants': []}
    for index in (2, 3):
        x, u1w, u2w, time_us, m3 = axis_fields(NAMES['3d'], index, k); _, u1p, u2p, _, m2 = axis_fields(NAMES['2d'], index, k)
        envelope = np.abs(u1p); use = (x >= b) & (x <= (LENGTH - END_LAYER - 8.) * 1e-3) & (envelope >= .3 * envelope.max()); pick = np.flatnonzero(use)[::64]; xs = x[pick]; far = xs >= FAR * 1e-3
        s1, s2, a1, a2 = slit(xs, k, b); d1 = np.abs(u1w[pick]) / np.abs(u1p[pick]); d2 = (u2w[pick] / u1w[pick] ** 2) / (u2p[pick] / u1p[pick] ** 2)
        t1 = np.abs(a1); t2 = a2 / (xs * a1 * a1); e1 = d1 / t1 - 1; e2 = np.abs(d2) / np.abs(t2) - 1; scalar1 = d1 / np.abs(s1) - 1; scalar2 = np.abs(d2) / np.abs(s2 / (xs * s1 * s1)) - 1
        spread = lambda e: {'rms': float(np.sqrt(np.mean(e ** 2))), 'from': float(e.min()), 'to': float(e.max())}
        out['instants'].append({'time_us': time_us, 'x_mm': (xs * 1e3).tolist(), 'envelope': (envelope[pick] / envelope.max()).tolist(),
                                'fundamental': {'software': d1.tolist(), 'theory': t1.tolist(), 'difference': e1.tolist(), 'difference_without_projection': scalar1.tolist()},
                                'second_harmonic_ratio': {'software_abs': np.abs(d2).tolist(), 'theory_abs': np.abs(t2).tolist(), 'difference': e2.tolist(), 'difference_without_projection': scalar2.tolist(),
                                                          'phase_difference_deg': np.degrees(np.angle(d2 / t2)).tolist()},
                                'far': {'points': int(far.sum()), 'fundamental': spread(e1[far]), 'second_harmonic_ratio': spread(e2[far]),
                                        'without_projection': {'fundamental': spread(scalar1[far]), 'second_harmonic_ratio': spread(scalar2[far])}},
                                'near': {'points': int((~far).sum()), 'fundamental': spread(e1[~far]), 'second_harmonic_ratio': spread(e2[~far])} if (~far).any() else None})
        row = out['instants'][-1]; show = lambda r: f"{r['from']:+.1%} … {r['to']:+.1%} (rms {r['rms']:.1%})"
        print(f"t = {time_us:g} μs, x = {xs[0] * 1e3:.0f}–{xs[-1] * 1e3:.0f} mm | beyond {FAR:g} mm ({row['far']['points']} points): D1 {show(row['far']['fundamental'])}, D2 {show(row['far']['second_harmonic_ratio'])}"
              + (f" | nearer ({row['near']['points']}): D1 {show(row['near']['fundamental'])}, D2 {show(row['near']['second_harmonic_ratio'])}" if row['near'] else ''), flush=True)
    out['models'] = {key: {'nodes': m['nodes'], 'layer_step_us': m['layer_step_us'], 'controls': m['controls'], 'max_strain': max(m['max_strain']), 'elapsed_s': float((JOBS / NAMES[key] / 'elapsed_s').read_text())} for key, m in (('3d', m3), ('2d', m2))}
    (HERE.parent / 'results' / 'results-beam.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    if sys.argv[1:] == ['run']:
        for key in ('2d', '3d'): job = run(NAMES[key], plate(key == '3d'), controls='full_linear'); print(NAMES[key], 'done', (job / 'elapsed_s').read_text(), flush=True)
    elif sys.argv[1:] == ['analyse']: analyse()
    elif sys.argv[1:] == ['check']: check()
    else: raise SystemExit(__doc__)
