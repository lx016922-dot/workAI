"""Mode amplitudes radiated by the wedge and wafer cases against the normal-mode expansion; writes results/transducers.json.

    WORKAI_VERIFY_JOBS=<dir> python research/transducers/scripts/analyze.py

A surface traction t(x)e^{iωt} drives the mode of wavenumber k that leaves towards +x in proportion to ∫ t(x) e^{ikx} dx,
and the one that leaves towards −x in proportion to ∫ t(x) e^{−ikx} dx; the factor of proportionality belongs to the
mode (its shape and power flow), not to the load. The amplitude of one mode in two cases is therefore in the ratio of
two such integrals:

* wedge, uniform footprint L, trace wavenumber k_t = ω/v:  ∫ = L·sinc((k ∓ k_t)L/2)   (− for the wave that leaves forwards);
* wafer of half length a, edge zone e:                      ∫ = ±2i·e·sinc(ke/2)·sin(k(a − e/2)).

The software's amplitudes: the Fourier sum at f of every receiver over the whole record, then on each side of the
transducer a least-squares fit of the waves e^{∓ikx} of S0 and A0, travelling both ways, with k from the plate theory.
"""
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import cases as C

RESULTS = HERE.parent / 'results'
sinc = lambda v: float(np.sinc(v / math.pi))


def amplitudes(name, modes):
    """{'right'/'left': {mode: (outgoing, incoming) complex amplitudes at f}, 'residual'} from the receivers of a case."""
    t, values, manifest = C.load(name); u = values['murnaghan']; n = len(t) - 1; phase = np.exp(-2j * np.pi * C.F * t[:n]); U = 2 * (phase @ u[:n]) / n
    out = {'elapsed_s': float((C.JOBS / name / 'elapsed_s').read_text()), 'nodes': manifest['nodes'], 'layer_step_us': manifest['layer_step_us']}
    for side, xs, sign, columns in (('right', C.RIGHT, 1., slice(0, 16)), ('left', C.LEFT, -1., slice(16, 32))):
        x = (np.array(xs) - C.CENTRE) * 1e-3; names = list(modes)
        basis = np.stack([np.exp(-1j * sign * modes[m]['k'] * x) for m in names] + [np.exp(1j * sign * modes[m]['k'] * x) for m in names], 1)   # leaving the transducer, then coming back
        fit, *_ = np.linalg.lstsq(basis, U[columns], rcond=None)
        out[side] = {m: {'outgoing': abs(fit[i]), 'returning': abs(fit[i + len(names)])} for i, m in enumerate(names)}
        out[side]['fit_residual'] = float(np.linalg.norm(basis @ fit - U[columns]) / np.linalg.norm(U[columns]))
    return out


def wedge_factor(k, velocity, forwards, apodization='uniform', strips=None):
    """|∫ p(x) T(x) e^{±ikx} dx| / L for the footprint: the ideal travelling load (strips None) or the strips the software uses."""
    w = 2 * math.pi * C.F * 1e6; kt = w / velocity; L = C.FOOTPRINT * 1e-3; x = np.linspace(-L / 2, L / 2, 20001); s = x + L / 2
    taper = .5 * (1 - np.cos(2 * np.pi * s / L)) if apodization == 'hann' else np.ones_like(x)
    if strips is None: carrier = np.exp(-1j * kt * s)
    else:
        nodes = np.linspace(0, L, strips); spacing = L / (strips - 1); carrier = sum(np.clip(1 - np.abs(s - c) / spacing, 0, 1) * np.exp(-1j * kt * c) for c in nodes)
    return float(abs(np.trapezoid(taper * carrier * np.exp((1j if forwards else -1j) * k * x), x)) / L)


def patch_factor(k, length_mm):
    a = length_mm * 1e-3 / 2; e = C.EDGE * 1e-3; return abs(2 * e * sinc(k * e / 2) * math.sin(k * (a - e / 2)))


def main():
    from physics.loads import strips as count
    modes = {m: v for m, v in C.modes().items() if m in ('S0', 'A0')}; table = C.cases(); done = [n for n in table if (C.JOBS / n / 'key').exists()]
    measured = {n: amplitudes(n, modes) for n in done}; velocities = C.wedge_velocities(); lengths = C.patch_lengths(); rows = []
    def compare(kind, reference, factor):
        for name in done:
            if not name.startswith(kind) or name.endswith('-fine') or name == reference: continue
            for mode in modes:
                for side, forwards in (('right', True), ('left', False)):
                    base = measured[reference][side][mode]['outgoing']; ref_theory = factor(reference, mode, forwards)
                    if ref_theory['ideal'] < .05 * max(factor(reference, m, f)['ideal'] for m in modes for f in (True, False)): continue   # a reference the load hardly drives is no reference
                    theory = factor(name, mode, forwards)
                    rows.append({'case': name, 'reference': reference, 'mode': mode, 'direction': '+x' if forwards else '−x', 'software': measured[name][side][mode]['outgoing'] / base,
                                 **{f'theory_{key}': theory[key] / ref_theory[key] for key in theory}})
    def wedge(name, mode, forwards):
        label = name[2:].replace('-hann', ''); v = velocities[label]; apod = 'hann' if name.endswith('-hann') else 'uniform'
        n = count({'trace_velocity_m_s': v, 'length_mm': C.FOOTPRINT}, C.F)
        return {'ideal': wedge_factor(modes[mode]['k'], v, forwards, apod), 'as_implemented': wedge_factor(modes[mode]['k'], v, forwards, apod, n)}
    def wafer(name, mode, forwards): return {'ideal': patch_factor(modes[mode]['k'], lengths[name[2:]])}
    for reference in ('W-S0-peak', 'W-A0-peak'):
        if reference in done: compare('W-', reference, wedge)
    for reference in ('P-S0-peak', 'P-A0-peak'):
        if reference in done: compare('P-', reference, wafer)
    # a wedge sends little backwards: the wave that leaves against the travelling load over the one that leaves with it, for the mode the wedge is set to
    backward = []
    for name in done:
        if not name.startswith('W-') or name.endswith('-fine') or 'peak' not in name: continue
        mode = name[2:4]; ahead, behind = wedge(name, mode, True), wedge(name, mode, False)
        backward.append({'case': name, 'mode': mode, 'software': measured[name]['left'][mode]['outgoing'] / measured[name]['right'][mode]['outgoing'], **{f'theory_{key}': behind[key] / ahead[key] for key in ahead}})
    largest = lambda name: max(measured[name][s][m]['outgoing'] for s in ('right', 'left') for m in modes)
    for r in rows: r['difference_of_reference'] = r['software'] - r['theory_as_implemented' if 'theory_as_implemented' in r else 'theory_ideal']
    out = {'frequency_mhz': C.F, 'thickness_mm': C.THICKNESS, 'modes': modes, 'wedge_velocities_m_s': velocities, 'patch_lengths_mm': lengths, 'footprint_mm': C.FOOTPRINT, 'edge_mm': C.EDGE,
           'amplitudes': measured, 'ratios': rows, 'backward_over_forward': backward,
           'mesh': {coarse: {m: {s: measured[coarse + '-fine'][s][m]['outgoing'] / measured[coarse][s][m]['outgoing'] - 1 for s in ('right', 'left')} for m in modes}
                    for coarse in ('W-S0-peak', 'P-S0-peak') if coarse in done and coarse + '-fine' in done},
           'wafer_symmetry': {n: {m: measured[n]['right'][m]['outgoing'] / measured[n]['left'][m]['outgoing'] - 1 for m in modes if measured[n]['left'][m]['outgoing'] > .05 * largest(n)} for n in done if n.startswith('P-')}}
    RESULTS.mkdir(exist_ok=True); (RESULTS / 'transducers.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')
    for r in rows: print(f"{r['case']:16s} vs {r['reference']:10s} {r['mode']} {r['direction']}  software {r['software']:.4f}  ideal {r['theory_ideal']:.4f}" + (f"  as implemented {r['theory_as_implemented']:.4f}" if 'theory_as_implemented' in r else ''))
    for n in done: print(n, {s: {m: f"{v['outgoing']:.3e}/{v['returning']:.1e}" for m, v in measured[n][s].items() if m in modes} for s in ('right', 'left')}, 'residual', round(measured[n]['right']['fit_residual'], 4), round(measured[n]['left']['fit_residual'], 4))
    for r in backward: print(f"{r['case']:16s} {r['mode']} backward / forward  software {r['software']:.5f}  ideal {r['theory_ideal']:.5f}  as implemented {r['theory_as_implemented']:.5f}")
    print('mesh', out['mesh']); print('wafer symmetry', out['wafer_symmetry'])


if __name__ == '__main__': main()
