"""Gate metrics for coarse-graph predictions against fine references.

Same quantities as `sweep.convergence` (waveform relative L2, exact-frequency
f and 2f amplitudes over the whole record) but with a relative reporting
floor: a receiver enters a metric only if its fine-reference value is at least
`floor` (default 10 %) of the largest value of that metric in the case. The
nonlinear increment of a 1 MPa burst is 1e-15–1e-11 m, mostly below the
absolute 1e-13 m floor used for solver convergence, and its relative accuracy
does not depend on the amplitude (the increment scales with A²).
"""
import numpy as np
from dispersion.analysis import harmonic_coefficients

TOLERANCE = .05


def _features(times, values, frequency):
    h = harmonic_coefficients(times, values, frequency, times[0], times[-1])['harmonics']
    return float(np.sqrt(np.mean(values ** 2))), h[0]['amplitude'], h[1]['amplitude'] if h[1]['available'] else 0.


def compare(times, coarse, fine, frequency, terms=('waveform', 'a1', 'a2'), floor=.1, tolerance=TOLERANCE):
    """coarse, fine: frames × receivers. Returns worst relative errors over the receivers above the floor and the verdict."""
    fc = [_features(times, coarse[:, i], frequency) for i in range(coarse.shape[1])]; ff = [_features(times, fine[:, i], frequency) for i in range(fine.shape[1])]
    rms_f = np.array([x[0] for x in ff]); a1_f = np.array([x[1] for x in ff]); a2_f = np.array([x[2] for x in ff]); a1_c = np.array([x[1] for x in fc]); a2_c = np.array([x[2] for x in fc])
    out = {}
    keep = rms_f >= floor * rms_f.max(); l2 = np.linalg.norm(coarse - fine, axis=0) / np.maximum(np.linalg.norm(fine, axis=0), 1e-300)
    out['waveform'] = {'worst': float(l2[keep].max()), 'receivers': int(keep.sum())}
    for name, c, f in (('a1', a1_c, a1_f), ('a2', a2_c, a2_f)):
        keep = f >= floor * f.max(); rel = np.abs(c - f) / np.maximum(f, 1e-300)
        out[name] = {'worst': float(rel[keep].max()), 'receivers': int(keep.sum()), 'max_reference_m': float(f.max())}
    out['pass'] = bool(all(out[t]['worst'] <= tolerance for t in terms)); out['terms'] = list(terms); out['floor'] = floor
    return out


def vector_relative(times, coarse, fine, frequency, order):
    """‖H_coarse − H_fine‖ / ‖H_fine‖ over all receivers, H the complex Hann-weighted coefficient at order × f (informational)."""
    def coefficients(x):
        out = []
        for i in range(x.shape[1]):
            h = harmonic_coefficients(times, x[:, i], frequency, times[0], times[-1])['harmonics'][order - 1]; out.append(complex(h['real'], h['imaginary']))
        return np.array(out)
    c, f = coefficients(coarse), coefficients(fine)
    return float(np.linalg.norm(c - f) / max(np.linalg.norm(f), 1e-300))


def case_metrics(times, frequency, N, L, fine_N, fine_I, floor=.1):
    """Nonlinear record (waveform, A1, A2) and nonlinear increment (A2) gates for one case."""
    full = compare(times, N, fine_N, frequency, ('waveform', 'a1', 'a2'), floor)
    increment = compare(times, N - L, fine_I, frequency, ('a2',), floor)
    return {'record_l2': float(np.linalg.norm(N - fine_N) / np.linalg.norm(fine_N)), 'increment_l2': float(np.linalg.norm((N - L) - fine_I) / np.linalg.norm(fine_I)),
            'worst_waveform_l2': full['waveform']['worst'], 'worst_a1': full['a1']['worst'], 'worst_a2': full['a2']['worst'],
            'worst_increment_a2': increment['a2']['worst'], 'increment_a2_reference_max_m': increment['a2']['max_reference_m'],
            'waveform_gate': full['pass'], 'increment_gate': increment['pass'], 'passed': bool(full['pass'] and increment['pass']),
            'a2_vector_relative': vector_relative(times, N, fine_N, frequency, 2), 'increment_a2_vector_relative': vector_relative(times, N - L, fine_I, frequency, 2)}
