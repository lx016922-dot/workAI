"""Signal analysis for the nonlinear guided-wave workflow (pure NumPy, no solver).

`harmonic_coefficients` is the Python counterpart of the browser's analyzeTimeWindow:
Hann window on the selected samples, weighted mean removed, exact-frequency
single-sided complex coefficient, phase referenced to record time zero.
"""
import math
import numpy as np


def harmonic_coefficients(times_us, values, frequency_mhz, start_us=None, end_us=None, orders=(1, 2)):
    times = np.asarray(times_us, dtype=float); values = np.asarray(values, dtype=float)
    if times.shape != values.shape or len(times) < 4: raise ValueError('Waveform and time grid must match and hold at least 4 samples.')
    step = times[1] - times[0]
    if not np.allclose(np.diff(times), step, atol=1e-8): raise ValueError('Uniform time sampling required.')
    start = times[0] if start_us is None else start_us; end = times[-1] if end_us is None else end_us
    if not start < end or start < times[0] - 1e-9 or end > times[-1] + 1e-9: raise ValueError('Analysis window must lie inside the record.')
    mask = (times >= start - 1e-9) & (times <= end + 1e-9); t = times[mask]; v = values[mask]; count = len(t)
    if count < 8: raise ValueError('Analysis window needs at least 8 samples.')
    window = .5 * (1 - np.cos(2 * np.pi * np.arange(count) / (count - 1))); total = window.sum(); mean = (v * window).sum() / total
    nyquist = .5 / step; harmonics = []
    for order in orders:
        f = order * frequency_mhz
        if f > nyquist: harmonics.append({'order': order, 'frequency_mhz': f, 'available': False}); continue
        angle = 2 * np.pi * f * t; weighted = (v - mean) * window
        real = (weighted * np.cos(angle)).sum(); imaginary = -(weighted * np.sin(angle)).sum(); scale = (1 if abs(f - nyquist) < 1e-12 else 2) / total
        real *= scale; imaginary *= scale; amplitude = math.hypot(real, imaginary)
        harmonics.append({'order': order, 'frequency_mhz': f, 'available': True, 'real': real, 'imaginary': imaginary, 'amplitude': amplitude,
                          'phase_deg': math.degrees(math.atan2(imaginary, real)) if amplitude > 1e-300 else None})
    return {'start_us': float(t[0]), 'end_us': float(t[-1]), 'sample_count': count, 'carrier_cycles': float((t[-1] - t[0]) * frequency_mhz), 'harmonics': harmonics}


def line_harmonics(distances_mm, times_us, traces, frequency_mhz, window_us=None, group_velocity_m_s=None, burst_us=None, delay_us=0., margin_fraction=.25):
    """Per-point harmonics along a receiver line. Fixed window, or a window moving with x / cg."""
    traces = np.asarray(traces); rows = []
    for i, x in enumerate(distances_mm):
        if group_velocity_m_s is not None:
            if burst_us is None: raise ValueError('A moving window needs the burst duration.')
            travel = x * 1e-3 / group_velocity_m_s * 1e6; margin = margin_fraction * burst_us
            start, end = delay_us + travel - margin, delay_us + travel + burst_us + margin
        elif window_us is not None: start, end = window_us
        else: start, end = None, None
        start = None if start is None else max(start, times_us[0]); end = None if end is None else min(end, times_us[-1])
        analysis = harmonic_coefficients(times_us, traces[:, i], frequency_mhz, start, end)
        rows.append({'index': i, 'distance_mm': float(x), 'window_us': [analysis['start_us'], analysis['end_us']], 'carrier_cycles': analysis['carrier_cycles'], 'harmonics': analysis['harmonics']})
    return rows


def accumulation_fit(rows):
    """Linear fits of A2 and of A2 / A1^2 against distance; slope and R^2 are reported, not interpreted."""
    x = np.array([r['distance_mm'] for r in rows]) * 1e-3
    a1 = np.array([r['harmonics'][0]['amplitude'] for r in rows]); a2 = np.array([r['harmonics'][1]['amplitude'] for r in rows])
    def fit(y):
        slope, intercept = np.polyfit(x, y, 1); predicted = slope * x + intercept; residual = ((y - predicted)**2).sum(); total = ((y - y.mean())**2).sum()
        return {'slope_per_m': float(slope), 'intercept': float(intercept), 'r_squared': float(1 - residual / total) if total > 0 else None}
    ratio = a2 / a1**2
    return {'points': len(rows), 'distance_mm': (x * 1e3).tolist(), 'a1': a1.tolist(), 'a2': a2.tolist(), 'a2_over_a1_squared_per_m': ratio.tolist(),
            'fit_a2_vs_x': fit(a2), 'fit_ratio_vs_x': fit(ratio), 'a1_variation': float((a1.max() - a1.min()) / a1.mean()) if a1.mean() > 0 else None,
            'note': 'A2/A1^2 growing linearly with x is consistent with cumulative generation; it is not by itself a material beta and includes geometric and source contributions.'}


def amplitude_scaling(rows_a, rows_2a, tolerance=.1):
    """Doubling the drive should double A1 and quadruple A2 for a quadratic nonlinearity."""
    checks = []
    for a, b in zip(rows_a, rows_2a):
        r1 = b['harmonics'][0]['amplitude'] / a['harmonics'][0]['amplitude']; r2 = b['harmonics'][1]['amplitude'] / a['harmonics'][1]['amplitude'] if a['harmonics'][1]['amplitude'] > 0 else None
        checks.append({'distance_mm': a['distance_mm'], 'a1_ratio': r1, 'a2_ratio': r2, 'a1_pass': abs(r1 - 2) <= 2 * tolerance, 'a2_pass': r2 is not None and abs(r2 - 4) <= 4 * tolerance})
    return {'tolerance': tolerance, 'checks': checks, 'all_pass': all(c['a1_pass'] and c['a2_pass'] for c in checks)}


def envelope(values):
    """Analytic-signal magnitude via FFT (Hilbert transform)."""
    v = np.asarray(values, dtype=float); n = len(v); spectrum = np.fft.fft(v); h = np.zeros(n)
    h[0] = 1; h[1:(n + 1) // 2] = 2
    if n % 2 == 0: h[n // 2] = 1
    return np.abs(np.fft.ifft(spectrum * h))


def line_velocities(distances_mm, times_us, traces, frequency_mhz, window_us=None):
    """Phase velocity from the unwrapped phase slope along x, group velocity from envelope-peak arrival times."""
    x = np.asarray(distances_mm, dtype=float) * 1e-3; traces = np.asarray(traces); phases = []; arrivals = []
    for i in range(len(x)):
        start, end = (None, None) if window_us is None else window_us
        h = harmonic_coefficients(times_us, traces[:, i], frequency_mhz, start, end, orders=(1,))['harmonics'][0]
        phases.append(math.atan2(h['imaginary'], h['real'])); arrivals.append(times_us[int(np.argmax(envelope(traces[:, i])))])
    phase = np.unwrap(np.array(phases)); slope, intercept = np.polyfit(x, phase, 1); omega = 2 * np.pi * frequency_mhz * 1e6
    k = -slope   # coefficient convention: a wave travelling to +x has phase decreasing with x
    predicted = slope * x + intercept; r2 = 1 - ((phase - predicted)**2).sum() / max(((phase - phase.mean())**2).sum(), 1e-300)
    gslope, _ = np.polyfit(x, np.array(arrivals) * 1e-6, 1)
    return {'phase_velocity_m_s': float(omega / abs(k)) if k != 0 else None, 'wavenumber_rad_per_m': float(k), 'phase_fit_r_squared': float(r2),
            'group_velocity_m_s': float(1 / gslope) if gslope != 0 else None, 'arrival_times_us': [float(a) for a in arrivals], 'frequency_mhz': frequency_mhz}


def fk_spectrum(distances_mm, times_us, traces, pad=4):
    """Magnitude spectrum over (frequency, wavenumber) for a uniformly spaced receiver line.

    Sign convention: a wave exp(i(kx - wt)) travelling towards +x appears at positive k and positive f."""
    x = np.asarray(distances_mm, dtype=float) * 1e-3; t = np.asarray(times_us, dtype=float) * 1e-6; traces = np.asarray(traces)
    dx = np.diff(x); dt = np.diff(t)
    if not (np.allclose(dx, dx[0], rtol=1e-6) and np.allclose(dt, dt[0], rtol=1e-6)): raise ValueError('Uniform spacing in x and t required.')
    nt, nx = traces.shape; window = np.outer(np.hanning(nt), np.hanning(nx)); data = traces * window
    nt_pad = nt * pad; nx_pad = nx * pad
    spectrum = np.fft.fft(np.fft.ifft(data, n=nx_pad, axis=1) * nx_pad, n=nt_pad, axis=0)   # ifft over x gives e^{+ikx}, fft over t gives e^{-iwt}
    frequency = np.fft.fftfreq(nt_pad, dt[0]); wavenumber = 2 * np.pi * np.fft.fftfreq(nx_pad, dx[0])
    keep_f = frequency >= 0; order_k = np.argsort(wavenumber)
    magnitude = np.abs(spectrum[keep_f][:, order_k])
    return {'frequency_mhz': (frequency[keep_f] * 1e-6).tolist(), 'wavenumber_rad_per_mm': (wavenumber[order_k] * 1e-3).tolist(), 'magnitude': magnitude}


def mode_overlay(curves, fmax_mhz):
    """k(f) of each tracked mode for drawing on an f–k spectrum (rad/mm against MHz)."""
    rows = []
    for m in curves['modes']:
        f = np.array(m['frequency_mhz']); keep = f <= fmax_mhz
        rows.append({'name': m['name'], 'frequency_mhz': f[keep].tolist(), 'wavenumber_rad_per_mm': np.array(m['wavenumber_rad_per_mm'])[keep].tolist()})
    return rows


def thickness_decomposition(ux, uz, shapes):
    """Least-squares amplitudes of mode shapes in a measured through-thickness profile at one frequency.

    `ux`, `uz`: complex arrays over the same z samples as each shape's complex_ux / complex_uz."""
    measured = np.concatenate([np.asarray(ux, dtype=complex), np.asarray(uz, dtype=complex)])
    basis = np.stack([np.concatenate([np.asarray(s['complex_ux']), np.asarray(s['complex_uz'])]) for s in shapes], axis=1)
    amplitudes, *_ = np.linalg.lstsq(basis, measured, rcond=None); fitted = basis @ amplitudes
    residual = float(np.linalg.norm(measured - fitted) / max(np.linalg.norm(measured), 1e-300))
    return {'modes': [s.get('name', f'mode{i}') for i, s in enumerate(shapes)], 'amplitudes': [complex(a) for a in amplitudes], 'magnitudes': np.abs(amplitudes).tolist(),
            'relative_residual': residual, 'note': 'Least squares over the supplied shapes; residual above a few percent means missing modes, evanescent fields or mesh error.'}
