"""Excitation time signals: analytic families and user-sampled records.

Every source carries `frequency_mhz`, `delay_us`, `phase_deg` and `cycles`.
`delay_us` is always the start of the signal support. The optional `signal`
object selects the waveform family; without it a source is the version-1
Hann tone burst evaluated by `solid.fem.pulse`, and `evaluate` reproduces
that formula so both paths agree.

Amplitude convention: tone bursts are not renormalized (window peak <= 1, as
in version 1). Pulses, chirps and multi-tone bursts are scaled so that the
largest absolute sample over the support is 1, so `amplitude_mpa` remains
the peak traction. Sampled records are normalized to unit peak unless the
user disables it, in which case their own peak must not exceed 1.
"""
import math
import numpy as np
from solid.schema import require, number, exact

TYPES = ('tone_burst', 'gaussian_pulse', 'ricker', 'chirp', 'multi_tone', 'samples')
WINDOWS = ('hann', 'tukey', 'gaussian', 'rectangular')
NORMALIZED = ('gaussian_pulse', 'chirp', 'multi_tone')
MAX_SAMPLES = 20000
GAUSSIAN_TRUNCATION = 3.5       # Gaussian envelope truncated at ±3.5 sigma
RICKER_HALF_WIDTH_CYCLES = 1.5  # Ricker wavelet truncated at ±1.5 / f

def default_signal():
    return {'type': 'tone_burst', 'window': 'hann'}

def signal_of(source):
    return source.get('signal') or default_signal()

def validate_signal(signal, source):
    """Return a normalized copy of `signal` or raise ValueError."""
    require(isinstance(signal, dict) and signal.get('type') in TYPES, '信号类型需为 tone_burst、gaussian_pulse、ricker、chirp、multi_tone 或 samples。')
    s = dict(signal); kind = s['type']
    if kind == 'tone_burst':
        keys = ['type', 'window'] + (['tukey_alpha'] if s.get('window') == 'tukey' else []) + (['gaussian_width'] if s.get('window') == 'gaussian' else [])
        exact(s, keys, '信号'); require(s['window'] in WINDOWS, '窗函数需为 hann、tukey、gaussian 或 rectangular。')
        if s['window'] == 'tukey': s['tukey_alpha'] = number(s['tukey_alpha'], .05, 1, 'tukey_alpha')
        if s['window'] == 'gaussian': s['gaussian_width'] = number(s['gaussian_width'], .05, .5, 'gaussian_width')
    elif kind == 'gaussian_pulse':
        exact(s, ['type', 'bandwidth'], '信号'); s['bandwidth'] = number(s['bandwidth'], .1, 2, '相对带宽')
    elif kind == 'ricker':
        exact(s, ['type'], '信号')
    elif kind == 'chirp':
        exact(s, ['type', 'end_frequency_mhz', 'duration_us', 'window'], '信号')
        s['end_frequency_mhz'] = number(s['end_frequency_mhz'], .001, 10, '终止频率'); s['duration_us'] = number(s['duration_us'], .01, 1000, '线性调频时长')
        require(s['window'] in ('hann', 'rectangular'), '线性调频窗需为 hann 或 rectangular。')
        require(s['duration_us'] * min(source['frequency_mhz'], s['end_frequency_mhz']) >= 1, '线性调频至少需要覆盖一个最低频率周期。')
    elif kind == 'multi_tone':
        exact(s, ['type', 'components', 'duration_us', 'window'], '信号')
        require(isinstance(s['components'], list) and 1 <= len(s['components']) <= 8, '多频分量需要 1–8 个。')
        s['duration_us'] = number(s['duration_us'], .01, 1000, '多频包络时长'); require(s['window'] in ('hann', 'rectangular'), '多频窗需为 hann 或 rectangular。')
        parts = []
        for part in s['components']:
            exact(part, ['frequency_mhz', 'weight', 'phase_deg'], '多频分量')
            parts.append({'frequency_mhz': number(part['frequency_mhz'], .001, 10, '分量频率'), 'weight': number(part['weight'], 0, 1, '分量权重'), 'phase_deg': number(part['phase_deg'], -360, 360, '分量相位')})
        require(max(p['weight'] for p in parts) > 0, '多频分量至少一个权重为正。')
        require(any(abs(p['frequency_mhz'] - source['frequency_mhz']) <= 1e-9 for p in parts), '激励的 frequency_mhz 必须等于某个多频分量，作为分析载频。')
        require(s['duration_us'] * min(p['frequency_mhz'] for p in parts) >= 1, '多频包络至少覆盖一个最低频率周期。')
        s['components'] = parts
    else:
        exact(s, ['type', 'sample_step_us', 'values', 'normalize'], '信号')
        s['sample_step_us'] = number(s['sample_step_us'], .0001, 10, '采样间隔'); require(type(s['normalize']) is bool, 'normalize 需为布尔值。')
        values = s['values']; require(isinstance(values, list) and 8 <= len(values) <= MAX_SAMPLES, f'采样信号需要 8–{MAX_SAMPLES} 个数值。')
        array = np.array([number(v, -1e6, 1e6, '采样值') for v in values]); peak = float(np.max(np.abs(array)))
        require(peak > 0, '采样信号不能全为零。')
        if s['normalize']: array = array / peak
        else: require(peak <= 1 + 1e-12, '不归一化的采样信号峰值不能超过 1；幅值由 amplitude_mpa 给出。')
        s['values'] = array.tolist()
    return s

def _gaussian_sigma_us(frequency_mhz, bandwidth):
    # -6 dB fractional bandwidth b of a Gaussian envelope: 2*sqrt(2 ln 2)/(2 pi sigma) = b f
    return math.sqrt(2 * math.log(2)) / (math.pi * bandwidth * frequency_mhz)

def support_us(source):
    """Start and end of the nonzero signal in microseconds."""
    s = signal_of(source); start = source['delay_us']; f = source['frequency_mhz']
    if s['type'] == 'tone_burst': return start, start + source['cycles'] / f
    if s['type'] == 'gaussian_pulse': return start, start + 2 * GAUSSIAN_TRUNCATION * _gaussian_sigma_us(f, s['bandwidth'])
    if s['type'] == 'ricker': return start, start + 2 * RICKER_HALF_WIDTH_CYCLES / f
    if s['type'] in ('chirp', 'multi_tone'): return start, start + s['duration_us']
    return start, start + (len(s['values']) - 1) * s['sample_step_us']

def highest_frequency_mhz(source):
    s = signal_of(source)
    if s['type'] == 'chirp': return max(source['frequency_mhz'], s['end_frequency_mhz'])
    if s['type'] == 'multi_tone': return max(p['frequency_mhz'] for p in s['components'])
    if s['type'] == 'samples': return .5 / s['sample_step_us']
    return source['frequency_mhz']

def _window(kind, x, params):
    """Window on the unit interval; zero outside [0, 1]."""
    inside = (x >= 0) & (x <= 1)
    if kind == 'rectangular': w = np.ones_like(x)
    elif kind == 'hann': w = .5 * (1 - np.cos(2 * np.pi * x))
    elif kind == 'tukey':
        a = params['tukey_alpha']; edge = a / 2; w = np.ones_like(x)
        left = x < edge; right = x > 1 - edge
        w[left] = .5 * (1 - np.cos(np.pi * x[left] / edge)); w[right] = .5 * (1 - np.cos(np.pi * (1 - x[right]) / edge))
    else: w = np.exp(-.5 * ((x - .5) / params['gaussian_width']) ** 2)
    return np.where(inside, w, 0.)

def _waveform(source, times):
    """Un-normalized waveform at `times` (1-D array of microseconds)."""
    s = signal_of(source); f = source['frequency_mhz']; tau = times - source['delay_us']; phase = math.radians(source['phase_deg']); kind = s['type']
    if kind == 'tone_burst':
        return _window(s['window'], tau / (source['cycles'] / f), s) * np.sin(2 * np.pi * f * tau + phase)
    if kind == 'gaussian_pulse':
        sigma = _gaussian_sigma_us(f, s['bandwidth']); center = GAUSSIAN_TRUNCATION * sigma; local = tau - center
        return np.where((tau >= 0) & (tau <= 2 * center), np.exp(-.5 * (local / sigma) ** 2) * np.sin(2 * np.pi * f * local + phase), 0.)
    if kind == 'ricker':
        half = RICKER_HALF_WIDTH_CYCLES / f; q = (np.pi * f * (tau - half)) ** 2
        return np.where((tau >= 0) & (tau <= 2 * half), (1 - 2 * q) * np.exp(-q), 0.)
    if kind == 'chirp':
        d = s['duration_us']; rate = (s['end_frequency_mhz'] - f) / d
        return _window(s['window'], tau / d, s) * np.sin(2 * np.pi * (f * tau + .5 * rate * tau * tau) + phase)
    if kind == 'multi_tone':
        total = sum(p['weight'] for p in s['components'])
        value = sum(p['weight'] * np.sin(2 * np.pi * p['frequency_mhz'] * tau + math.radians(p['phase_deg'])) for p in s['components']) / total
        return _window(s['window'], tau / s['duration_us'], s) * value
    step = s['sample_step_us']; samples = np.asarray(s['values']); grid = np.arange(len(samples)) * step
    return np.where((tau >= 0) & (tau <= grid[-1]), np.interp(tau, grid, samples, left=0., right=0.), 0.)

def _normalization(source):
    if signal_of(source)['type'] not in NORMALIZED: return 1.
    start, end = support_us(source)
    grid = np.linspace(start, end, max(4096, int(64 * highest_frequency_mhz(source) * (end - start)) + 1))
    peak = float(np.max(np.abs(_waveform(source, grid))))
    return peak if peak > 0 else 1.

def evaluate(source, times_us):
    """Dimensionless signal at `times_us` (scalar or array); multiply by the amplitude."""
    times = np.asarray(times_us, dtype=np.float64); scalar = times.ndim == 0
    value = _waveform(source, np.atleast_1d(times)) / _normalization(source)
    return float(value[0]) if scalar else value

def bandwidth_mhz(source, level_db=30.):
    """Highest frequency whose amplitude spectrum stays within `level_db` of the peak."""
    start, end = support_us(source); f_hi = highest_frequency_mhz(source)
    duration = max(end - start, 1e-9); step = min(1 / (64 * f_hi), duration / 2048)
    count = int(2 ** math.ceil(math.log2(4 * duration / step)))
    spectrum = np.abs(np.fft.rfft(evaluate(source, start + np.arange(count) * step))); freqs = np.fft.rfftfreq(count, step)
    above = np.flatnonzero(spectrum >= spectrum.max() * 10 ** (-level_db / 20))
    return float(freqs[above[-1]]) if len(above) else f_hi

def describe(source):
    """Diagnostics entry: family, support, peak and bandwidth."""
    s = signal_of(source); start, end = support_us(source)
    values = evaluate(source, np.linspace(start, end, 4097))
    return {'type': s['type'], 'window': s.get('window'), 'support_us': [start, end], 'peak': float(np.max(np.abs(values))),
            'nominal_frequency_mhz': source['frequency_mhz'], 'bandwidth_30db_mhz': bandwidth_mhz(source, 30.),
            'normalized_unit_peak': s['type'] in NORMALIZED or (s['type'] == 'samples' and s['normalize'])}
