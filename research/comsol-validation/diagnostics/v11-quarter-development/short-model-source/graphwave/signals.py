"""Explicit excitation definitions shared by the UI and neural input layer."""
import math
import numpy as np

KINDS = ('hann', 'gaussian', 'rectangular', 'chirp')


def specification(scene):
    signal = scene.get('signal', {'kind': 'hann'})
    if not isinstance(signal, dict) or signal.get('kind') not in KINDS or set(signal) - {'kind', 'end_frequency_mhz'}:
        raise ValueError('激励类型需要为 Hann、Gaussian、矩形窗正弦或线性扫频。')
    kind = signal['kind']
    start = scene['source']['frequency_mhz']
    end = signal.get('end_frequency_mhz', start) if kind == 'chirp' else start
    if type(end) not in (int, float) or not math.isfinite(end) or not .18 <= end <= .28:
        raise ValueError('扫频终止频率需为 0.18–0.28 MHz。')
    duration = scene['source']['cycles'] / (.5 * (start + end))
    return {'kind': kind, 'start_frequency_mhz': start, 'end_frequency_mhz': end,
            'duration_us': duration, 'cycles': scene['source']['cycles'],
            'single_carrier': kind != 'chirp', 'window': 'hann' if kind in ('hann', 'chirp') else kind}


def pulse_values(scene, times_us):
    source = scene['source']
    signal = specification(scene)
    t = np.asarray(times_us) - source['delay_us']
    duration = signal['duration_us']
    rate = (signal['end_frequency_mhz'] - signal['start_frequency_mhz']) / duration
    phase = 2 * np.pi * (signal['start_frequency_mhz'] * t + .5 * rate * t * t) + source['phase_deg'] * np.pi / 180
    if signal['kind'] in ('hann', 'chirp'):
        envelope = .5 * (1 - np.cos(2 * np.pi * t / duration))
    elif signal['kind'] == 'gaussian':
        # Six sigma span: the declared burst contains exactly the requested
        # carrier cycles. It is truncated outside [0, duration].
        envelope = np.exp(-.5 * ((t - duration / 2) / (duration / 6))**2)
    else:
        envelope = np.ones_like(t)
    return np.where((t >= 0) & (t <= duration), envelope * np.sin(phase), 0.)
