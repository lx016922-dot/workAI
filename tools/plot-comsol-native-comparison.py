"""Plot verified, unaligned native COMSOL and frozen neural 5 ms records."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location('compare_native', Path(__file__).with_name('compare-comsol-reference.py'))
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--prediction-archive', type=Path, required=True)
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config, receipt, controls = compare.load_case(args.case)
    archive = json.loads((args.prediction_archive / 'archive.json').read_text())
    for name, digest in archive['files_sha256'].items():
        if Path(name).name != name or sha(args.prediction_archive / name) != digest:
            raise ValueError('Prediction archive changed')
    identity = json.loads((args.prediction_archive / 'identity.json').read_text())
    report = json.loads(args.comparison.read_text())
    if identity['case']['scene'] != config['scene'] or report['reference_job_id'] != receipt['job_id']:
        raise ValueError('Physical scene or reference identity differs')
    if report['candidate']['prediction_archive_sha256'] != sha(args.prediction_archive / 'archive.json'):
        raise ValueError('Comparison used a different prediction')
    with np.load(args.prediction_archive / 'traces.npz', allow_pickle=False) as saved:
        time = saved['times_us']; predicted = saved['predicted'][0]
    expected = controls[0]
    if predicted.shape != expected.shape or len(time) != 25001 or time[-1] != 5000:
        raise ValueError('Complete matching 5 ms records required')
    actual_metrics = compare.metrics(predicted, expected)
    if actual_metrics != report['comparisons']['linear']:
        raise ValueError('Plot data do not reproduce comparison metrics')

    # Fixed receiver P6 is the predefined center-line point, not selected by error.
    receiver = 5
    point = config['scene']['receivers'][receiver]['point_mm']
    error_sum = np.cumsum(np.sum((predicted - expected) ** 2, axis=2), axis=0)
    reference_sum = np.cumsum(np.sum(expected ** 2, axis=2), axis=0)
    reference_rms = np.sqrt(reference_sum / np.arange(1, len(time) + 1)[:, None])
    relative = np.full_like(error_sum, np.nan)
    np.divide(error_sum, reference_sum, out=relative, where=reference_rms > 1e-13)
    relative = 100 * np.sqrt(relative)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = args.output.with_suffix('.csv')
    with table.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_us'] + ['P' + str(i + 1) + '_cumulative_vector_relative_l2_percent' for i in range(8)])
        for index in range(0, len(time), 250):
            writer.writerow([float(time[index])] + [float(x) if np.isfinite(x) else '' for x in relative[index]])

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#28313a', 'text.color': '#28313a',
                         'axes.edgecolor': '#697480', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    blue, orange = '#246495', '#b85723'
    for ax, limits, title in [(axes[0, 0], (0, 50), 'Initial 50 microseconds'),
                              (axes[0, 1], (4950, 5000), 'Final 50 microseconds')]:
        mask = (time >= limits[0]) & (time <= limits[1])
        ax.plot(time[mask], 1e9 * expected[mask, receiver, 0], color=blue, lw=1.7, label='Native COMSOL')
        ax.plot(time[mask], 1e9 * predicted[mask, receiver, 0], color=orange, lw=1.5, ls='--', label='Neural v6')
        ax.set(xlim=limits, xlabel='Time (microseconds)', ylabel='P6 longitudinal displacement (nm)', title=title)
        ax.grid(axis='y', color='#e5e8eb', lw=.6)
    amplitude = 1e9 * max(np.max(abs(expected[:, receiver, 0])), np.max(abs(predicted[:, receiver, 0])))
    for ax in axes[0]:
        ax.set_ylim(-1.08 * amplitude, 1.08 * amplitude)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(.5, .895), ncol=2, frameon=False)

    ax = axes[1, 0]
    for index in range(8):
        ax.plot(time / 1000, relative[:, index], color=blue if index == receiver else '#9ba5af',
                lw=1.8 if index == receiver else .7, alpha=1 if index == receiver else .65,
                label='P6' if index == receiver else 'Other seven points' if index == 0 else None)
    ax.axhline(5, color='#28313a', lw=1, ls=':', label='5% criterion')
    ax.set(xlim=(0, 5), ylim=(0, max(160, np.nanmax(relative) * 1.03)), xlabel='Record endpoint (ms)',
           ylabel='Cumulative vector relative L2 (%)', title='Accumulated waveform difference at all eight points')
    ax.grid(axis='y', color='#e5e8eb', lw=.6); ax.legend(frameon=False, loc='lower right')

    ax = axes[1, 1]
    # Full-window amplitude spectrum, without alignment or frequency correction.
    window = np.hanning(len(time)); window /= window.sum()
    frequency = np.fft.rfftfreq(len(time), d=.2e-6) / 1000
    for value, color, style, label in [(expected, blue, '-', 'Native COMSOL'), (predicted, orange, '--', 'Neural v6')]:
        trace = value[:, receiver, 0]
        spectrum = 2e9 * abs(np.fft.rfft(window * (trace - window @ trace)))
        ax.plot(frequency, spectrum, color=color, lw=1.5, ls=style, label=label)
    ax.set(xlim=(150, 175), xlabel='Frequency (kHz)', ylabel='P6 displacement amplitude (nm)',
           title='Dominant longitudinal peak (full 5 ms window)')
    ax.set_ylim(bottom=0); ax.grid(axis='y', color='#e5e8eb', lw=.6)
    fig.suptitle('Frozen neural v6 versus native COMSOL: Hann burst, 6 cycles, 5 ms', fontsize=16, y=.98)
    fig.text(.5, .935, f'Linear control only; P6 = {point} mm; 25,001 samples; no time alignment or amplitude scaling', ha='center', fontsize=10)
    fig.text(.075, .035, 'The COMSOL time/mesh convergence is not certified. This comparison fails the waveform criterion.\n'
             'The two waveform panels share a vertical scale; cumulative errors use every sample, not the plotted CSV subset.', fontsize=10)
    fig.subplots_adjust(top=.835, bottom=.15, hspace=.42, wspace=.3)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    provenance = {'schema': 'workai.comsol_native_plot.v1', 'reference_job_id': receipt['job_id'],
                  'comparison_sha256': sha(args.comparison), 'prediction_archive_sha256': sha(args.prediction_archive / 'archive.json'),
                  'selected_receiver': receiver, 'point_mm': point, 'frames': len(time), 'duration_us': float(time[-1]),
                  'controls_plotted': ['linear'], 'csv_stride': 250, 'csv_used_for_metrics': False,
                  'plot_sha256': sha(args.output), 'csv_sha256': sha(table),
                  'reference_convergence_verified': False, 'scientific_accuracy_certified': False}
    args.output.with_suffix('.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance))


if __name__ == '__main__':
    main()
