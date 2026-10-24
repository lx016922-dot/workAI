"""Render the audited signal/cycle matrix without changing its acceptance rules."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--candidate-label', default='v6')
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    kinds = [('hann', 'Hann sine'), ('gaussian', 'Gaussian sine'),
             ('rectangular', 'Rectangular sine'), ('chirp', 'Hann linear chirp')]
    cycles = [1, 3, 5, 10, 20]
    rows = {r['case_id']: r for r in report['rows']}
    wanted = {f'{kind}-{cycle:02d}' for kind, _ in kinds for cycle in cycles}
    if set(rows) != wanted or report['completed_cases'] != 20 or not report['metrics_independently_recomputed']:
        raise ValueError('A complete, independently audited 20-case matrix is required')
    for key, row in rows.items():
        if sha(args.report.parent / 'metrics' / (key + '.json')) != row['metrics_sha256']:
            raise ValueError('Audited metrics changed')
        if row['duration_us'] != 5000 or row['samples'] != 25001 or row['receivers'] != 8:
            raise ValueError('Inconsistent matrix scope')
    values = np.array([[100 * rows[f'{kind}-{cycle:02d}']['max_linear_vector_relative_l2']
                        for cycle in cycles] for kind, _ in kinds])
    if not np.isfinite(values).all():
        raise ValueError('No placeholder values are allowed')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'text.color': '#28313a', 'axes.labelcolor': '#28313a'})
    fig, ax = plt.subplots(figsize=(11.6, 5.5))
    palette = LinearSegmentedColormap.from_list('workai_difference', ['#f2f6fa', '#86abc9', '#174d78'])
    top = max(160, 20 * np.ceil(values.max() / 20))
    chart = ax.imshow(values, cmap=palette, vmin=0, vmax=top, aspect='auto')
    ax.set_xticks(range(5), cycles)
    ax.set_yticks(range(4), [name for _, name in kinds])
    ax.set_xlabel('Excitation cycles', labelpad=8)
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='both', length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(4):
        for column in range(5):
            value = values[row, column]
            ax.text(column, row, f'{value:.2f}%', ha='center', va='center',
                    color='white' if value > .62 * top else '#28313a', fontsize=13)
    bar = fig.colorbar(chart, ax=ax, fraction=.035, pad=.035, ticks=np.arange(0, top + 1, 40))
    bar.set_label('Vector relative L2 (%)', labelpad=9)
    bar.outline.set_visible(False)
    fig.suptitle('20 complete 5 ms records: maximum vector waveform difference', fontsize=16, y=.965)
    fig.text(.5, .897, f'Neural {args.candidate_label} versus 80-mode COMSOL-sourced linear response; maximum over eight receiver points',
             ha='center', fontsize=10.5)
    failures = int(np.count_nonzero(values > 5))
    fig.text(.19, .105, f'{failures}/20 vector values exceed the 5% criterion. Each record contains 25,001 real samples.', fontsize=10.5)
    fig.text(.19, .062,
             f'Reference checks: mode truncation {report["modal_truncation_passes"]}/20; time integration {report["time_refinement_passes"]}/20; '
             f'mesh refinement {report["mesh_refinement_passes"]}/20. Reference convergence is not certified.', fontsize=10)
    fig.subplots_adjust(left=.19, right=.91, top=.82, bottom=.23)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor='white')
    plt.close(fig)
    record = {'schema': 'workai.comsol_matrix_plot.v1', 'source_report_sha256': sha(args.report),
              'candidate_label': args.candidate_label,
              'candidate_weights_sha256': report['weights_sha256'], 'source_workflow_run_url': report['workflow_run_url'],
              'plotted_measure': 'maximum vector waveform relative L2 over eight receiver points, multiplied by 100',
              'rows': [kind for kind, _ in kinds], 'columns_cycles': cycles, 'values_percent': values.tolist(),
              'plot_sha256': sha(args.output), 'scientific_accuracy_certified': False}
    args.output.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'plot': str(args.output), 'cases': len(rows), 'source_report_sha256': record['source_report_sha256']}))


if __name__ == '__main__':
    main()
