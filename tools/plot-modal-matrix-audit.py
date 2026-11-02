"""Plot audited complete 5 ms linear diagnostics, without a nonlinear certificate."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.summary.read_text())
    rows = report['rows']
    if len(rows) != 20 or report['nonlinear_accuracy_certified']:
        raise ValueError('Expected twenty uncertified linear diagnostic cases')
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'cases.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True, sharey=True)
    cycles = [1, 3, 5, 10, 20]
    for axis, signal in zip(axes.flat, ['hann', 'gaussian', 'rectangular', 'chirp']):
        selected = [next(r for r in rows if r['case_id'] == f'{signal}-{c:02d}') for c in cycles]
        axis.plot(cycles, [r['maximum_vector_error_percent'] for r in selected], 'o-', color='#167c80', label='Vector waveform')
        axis.plot(cycles, [r['maximum_complete_error_percent'] for r in selected], 's-', color='#b04b39', label='All component/band checks')
        unresolved = [r for r in selected if not all(r[k] for k in ['reference_modal_truncation_passed', 'reference_time_refinement_passed', 'reference_mesh_refinement_passed'])]
        axis.scatter([int(r['case_id'].split('-')[1]) for r in unresolved],
                     [r['maximum_complete_error_percent'] for r in unresolved],
                     marker='x', s=90, linewidths=2, color='#202b36', zorder=4, label='Reference convergence unresolved')
        axis.axhline(5, color='#697585', linestyle='--', linewidth=1, label='5% threshold')
        axis.set_title(signal.title()); axis.set_xticks(cycles); axis.set_ylim(0, 38)
        axis.grid(axis='y', color='#e3e7eb'); axis.spines[['top', 'right']].set_visible(False)
    for axis in axes[-1]: axis.set_xlabel('Excitation cycles')
    for axis in axes[:, 0]: axis.set_ylabel('Relative L2 error (%)')
    fig.suptitle('Complete 5 ms candidate records: linear COMSOL-derived reference checks', fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(.5, .035))
    fig.text(.5, .012, '20 cases; 10 pass all linear checks. Nonlinear 5 ms COMSOL accuracy is not certified.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .13, 1, .95])
    fig.savefig(args.output/'matrix.png', dpi=170)
    fig.savefig(args.output/'matrix.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
