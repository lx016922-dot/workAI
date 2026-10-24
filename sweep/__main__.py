"""python -m sweep plan|run|summarize|dataset ..."""
import argparse
import json
import sys
from pathlib import Path
from .spec import load_spec, plan
from .runner import Campaign
from .summarize import summarize
from .dataset import build_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m sweep', description='Parameter sweeps and training-data campaigns over the plane-strain solver.')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan', help='expand and validate the cases without running'); p.add_argument('--spec', type=Path, required=True)
    r = sub.add_parser('run', help='run (or resume) a campaign'); r.add_argument('--spec', type=Path, required=True); r.add_argument('--output', type=Path, required=True); r.add_argument('--workers', type=int); r.add_argument('--resume', action='store_true')
    s = sub.add_parser('summarize', help='harmonics, fits, velocities, scaling pairs'); s.add_argument('--campaign', type=Path, required=True)
    d = sub.add_parser('dataset', help='stack receiver lines into one NPZ'); d.add_argument('--campaign', type=Path, required=True); d.add_argument('--output', type=Path, required=True); d.add_argument('--line'); d.add_argument('--dtype', default='float32', choices=['float32', 'float64'])
    c = sub.add_parser('converge', help='refinement reruns (space, degree, time) with waveform and f/2f comparisons'); c.add_argument('--campaign', type=Path, required=True); c.add_argument('--cases', help='comma-separated case ids, N for the first N completed cases, or a JSON parameter filter such as {"constitutive": "murnaghan"} (default all); linear controls are added automatically')
    c.add_argument('--levels', help='comma-separated subset of space,degree,time'); c.add_argument('--space-factor', type=float); c.add_argument('--time-factor', type=float); c.add_argument('--workers', type=int)
    args = parser.parse_args(argv)
    if args.command == 'plan':
        spec = load_spec(args.spec); rows = plan(spec)
        print(json.dumps({'cases': len(rows), 'splits': {k: sum(1 for r in rows if r['split'] == k) for k in ('train', 'validation', 'test')}, 'first': [{'case_id': r['case_id'], 'parameters': r['parameters'], 'split': r['split']} for r in rows[:5]]}, ensure_ascii=False, indent=1))
    elif args.command == 'run':
        spec = load_spec(args.spec); campaign = Campaign.create(spec, args.output, resume=args.resume)
        result = campaign.run(spec, workers=args.workers, progress=lambda v: print(json.dumps(v), flush=True)); print(json.dumps(result))
        if result['failed']: sys.exit(2)
    elif args.command == 'summarize':
        summary = summarize(args.campaign); print(json.dumps({'completed_cases': summary['completed_cases'], 'failed_cases': summary['failed_cases'], 'amplitude_scaling_pairs': [(p['case_a'], p['case_2a'], p['all_pass']) for p in summary['amplitude_scaling_pairs']]}, ensure_ascii=False))
    elif args.command == 'dataset':
        meta = build_dataset(args.campaign, args.output, line=args.line, dtype=args.dtype); print(json.dumps({k: meta[k] for k in ('cases', 'splits', 'shapes', 'dataset_sha256')}, ensure_ascii=False))
    else:
        from .spec import validate_spec
        spec = validate_spec(json.loads((args.campaign / 'spec.json').read_text(encoding='utf-8'))); campaign = Campaign.create(spec, args.campaign, resume=True)
        selected = args.cases
        overrides = {'levels': args.levels.split(',') if args.levels else None, 'space_factor': args.space_factor, 'time_factor': args.time_factor}
        result = campaign.converge(spec, selected, overrides, workers=args.workers, progress=lambda v: print(json.dumps(v), flush=True)); print(json.dumps(result))
        if result['failed'] or result['increment_failed']: sys.exit(3)


if __name__ == '__main__': main()
