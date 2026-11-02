"""Package only the hash-verified numerical evidence from the frozen CI run."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
report = json.loads((repo / 'research/long-validation/summary.json').read_text())
assert report['completed_cases'] == report['required_cases'] == 20
assert len({r['case_id'] for r in report['rows']}) == 20
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
args.output.mkdir(parents=True, exist_ok=True)
archive = args.output / 'WorkAI-5ms-validation-evidence-0.3.0.zip'
entries = []
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as output:
    for row in report['rows']:
        case_id = row['case_id']
        directory = args.input / ('long-' + case_id)
        assert sha(directory / 'plan.json') == report['plan_sha256']
        metrics = directory / case_id / 'metrics.json'
        assert sha(metrics) == row['metrics_sha256']
        data = json.loads(metrics.read_text())
        assert data['frame_count'] == 25001 and data['duration_us'] == 5000
        selected = [directory / 'plan.json', metrics]
        for block in data['probe_blocks']:
            path = directory / case_id / block['file']
            assert path.name == block['file'] and path.name.startswith('probes-')
            assert sha(path) == block['sha256']
            selected.append(path)
        # These are simulation state/field-error accumulators, not training checkpoints.
        selected.extend(directory / case_id / name for name in ['state.npz', 'transfer.npz', 'transfer.json', 'progress.json'])
        for path in selected:
            name = str(Path('cases') / case_id / path.relative_to(directory))
            output.write(path, name)
            entries.append({'path': name, 'bytes': path.stat().st_size, 'sha256': sha(path)})
    for path in sorted((repo / 'research/long-validation').rglob('*')):
        if path.is_file():
            name = str(Path('report') / path.relative_to(repo / 'research/long-validation'))
            output.write(path, name)
            entries.append({'path': name, 'bytes': path.stat().st_size, 'sha256': sha(path)})
    output.writestr('MANIFEST.json', json.dumps({'schema':'workai.validation_evidence.v1',
        'source_workflow_run_url': report['workflow_run_url'], 'completed_cases': 20,
        'scientific_accuracy_certified': False, 'files': entries}, indent=2) + '\n')
with archive.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
(args.output / 'SHA256SUMS-validation.txt').write_text(digest + '  ' + archive.name + '\n')
print(json.dumps({'archive': archive.name, 'bytes': archive.stat().st_size,
                  'sha256': digest, 'verified_files': len(entries)}))
