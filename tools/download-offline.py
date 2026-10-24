"""Fetch pinned package build inputs; only the build stage requires networking."""
import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--cache', type=Path, required=True)
parser.add_argument('--platform', choices=['linux', 'windows'], required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
rows = [r for r in json.loads((root / 'offline/dependencies.json').read_text())['files'] if r['platform'] == args.platform]
for row in rows:
    target = args.cache / row['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        if row['path'].endswith('.whl'):
            package, version = target.name.split('-')[:2]
            with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json', timeout=60) as response:
                meta = json.load(response)
            url = next(a['url'] for a in meta['urls'] if a['filename'] == target.name and a['digests']['sha256'] == row['sha256'])
        else:
            url = row['source']
        temp = target.with_suffix(target.suffix + '.part')
        with urllib.request.urlopen(url, timeout=90) as response, temp.open('wb') as out:
            import shutil
            shutil.copyfileobj(response, out)
        temp.replace(target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != row['sha256']:
        raise SystemExit('SHA-256 mismatch: ' + target.name)
    print('verified', target.name, flush=True)
