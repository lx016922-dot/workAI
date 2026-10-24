"""Verify immutable shipped files; generated data/cache are not package inputs."""
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
failures = []
for row in manifest['files']:
    p = root / row['path']
    if not p.is_file() or p.stat().st_size != row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
        failures.append(row['path'])
print(json.dumps({'passed': not failures, 'checked_files': len(manifest['files']), 'failures': failures}, ensure_ascii=False))
raise SystemExit(bool(failures))
