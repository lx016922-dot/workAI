"""Assemble portable packages from hash-pinned, predownloaded runtimes/wheels."""
import argparse
import hashlib
import json
import os
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--cache', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--platform', choices=['linux', 'windows'], required=True)
parser.add_argument('--archive', action='store_true')
args = parser.parse_args()
target = args.output / ('WorkAI-Offline-' + args.platform + '-x64')
if target.exists():
    raise SystemExit('Use a new output directory; existing builds are preserved.')
target.mkdir(parents=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
lock = json.loads((REPO / 'offline/dependencies.json').read_text())
for row in lock['files']:
    if row['platform'] == args.platform:
        p = args.cache / row['path']
        if not p.is_file() or sha(p) != row['sha256']:
            raise SystemExit('Missing or mismatched build input: ' + row['path'])

# Runtime allowlist: neither credentials nor the historical cloud state enter a package.
files = {'graphwave': ['__init__.py', 'inference.py', 'predict.py'],
         'spectral3d': ['__init__.py', 'mesh.py', 'curved_mesh.py', 'result.py'],
         'solid': ['__init__.py', 'schema.py']}
for module, names in files.items():
    (target / module).mkdir()
    for name in names:
        shutil.copyfile(REPO / module / name, target / module / name)
shutil.copytree(REPO / 'offline', target / 'offline', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.nbc', '*.nbi'))
for name in ['README-OFFLINE.txt', 'Start-WorkAI.bat', 'start-workai.sh', 'verify-package.py']:
    shutil.copyfile(REPO / 'offline' / name, target / name)
(target / 'start-workai.sh').chmod(0o755)
(target / 'training').mkdir()
for name in ['model.py', 'train.py', 'test_network.py']:
    shutil.copyfile(REPO / 'graphwave' / name, target / 'training' / name)
shutil.copyfile(REPO / 'offline/model/training.json', target / 'training/training.json')

if args.platform == 'windows':
    runtime = target / 'runtime'
    runtime.mkdir()
    with zipfile.ZipFile(args.cache / 'downloads/windows-python.zip') as z:
        z.extractall(runtime)
    site = runtime / 'Lib/site-packages'
    (runtime / 'python312._pth').write_text('python312.zip\n.\n..\nLib/site-packages\nimport site\n', encoding='utf-8')
else:
    with tarfile.open(args.cache / 'downloads/linux-python.tar.gz') as z:
        z.extractall(target, filter='data')
    (target / 'python').rename(target / 'runtime')
    site = target / 'runtime/lib/python3.12/site-packages'
site.mkdir(parents=True, exist_ok=True)
for wheel in sorted((args.cache / 'wheels' / args.platform).glob('*.whl')):
    with zipfile.ZipFile(wheel) as z:
        for item in z.infolist():
            if item.is_dir():
                continue
            parts = Path(item.filename).parts
            if parts[0].endswith('.data'):
                scheme = {'purelib': site, 'platlib': site,
                          'scripts': target / 'runtime' / ('Scripts' if args.platform == 'windows' else 'bin'),
                          'data': target / 'runtime', 'headers': target / 'runtime/include'}
                if parts[1] not in scheme:
                    raise SystemExit('Unsupported wheel data scheme: ' + item.filename)
                dest = scheme[parts[1]].joinpath(*parts[2:])
            else:
                dest = site.joinpath(*parts)
            if not dest.resolve().is_relative_to(target.resolve()):
                raise SystemExit('Unsafe wheel path')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(item))
# Relocated CPython can regenerate its upstream bytecode on first startup.
# Ship verified source and let Python create caches, outside the immutable manifest.
if args.platform == 'linux':
    for cache_file in (target / 'runtime').rglob('*.pyc'):
        cache_file.unlink()
rows = [{'path': str(p.relative_to(target)).replace(os.sep, '/'), 'bytes': p.stat().st_size, 'sha256': sha(p)}
        for p in sorted(target.rglob('*')) if p.is_file()]
(target / 'MANIFEST.json').write_text(json.dumps({'schema': 'workai.offline.bundle.v1', 'platform': args.platform,
    'source_commit': os.environ.get('WORKAI_BUILD_COMMIT', 'working-tree'), 'files': rows}, indent=2) + '\n')
if args.archive:
    path = args.output / (target.name + '-0.2.0-recovery.1.zip')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for p in sorted(target.rglob('*')):
            if p.is_file():
                z.write(p, str(Path(target.name) / p.relative_to(target)))
    print(json.dumps({'archive': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}))
else:
    print(target)
