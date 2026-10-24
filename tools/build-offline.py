"""Assemble portable packages from hash-pinned, predownloaded runtimes/wheels."""
import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--cache', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--platform', choices=['linux', 'windows'], required=True)
parser.add_argument('--archive', action='store_true')
parser.add_argument('--example', action='store_true', help='Compute the pinned example using the bundled runtime.')
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
if args.example:
    example = target / 'offline/examples/local-20261024000000000000000000000001'
    if example.exists():
        raise SystemExit('--example requires a source tree without precomputed examples.')
    example.mkdir(parents=True)
    shutil.copyfile(target / 'offline/example-request.json', example / 'request.json')
    (example / 'status.json').write_text(json.dumps({'job_id': example.name, 'name': '内置示例 · 铝板网络预测',
        'phase': 'starting', 'started_at_utc': '2026-10-24T00:00:00+00:00', 'controls_completed': 0}), encoding='utf-8')
    python = target / ('runtime/python.exe' if args.platform == 'windows' else 'runtime/bin/python3')
    env = dict(os.environ, NUMBA_CACHE_DIR=str((args.cache / ('compile-' + args.platform)).resolve()),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    subprocess.run([str(python.resolve()), '-X', 'utf8', '-m', 'offline.worker', '--job-dir', str(example.resolve())],
                   cwd=target, env=env, check=True, timeout=900)
    if json.loads((example / 'status.json').read_text(encoding='utf-8'))['phase'] != 'completed':
        raise SystemExit('Packaged example did not complete.')
# Relocated CPython can regenerate its upstream bytecode on first startup.
# Ship verified source and let Python create caches, outside the immutable manifest.
for cache_file in target.rglob('*.pyc'):
    cache_file.unlink()
rows = [{'path': str(p.relative_to(target)).replace(os.sep, '/'), 'bytes': p.stat().st_size, 'sha256': sha(p)}
        for p in sorted(target.rglob('*')) if p.is_file()]
(target / 'MANIFEST.json').write_text(json.dumps({'schema': 'workai.offline.bundle.v1', 'platform': args.platform,
    'source_commit': os.environ.get('WORKAI_BUILD_COMMIT', 'working-tree'), 'files': rows}, indent=2) + '\n')
if args.archive:
    version = json.loads((target / 'offline/model/model.json').read_text(encoding='utf-8'))['version']
    path = args.output / (target.name + '-' + version + '.zip')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for p in sorted(target.rglob('*')):
            if p.is_file():
                z.write(p, str(Path(target.name) / p.relative_to(target)))
    print(json.dumps({'archive': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}))
else:
    print(target)
