"""Assemble the WorkAI desktop release for Windows: WorkAI.exe (desktop window) + the portable computation package.

Inputs: the hash-pinned runtime/wheel cache of tools/download-offline.py and the built launcher (dotnet build of
desktop/WorkAI.csproj). Output: <output>/WorkAI-<version>-windows-x64/ and, with --archive, a .zip of that folder.
The portable package is produced by tools/build-offline.py unchanged; this script adds the launcher, the user guide
and the version stamp, and rewrites MANIFEST.json so verify-package.py covers every shipped file.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--cache', type=Path, required=True)
parser.add_argument('--launcher', type=Path, required=True, help='directory with WorkAI.exe and its WebView2 libraries')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--archive', action='store_true')
args = parser.parse_args()

version = json.loads((REPO / 'offline/version.json').read_text(encoding='utf-8'))['version']
project = (REPO / 'desktop/WorkAI.csproj').read_text(encoding='utf-8')
for tag, expected in (('Version', version), ('FileVersion', version + '.0'), ('AssemblyVersion', version + '.0')):
    if f'<{tag}>{expected}</{tag}>' not in project:
        raise SystemExit(f'desktop/WorkAI.csproj <{tag}> must be {expected} (offline/version.json is {version}).')
name = f'WorkAI-{version}-windows-x64'
target = args.output / name
if target.exists():
    raise SystemExit('Use a new output directory; existing builds are preserved.')
if not (args.launcher / 'WorkAI.exe').is_file():
    raise SystemExit('Launcher not found: ' + str(args.launcher / 'WorkAI.exe'))
stage = args.output / 'stage'
subprocess.run([sys.executable, str(REPO / 'tools/build-offline.py'), '--cache', str(args.cache), '--output', str(stage), '--platform', 'windows'], check=True)
(stage / 'WorkAI-Offline-windows-x64').rename(target)
shutil.rmtree(stage)

# launcher: the exe, its config and the WebView2 libraries (including runtimes\win-x64\native when present)
for item in sorted(args.launcher.iterdir()):
    if item.suffix.lower() in ('.pdb', '.xml'):
        continue
    if item.is_dir():
        shutil.copytree(item, target / item.name)
    else:
        shutil.copyfile(item, target / item.name)

# browser fallback keeps the old start script under a clearer name; the desktop guide replaces the old readme
(target / 'Start-WorkAI.bat').rename(target / 'Start-WorkAI-Browser.bat')
(target / 'README-OFFLINE.txt').unlink()
for linux_only in ('start-workai.sh',):
    (target / linux_only).unlink(missing_ok=True)
# Windows text: UTF-8 with BOM and CRLF line ends, so every Notepad version shows the Chinese guide correctly
guide = (REPO / 'desktop/README-DESKTOP.txt').read_text(encoding='utf-8').replace('\r\n', '\n').replace('\n', '\r\n')
(target / '使用说明.txt').write_bytes(guide.encode('utf-8-sig'))
stamp = json.loads((REPO / 'offline/version.json').read_text(encoding='utf-8'))
commit = os.environ.get('WORKAI_BUILD_COMMIT', 'working-tree')
stamp['build'] = f"{commit[:12]} · {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
(target / 'offline/version.json').write_text(json.dumps(stamp, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(target / 'MANIFEST.json').unlink()
rows = [{'path': str(p.relative_to(target)).replace(os.sep, '/'), 'bytes': p.stat().st_size, 'sha256': sha(p)}
        for p in sorted(target.rglob('*')) if p.is_file()]
(target / 'MANIFEST.json').write_text(json.dumps({'schema': 'workai.offline.bundle.v1', 'platform': 'windows', 'product': 'WorkAI desktop',
    'version': version, 'source_commit': commit, 'files': rows}, indent=2) + '\n')
summary = {'folder': str(target), 'files': len(rows), 'bytes': sum(r['bytes'] for r in rows)}
if args.archive:
    path = args.output / (name + '.zip')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(target.rglob('*')):
            if p.is_file():
                z.write(p, str(Path(name) / p.relative_to(target)))
    summary.update(archive=str(path), archive_bytes=path.stat().st_size, sha256=sha(path))
print(json.dumps(summary, ensure_ascii=False))
