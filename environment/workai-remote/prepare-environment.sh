#!/usr/bin/env bash
set -euo pipefail
cd /workspace
workai_root=/workspace/workai-remote
mkdir -p "$workai_root/downloads" "$workai_root/tools"

# The user-supplied ZIP is retained in the environment snapshot. This is a
# server update package; never execute its updater on this Linux machine.
python3 - <<'PY'
from pathlib import Path, PurePosixPath
import hashlib, json, stat, zipfile
source = Path('/workspace/attachments/7f0b3536-f377-4521-8dbc-c67e0eaa06f0/volume-protocol-update-20261014-v2.zip')
if hashlib.sha256(source.read_bytes()).hexdigest() != '59280383b5713663a56ad383fbf76e9a7f06faba5feb4c35c79f9dc929f6ca38':
    raise SystemExit('Uploaded archive hash mismatch; preserve it and investigate.')
target = Path('/workspace/workai-remote/volume-protocol-update-20261014-v2')
with zipfile.ZipFile(source) as archive:
    entries = []
    for info in archive.infolist():
        name = PurePosixPath(info.filename)
        if name.is_absolute() or '..' in name.parts or stat.S_ISLNK(info.external_attr >> 16):
            raise SystemExit('Unsafe archive member')
        path = target.joinpath(*name.parts)
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise SystemExit('Refusing to follow a destination symlink')
        if info.is_dir():
            continue
        data = archive.read(info)
        if path.exists() and path.read_bytes() != data:
            raise SystemExit('Preserving an existing modified file: ' + str(path))
        entries.append((path, data))
    for path, data in entries:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
manifest = json.loads((target / 'volume-protocol-payload/manifest.json').read_text())
for name, expected in manifest['payload_sha256'].items():
    if hashlib.sha256((target / 'volume-protocol-payload' / name).read_bytes()).hexdigest() != expected:
        raise SystemExit('Payload checksum mismatch: ' + name)
print('PASS: uploaded ZIP and all three payload hashes verified.')
PY

workai_archive="$workai_root/downloads/powershell-7.6.7-linux-x64.tar.gz"
if [ ! -f "$workai_archive" ]; then
    curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
        --max-time 180 --output "$workai_archive" \
        https://github.com/PowerShell/PowerShell/releases/download/v7.6.7/powershell-7.6.7-linux-x64.tar.gz
fi
python3 - <<'PY'
from pathlib import Path
import hashlib, tarfile
root = Path('/workspace/workai-remote')
archive = root / 'downloads/powershell-7.6.7-linux-x64.tar.gz'
# Verified against Microsoft's official v7.6.7 hashes.sha256 over HTTPS.
expected = '6896f9e3a6b0a354bf470835a96da76d2f801cd50dcf047ad7c4563bf93bf431'
with archive.open('rb') as stream:
    if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
        raise SystemExit('PowerShell archive checksum mismatch')
destination = root / 'tools/powershell-7.6.7'
marker = destination / '.workai-archive-sha256'
if destination.exists():
    if not marker.is_file() or marker.read_text().strip() != expected:
        raise SystemExit('Preserving existing unrecognized PowerShell installation')
else:
    destination.mkdir(parents=True)
    with tarfile.open(archive) as bundle:
        bundle.extractall(destination, filter='data')
    (destination / 'pwsh').chmod(0o755)
    marker.write_text(expected + '\n')
print('PASS: verified portable PowerShell installation is available.')
PY

# The wrapper and validation helper are retained outside the repository.
# XDG paths in the wrapper keep PowerShell caches in the writable workspace.
test -x "$workai_root/bin/workai-pwsh"
test -f "$workai_root/check-protocol.ps1"
"$workai_root/bin/workai-pwsh" -NoLogo -NoProfile -NonInteractive \
    -File "$workai_root/check-protocol.ps1"

# Retained recovery inputs and client identity are part of this environment.
# Never silently rotate the key after the Windows server has been paired.
python3 - <<'PY'
from pathlib import Path
import hashlib, stat
root = Path('/workspace/workai-remote')
for name, expected in {
    'connection/server-original.json': '096bfc3716351062753cfe3845f3a0bf17f02273f683ec8f00228324418dcb6b',
    'connection/server-paired-20261022.json': '883032f1ee52c796169a21835a3d039f631f1d7d58ca7df8e8c246c2b7bdcc2b',
    'connection/Http.Common.ps1': '091cfcad0077b58903a4e9117fa85d8f56b1d3950350f0b974cfb0acddda9b41',
}.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise SystemExit('Recovered input checksum changed: ' + name)
private = root / 'private/client-private.xml'
if not private.is_file() or private.is_symlink():
    raise SystemExit('Retained client private key is missing; recover the existing identity instead of rotating it.')
if stat.S_IMODE(private.stat().st_mode) != 0o600 or stat.S_IMODE(private.parent.stat().st_mode) != 0o700:
    raise SystemExit('Client identity permissions must be 0700 for its directory and 0600 for its file.')
PY
umask 077
"$workai_root/bin/workai-pwsh" -NoLogo -NoProfile -NonInteractive \
    -File "$workai_root/New-WorkAIIdentity.ps1"
"$workai_root/bin/workai-pwsh" -NoLogo -NoProfile -NonInteractive \
    -File "$workai_root/check-client.ps1"
test -x "$workai_root/bin/workai"
test -f "$workai_root/Get-WorkAIArtifacts.ps1"
