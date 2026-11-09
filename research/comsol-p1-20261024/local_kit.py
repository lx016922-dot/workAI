"""Run the P1 COMSOL jobs on your own COMSOL installation, then convert the results.

    python research/comsol-p1-20261024/local_kit.py build   --output p1-kit
    python research/comsol-p1-20261024/local_kit.py collect p1-kit/runs --output p1-delivery

`build` makes one folder per job with the corrected model source (serendipity-54e90010/UltraResearch.java:
quartic serendipity hexahedra `shnserp`, as in every completed run; the first archived source c7140d6c built
Lagrange elements), the prepared, hash-pinned inputs, the request/case files the model reads, and run scripts
for Windows and Linux. Job IDs and source hash come from the archived corrected kit P1_SERP_20261102.zip
(revision-manifest.json), so the folders are byte-identical to the ones the completed runs used. The jobs are
the 21 prepared P1 runs and the single-element constitutive self-check, ordered by tier; tiers 0-1 were
completed on 2026-11-09, so `--tiers 2 3 4` builds the remaining 15. Every nonlinear increment must come from
one model source; the linear runs made with the unarchived source d36366 serve as a reproducibility check.

`collect` converts finished job folders into the delivery contract of docs/comsol/comsol-data-request.md
(40-column CSV, 601 rows, 17 significant digits, one metadata JSON per case). It checks that each job
used the archived source and unchanged inputs, the receiver order, the time grid, zero initial state
and the Green strain / J guards. It does not need the signed remote service.
"""
import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REVISION = HERE / 'serendipity-54e90010'
KIT_ARCHIVE_SHA256 = 'e4607e65201cccc64d4c2e57e96eadd7197b0a0c7f7af583729eaa9c38a89ee7'
SOURCE_SHA256 = '54e90010ad0c63181afd7d0ff32348afa7a6b02f49f96cf77fcef48996548344'
LAWS = ['linear', 'stvk', 'murnaghan']
CASES = ['A-f0.40-T1.6', 'B-f0.40-T1.6', 'C-f0.25-T2.0']
# (tier, purpose) per prepared input; jobs run in this order.
TIERS = [
    (0, 'v4-material-check', '本构自检（单个单元，几分钟）'),
    *[(1, f'v4-{c}-murnaghan', '非线性增量：Murnaghan 基准') for c in CASES],
    *[(1, f'v4-{c}-linear', '非线性增量：同一源码的线性基准（与已归档线性结果互为复现检查）') for c in CASES],
    *[(2, f'v4-{c}-stvk', '区分几何非线性与材料非线性：StVK 基准') for c in CASES],
    (3, 'v4-A-f0.40-T1.6-linear-dt2', 'A 工况时间收敛'), (3, 'v4-A-f0.40-T1.6-murnaghan-dt2', 'A 工况时间收敛'),
    (3, 'v4-A-f0.40-T1.6-linear-mesh2', 'A 工况网格收敛'), (3, 'v4-A-f0.40-T1.6-murnaghan-mesh2', 'A 工况网格收敛'),
    *[(4, f'v4-{c}-{law}-{v}', f'{c[0]} 工况{"时间" if v == "dt2" else "网格"}收敛') for c in CASES[1:] for v in ('dt2', 'mesh2') for law in ('linear', 'murnaghan')],
]
OUTPUT_FILE = {'coarse': 'coarse.csv', 'fine': 'fine.csv', 'timecheck': 'timecheck.csv'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prop_value(v):
    return str(int(v)) if isinstance(v, bool) else str(v)


def revision():
    """Corrected source and per-folder job identities of the archived serendipity kit."""
    archive = REVISION / 'P1_SERP_20261102.zip'; source = REVISION / 'UltraResearch.java'
    if digest(archive) != KIT_ARCHIVE_SHA256: raise SystemExit('P1_SERP_20261102.zip does not match its recorded SHA-256')
    if digest(source) != SOURCE_SHA256: raise SystemExit('serendipity-54e90010/UltraResearch.java does not match its recorded SHA-256')
    with zipfile.ZipFile(archive) as z: manifest = json.loads(z.read('revision-manifest.json'))
    plan = json.loads((HERE / 'plan.json').read_text())
    if manifest['source_sha256'] != SOURCE_SHA256 or manifest['original_source_sha256'] != plan['source_sha256']: raise SystemExit('revision manifest does not match the sources')
    return source, {row['folder']: row for row in manifest['job_identity_mappings']}


def build(args):
    source, mapping = revision(); plan = json.loads((HERE / 'plan.json').read_text())
    names = [name for _, name, _ in TIERS]
    if sorted(names) != sorted([plan['material_check'], *plan['runs']]): raise SystemExit('Tier list does not cover plan.json exactly')
    out = Path(args.output); runs = out / 'runs'
    if out.exists() and any(out.iterdir()): raise SystemExit(f'{out} is not empty')
    runs.mkdir(parents=True)
    rows = []
    for order, (tier, name, purpose) in enumerate(TIERS):
        if args.tiers and tier not in args.tiers: continue
        src = HERE / 'inputs' / name; identity = json.loads((src / 'identity.json').read_text())
        scene_text = (src / 'scene.json').read_bytes().decode('utf-8'); params_text = (src / 'parameters.json').read_bytes().decode('utf-8')
        if hashlib.sha256(scene_text.encode()).hexdigest() != identity['scene_sha256'] or hashlib.sha256(params_text.encode()).hexdigest() != identity['parameters_sha256']:
            raise SystemExit(f'{name}: prepared input changed')
        folder = runs / f'{order:02d}-{name[3:]}'; folder.mkdir(); row = mapping[folder.name]
        if (row['old_job_id'], row['scene_sha256'], row['parameters_sha256']) != (identity['job_id'], identity['scene_sha256'], identity['parameters_sha256']):
            raise SystemExit(f'{name}: revision manifest does not match the prepared input')
        identity.update(job_id=row['new_job_id'], source_sha256=row['new_source_sha256'])
        for f in ('scene.json', 'parameters.json'): shutil.copyfile(src / f, folder / f)
        (folder / 'identity.json').write_text(json.dumps(identity, ensure_ascii=True, indent=2) + '\n', encoding='ascii')  # prepare.py format
        shutil.copyfile(source, folder / 'UltraResearch.java')
        request = {'job_id': identity['job_id'], 'kind': 'research_reference', 'source_sha256': identity['source_sha256'], 'scene_sha256': identity['scene_sha256'],
                   'parameters_sha256': identity['parameters_sha256'], 'scene_json': scene_text, 'parameters_json': params_text}
        (folder / 'request.json').write_text(json.dumps(request, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
        if len((folder / 'request.json').read_text()) > 65536: raise SystemExit(f'{name}: request too large for the model reader')
        params = json.loads(params_text)
        lines = [f'{k}={identity[k]}' for k in ('job_id', 'scene_sha256', 'source_sha256', 'parameters_sha256')] + [f'{k}={prop_value(v)}' for k, v in params.items()]
        (folder / 'case.properties').write_text('\n'.join(lines) + '\n', encoding='latin1')
        execution = json.loads(scene_text)['execution']
        rows.append({'folder': folder.name, 'tier': tier, 'case_id': identity['case_id'], 'variant': identity['variant'],
                     'law': ','.join(LAWS[l] for l in identity['laws']), 'h_mm': execution['h_mm'], 'max_step_us': execution['max_step_us'],
                     'output_file': OUTPUT_FILE[execution['level']], 'purpose': purpose})
    (out / 'jobs.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    (out / 'run-all.ps1').write_text(PS1, encoding='utf-8-sig'); (out / 'run-all.sh').write_text(SH, encoding='utf-8')
    (out / 'README.md').write_text(README.replace('{{JOBS}}', '\n'.join(
        f"| {r['folder']} | {r['tier']} | {r['case_id']} | {r['law']} | {r['variant']} | {r['h_mm']} | {r['max_step_us']} | {r['purpose']} |" for r in rows)), encoding='utf-8')
    print(json.dumps({'kit': str(out), 'jobs': len(rows)}, ensure_ascii=False))


def properties(path):
    result = {}
    for line in Path(path).read_text(encoding='latin1').splitlines():
        if not line or line.startswith(('#', '!')) or '=' not in line: continue
        key, value = line.split('=', 1)
        value = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m[1], 16)), value)
        result[key] = re.sub(r'\\([:=\\ ])', r'\1', value)
    return result


def checked(folder):
    """Identity checks for a finished local job; returns (identity, receipt, scene)."""
    identity = json.loads((folder / 'identity.json').read_text())
    if digest(folder / 'UltraResearch.java') != identity['source_sha256']: raise ValueError(f'{folder.name}: model source changed')
    for name in ('scene', 'parameters'):
        if digest(folder / f'{name}.json') != identity[f'{name}_sha256']: raise ValueError(f'{folder.name}: {name}.json changed')
    if not (folder / 'completed.properties').is_file(): raise ValueError(f'{folder.name}: not completed (no completed.properties)')
    receipt = properties(folder / 'completed.properties')
    for key in ('job_id', 'source_sha256', 'scene_sha256', 'parameters_sha256'):
        if receipt.get(key) != identity[key]: raise ValueError(f'{folder.name}: receipt {key} does not match the input')
    return identity, receipt, json.loads((folder / 'scene.json').read_text())


def read_table(path):
    lines = [l for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]
    head = lines[0].split(','); return [dict(zip(head, l.split(','))) for l in lines[1:]]


def material_check(folder):
    import numpy as np
    identity, receipt, scene = checked(folder)
    if receipt.get('analysis') != 'material_check': raise ValueError('material-check folder is not a material check')
    rows = read_table(folder / 'coarse.csv')
    if len(rows) != 27: raise ValueError('Three 3x3 stress tensors required')
    actual = np.array([float(r['S_MPa']) for r in rows]).reshape(3, 3, 3); expected = np.array([float(r['S_expected_MPa']) for r in rows]).reshape(3, 3, 3)
    error = float(np.max(abs(actual - expected)))
    if error > 1e-5: raise ValueError(f'Constitutive self-check failed (max error {error:.3g} MPa)')
    return {'uniaxial_E11_1e-3_S_MPa': actual[0].tolist(), 'shear_E12_1e-3_S_MPa': actual[1].tolist(), 'general_S_MPa': actual[2].tolist(),
            'max_absolute_error_MPa': error, 'job_id': identity['job_id'], 'source_sha256': identity['source_sha256'],
            'comsol_version': receipt.get('comsol_version'), 'passed': True}


def step_counts(folder, laws):
    log = folder / 'solver.log'
    if not log.is_file(): return {law: {'time_steps': None, 'step_count_source': 'solver.log not provided'} for law in laws}
    raw = log.read_bytes(); text = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else raw.decode('utf-8', 'replace')
    sections = re.split(r'Step\s+Time\s+Stepsize[^\n]*\n', text)[1:]
    out = {}
    for i, law in enumerate(laws):
        steps = [(int(m[1]), None if m[3] == '-' else float(m[3])) for m in re.finditer(r'^\s*(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+|-)', sections[i], re.M)] if i < len(sections) else []
        out[law] = {'time_steps': max((s[0] for s in steps), default=None), 'maximum_logged_step_us': max((s[1] or 0 for s in steps), default=0) * 1e6 or None,
                    'step_count_source': 'COMSOL batch log' if steps else 'no step table found in solver.log'}
    return out


def collect(args):
    import numpy as np
    runs = sorted(p for p in Path(args.runs).iterdir() if p.is_dir()); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    header = (REPO / 'docs/comsol/header-template.csv').read_text().strip()
    checks = [p for p in runs if 'material-check' in p.name and (p / 'completed.properties').is_file()]
    if not checks: raise SystemExit('The constitutive self-check (00-material-check) must be completed first.')
    check = material_check(checks[0]); (out / 'material-check.json').write_text(json.dumps(check, indent=1) + '\n')
    report = {'collected': [], 'skipped': []}
    for folder in runs:
        if folder == checks[0]: continue
        try:
            identity, receipt, reference = checked(folder)
        except ValueError as error:
            report['skipped'].append(str(error)); continue
        if identity['source_sha256'] != check['source_sha256']: raise SystemExit(f'{folder.name}: self-check covers a different source')
        cfg, scene = reference['execution'], reference['scene']
        raw = read_table(folder / OUTPUT_FILE[cfg['level']]); guards = read_table(folder / OUTPUT_FILE[cfg['level']].replace('.csv', '.field.csv'))
        if len(raw) != len(identity['laws']) * 13 * 601 or len(guards) != len(identity['laws']) * 601: raise SystemExit(f'{folder.name}: incomplete output')
        steps = step_counts(folder, identity['laws'])
        meta_path = out / f"{identity['case_id']}.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
            'case_id': identity['case_id'], 'comsol_version': receipt.get('comsol_version'),
            'source_provenance': {'mode': 'local_run_of_archived_source', 'source_sha256': identity['source_sha256']},
            'element': {'shape': 'hexahedral, swept', 'order': 'quartic', 'degree': 4, 'discretization': 'linear Hooke or total Lagrangian P=F*S'},
            'integration_order': cfg['quadrature_order'],
            'time_stepping': {'method': 'generalized-alpha', 'rho_infinity': 1.0, 'output_step_us': 0.05, 'steps': 'strict'},
            'tolerances': {'time_relative': cfg['relative_tolerance'], 'absolute_m': cfg['absolute_tolerance_m'], 'newton': 'tolerance factor 0.001'},
            'load': {'type': 'nominal traction per reference area (dead load, fixed direction)', 'sources': scene['sources']},
            'absorbing': {'method': 'body load -rho0*alpha(x)*du/dt, all three components', 'alpha_max_per_s': scene['alpha_max_per_s'], 'ramp_power': 3, 'faces': scene['absorbing_faces']},
            'geometry': scene['geometry'], 'material': scene['material'], 'receivers': scene['receivers'], 'material_check': check, 'runs': {}}
        for law in identity['laws']:
            rows = [r for r in raw if int(float(r['law'])) == law]
            if [int(float(r['rx'])) for r in rows] != list(range(13)) * 601: raise SystemExit(f'{folder.name}: receiver order changed')
            t = np.array([float(r['time_s']) for r in rows]).reshape(601, 13)
            if not np.allclose(t, np.arange(601)[:, None] * 0.05e-6, atol=1e-15, rtol=0): raise SystemExit(f'{folder.name}: time grid mismatch')
            U = np.array([[float(r['u_m']), float(r['v_m']), float(r['w_m'])] for r in rows]).reshape(601, 39)
            if not np.isfinite(U).all() or np.max(abs(U[0])) > 1e-18: raise SystemExit(f'{folder.name}: non-finite or non-zero initial displacement')
            g = [r for r in guards if int(float(r['law'])) == law]; strain = max(float(r['max_green_strain']) for r in g); J = min(float(r['min_J']) for r in g)
            if strain > 1e-3 or J <= 0: raise SystemExit(f'{folder.name}: Green strain / J guard failed')
            name = LAWS[law] + ('' if identity['variant'] == 'base' else '_' + identity['variant'])
            path = out / f"{identity['case_id']}_{name}.csv"
            np.savetxt(path, np.column_stack([np.arange(601) * 0.05, U]), delimiter=',', header=header, comments='', fmt='%.17e')
            h = {d: float(receipt[f'h_axis_{i}_mm']) for i, d in enumerate('xyz') if f'h_axis_{i}_mm' in receipt}
            meta['runs'][name] = {'job_id': identity['job_id'], 'max_step_us': cfg['max_step_us'], 'mesh_requested_h_mm': cfg['h_mm'], 'max_element_size_mm': h,
                                  'elements': int(receipt['volume_hexahedra']) if 'volume_hexahedra' in receipt else None, 'dofs': int(receipt['dofs']) if 'dofs' in receipt else None,
                                  'solve_time_s': float(receipt[f'law{law}_solve_seconds']) if f'law{law}_solve_seconds' in receipt else None,
                                  'max_green_strain': strain, 'min_J': J, 'csv_sha256': digest(path), **steps[law]}
            if identity['variant'] == 'base':
                for key in ('elements', 'dofs'): meta[key] = meta['runs'][name][key]
                meta['max_element_size_mm'] = h
            report['collected'].append(path.name)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=1))


PS1 = r'''# Runs every job folder under runs\ in order, skipping finished ones. Usage (PowerShell):
#   .\run-all.ps1 -ComsolBin "C:\Program Files\COMSOL\COMSOL64\Multiphysics\bin\win64" [-Tier 0,1] [-Cores 8]
param([Parameter(Mandatory=$true)][string]$ComsolBin, [int[]]$Tier, [int]$Cores = 0)
$ErrorActionPreference = 'Stop'
$jobs = Get-Content (Join-Path $PSScriptRoot 'jobs.json') -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($job in $jobs) {
  if ($Tier -and -not ($Tier -contains [int]$job.tier)) { continue }
  $dir = Join-Path (Join-Path $PSScriptRoot 'runs') $job.folder
  if (Test-Path (Join-Path $dir 'completed.properties')) { Write-Host "done    $($job.folder)"; continue }
  Write-Host "running $($job.folder)  ($($job.purpose))"
  Push-Location $dir
  try {
    & (Join-Path $ComsolBin 'comsolcompile.exe') 'UltraResearch.java' *> 'compile.log'
    if (-not (Test-Path 'UltraResearch.class')) { throw "compile failed, see $dir\compile.log" }
    $extra = @(); if ($Cores -gt 0) { $extra = @('-np', "$Cores") }
    & (Join-Path $ComsolBin 'comsolbatch.exe') @extra -inputfile 'UltraResearch.class' -batchlog 'solver.log'
    if (-not (Test-Path 'completed.properties')) { throw "job did not complete, see $dir\solver.log" }
  } finally { Pop-Location }
}
'''

SH = r'''#!/bin/sh
# Runs every job folder under runs/ in order, skipping finished ones.
#   COMSOL_BIN=/usr/local/comsol64/multiphysics/bin sh run-all.sh [tier ...]
set -e
here=$(cd "$(dirname "$0")" && pwd)
: "${COMSOL_BIN:?set COMSOL_BIN to the COMSOL bin directory}"
for dir in "$here"/runs/*/; do
  name=$(basename "$dir")
  tier=$(python3 -c "import json,sys;print(next(j['tier'] for j in json.load(open('$here/jobs.json')) if j['folder']=='$name'))")
  if [ $# -gt 0 ] && ! echo " $* " | grep -q " $tier "; then continue; fi
  if [ -f "$dir/completed.properties" ]; then echo "done    $name"; continue; fi
  echo "running $name"
  (cd "$dir" && "$COMSOL_BIN/comsol" compile UltraResearch.java > compile.log 2>&1 \
     && "$COMSOL_BIN/comsol" batch -inputfile UltraResearch.class -batchlog solver.log)
  [ -f "$dir/completed.properties" ] || { echo "job $name did not complete; see $dir/solver.log"; exit 1; }
done
'''

README = '''# P1 剩余 COMSOL 运行包

每个 `runs/` 子文件夹是一个独立作业，包含修正版模型源码 `UltraResearch.java`（SHA-256 `54e90010…`，四阶 serendipity 单元，与已完成的全部运行相同），以及按哈希锁定的输入。层级 0–1 已于 2026-11-09 完成。模型完全由源码和输入决定，不需要在图形界面里建模。

## 运行

Windows（PowerShell）：

```powershell
.\\run-all.ps1 -ComsolBin "C:\\Program Files\\COMSOL\\COMSOL64\\Multiphysics\\bin\\win64" -Tier 0,1
```

Linux：

```sh
COMSOL_BIN=/usr/local/comsol64/multiphysics/bin sh run-all.sh 0 1
```

- 按层级运行，先跑 0 和 1。已完成的作业（有 `completed.properties`）会自动跳过，中断后重新执行同一命令即可续跑。
- 每个作业在自己的文件夹里编译并运行，生成 `completed.properties`、结果 CSV、`*.field.csv`（应变与 J 检查）、`solver.log` 和 `.mph`。
- 多台机器或多核时可以把不同层级或不同作业分开跑；`-Cores` 限制每个作业的核数。
- **不要修改任何作业文件夹里的 `UltraResearch.java`、`scene.json`、`parameters.json`。** 改动后哈希不符，结果无法收集。

## 作业清单

| 文件夹 | 层级 | 工况 | 本构 | 变体 | h (mm) | 最大步长 (μs) | 用途 |
|---|---|---|---|---|---|---|---|
{{JOBS}}

- 层级 0：本构自检，几分钟。必须通过，否则不要继续。
- 层级 1：三个工况的 Murnaghan 与同一源码的线性运行，用来算非线性增量。已归档的线性结果来自没有归档的另一份源码，所以线性要用同一源码重算一次；它与旧结果的差异同时就是复现检查。
- 层级 2：StVK，区分几何非线性和材料非线性。
- 层级 3：A 工况的时间步减半（dt2）和网格加密（mesh2），检查 COMSOL 自身是否收敛。
- 层级 4：B、C 工况的收敛检查。层级 3 显示收敛良好时可以不跑。

已归档的线性运行在原服务器上每次约 24–30 小时（2.2–3.1 万自由度）。非线性运行估计约为线性的 1.5–3 倍，dt2 约 2 倍，mesh2 约 4–6 倍；以实测为准。

## 跑完以后

把整个 `runs/` 文件夹打包发回即可（`.mph` 可以不带）。也可以自己转换并检查：

```sh
python research/comsol-p1-20261024/local_kit.py collect runs --output p1-delivery
python -m coarsegraph.comsol check p1-delivery --partial
```
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0]); sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build'); b.add_argument('--output', required=True); b.add_argument('--tiers', type=int, nargs='*', default=[])
    c = sub.add_parser('collect'); c.add_argument('runs'); c.add_argument('--output', required=True)
    args = parser.parse_args(); build(args) if args.command == 'build' else collect(args)
