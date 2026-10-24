"""Resumable, parallel execution of planned cases in isolated solver processes."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from solid.schema import require
from .spec import SCHEMA, plan

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT_MODULES = ('physics', 'spectral2d', 'spectral3d', 'solid', 'dispersion', 'sweep')


def now(): return datetime.now(timezone.utc).isoformat()
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8'); temporary.replace(path)


def source_hashes():
    return {f'{module}/{p.name}': digest(p) for module in SNAPSHOT_MODULES for p in sorted((REPO / module).glob('*.py'))}


class Campaign:
    def __init__(self, directory):
        self.directory = Path(directory); self.cases = self.directory / 'cases'; self.lock = threading.Lock(); self.manifest = None

    @classmethod
    def create(cls, spec, directory, resume=False):
        directory = Path(directory); campaign = cls(directory)
        if directory.exists() and not resume: raise FileExistsError('输出目录已存在；使用 --resume 继续，或换一个目录。')
        if directory.exists():
            campaign.manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
            require(campaign.manifest['spec_sha256'] == _spec_sha(spec), '继续运行时扫描规格必须与原 manifest 相同。')
            return campaign
        directory.mkdir(parents=True); campaign.cases.mkdir()
        rows = plan(spec)
        atomic(directory / 'spec.json', spec)
        campaign.manifest = {'schema': 'workai.sweep.campaign.v1', 'name': spec['name'], 'created_at_utc': now(), 'spec_sha256': _spec_sha(spec), 'degree': spec['degree'],
                             'source_sha256': source_hashes(), 'engine': 'workai_spectral_plane_strain_2d', 'accepted_for_solver_routing': False,
                             'cases': [{'case_id': r['case_id'], 'parameters': r['parameters'], 'split': r['split'], 'phase': 'planned'} for r in rows]}
        for r in rows:
            case_dir = campaign.cases / r['case_id']; case_dir.mkdir()
            atomic(case_dir / 'scene.json', r['scene']); atomic(case_dir / 'settings.json', {'degree': spec['degree']})
        campaign.save(); return campaign

    def save(self):
        with self.lock: self.manifest['updated_at_utc'] = now(); atomic(self.directory / 'manifest.json', self.manifest)

    def entry(self, cid): return next(c for c in self.manifest['cases'] if c['case_id'] == cid)

    def completed(self, cid):
        """A case is complete only when its result.json matches the manifest hash."""
        entry = self.entry(cid); result = self.cases / cid / 'out' / 'result.json'
        return entry.get('phase') == 'completed' and result.is_file() and digest(result) == entry.get('result_sha256')

    def run_case(self, cid, spec):
        case_dir = self.cases / cid; out = case_dir / 'out'; started = time.monotonic()
        if out.exists(): shutil.rmtree(out)
        with self.lock: self.entry(cid).update(phase='running', started_at_utc=now()); 
        self.save()
        command = [sys.executable, '-m', 'spectral2d.run', '--scene', str(case_dir / 'scene.json'), '--output', str(out), '--degree', str(spec['degree'])]
        if spec['outputs']['fields']: command.append('--fields')
        if spec['outputs']['csv']: command.append('--csv')
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUTF8='1')
        with (case_dir / 'run.log').open('w', encoding='utf-8') as log:
            try: code = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=spec['budget']['seconds_per_case']).returncode
            except subprocess.TimeoutExpired: code = 'timeout'
        elapsed = time.monotonic() - started; entry = self.entry(cid)
        if code == 0 and (out / 'result.json').is_file():
            record = json.loads((out / 'result.json').read_text(encoding='utf-8')); d = record['diagnostics']
            with self.lock: entry.update(phase='completed', result_sha256=digest(out / 'result.json'), files={k: v for k, v in record['files'].items()}, elapsed_s=elapsed, completed_at_utc=now(),
                                         nodes=d['nodes'], internal_steps=d['internal_steps'], energy_balance_relative=d['external_work_balance_relative'], max_strain=d['max_strain'])
        else:
            tail = (case_dir / 'run.log').read_text(encoding='utf-8', errors='replace').strip().split('\n')[-1:] if (case_dir / 'run.log').exists() else []
            with self.lock: entry.update(phase='failed', elapsed_s=elapsed, error=f'exit {code}: ' + ' '.join(tail)[:400])
        self.save(); return entry['phase']

    def run(self, spec, workers=None, progress=None):
        workers = workers or spec['budget']['workers']; pending = [c['case_id'] for c in self.manifest['cases'] if not self.completed(c['case_id'])]
        def task(cid):
            phase = self.run_case(cid, spec)
            if progress: progress({'case_id': cid, 'phase': phase, 'remaining': sum(1 for c in self.manifest['cases'] if c.get('phase') not in ('completed', 'failed'))})
            return phase
        with ThreadPoolExecutor(max_workers=workers) as pool: phases = list(pool.map(task, pending))
        self.manifest['summary'] = {'cases': len(self.manifest['cases']), 'completed': sum(1 for c in self.manifest['cases'] if c.get('phase') == 'completed'), 'failed': sum(1 for c in self.manifest['cases'] if c.get('phase') == 'failed'), 'skipped_already_complete': len(self.manifest['cases']) - len(pending)}
        self.save(); return self.manifest['summary']


def _spec_sha(spec):
    clean = {k: v for k, v in spec.items() if k != 'base_scene_path'}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
