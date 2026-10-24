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
from .spec import SCHEMA, ENGINES, ENGINE_IDS, ENGINE_ARGS, plan

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT_MODULES = ('physics', 'spectral2d', 'spectral3d', 'solid', 'dispersion', 'sweep', 'graphnet', 'graphwave')


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
        campaign.manifest = {'schema': 'workai.sweep.campaign.v1', 'name': spec['name'], 'created_at_utc': now(), 'spec_sha256': _spec_sha(spec), 'degree': spec['degree'], 'substeps': spec['substeps'],
                             'source_sha256': source_hashes(), 'engine': ENGINE_IDS[spec['engine']], 'sweep_engine': spec['engine'], 'solver_module': ENGINES[spec['engine']], 'accepted_for_solver_routing': False,
                             'cases': [{'case_id': r['case_id'], 'parameters': r['parameters'], 'split': r['split'], 'phase': 'planned'} for r in rows]}
        for r in rows:
            case_dir = campaign.cases / r['case_id']; case_dir.mkdir()
            atomic(case_dir / 'scene.json', r['scene']); atomic(case_dir / 'settings.json', {'degree': spec['degree'], 'substeps': spec['substeps']})
        campaign.save(); return campaign

    def save(self):
        with self.lock: self.manifest['updated_at_utc'] = now(); atomic(self.directory / 'manifest.json', self.manifest)

    def entry(self, cid): return next(c for c in self.manifest['cases'] if c['case_id'] == cid)

    def completed(self, cid):
        """A case is complete only when its result.json matches the manifest hash."""
        entry = self.entry(cid); result = self.cases / cid / 'out' / 'result.json'
        return entry.get('phase') == 'completed' and result.is_file() and digest(result) == entry.get('result_sha256')

    def solve(self, spec, scene_path, out, settings, log_path, fields=False, write_csv=False):
        """One isolated solver process; returns the exit code (or 'timeout')."""
        if out.exists(): shutil.rmtree(out)
        command = [sys.executable, '-m', ENGINES[spec['engine']], '--scene', str(scene_path), '--output', str(out), '--degree', str(settings['degree']), '--substeps', str(settings.get('substeps', spec['substeps']))]
        command += ENGINE_ARGS.get(spec['engine'], [])
        if fields: command.append('--fields')
        if write_csv: command.append('--csv')
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUTF8='1')
        with Path(log_path).open('w', encoding='utf-8') as log:
            try: return subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=spec['budget']['seconds_per_case']).returncode
            except subprocess.TimeoutExpired: return 'timeout'

    def run_case(self, cid, spec):
        case_dir = self.cases / cid; out = case_dir / 'out'; started = time.monotonic()
        with self.lock: self.entry(cid).update(phase='running', started_at_utc=now())
        self.save()
        settings = json.loads((case_dir / 'settings.json').read_text(encoding='utf-8'))
        code = self.solve(spec, case_dir / 'scene.json', out, settings, case_dir / 'run.log', fields=spec['outputs']['fields'], write_csv=spec['outputs']['csv'])
        elapsed = time.monotonic() - started; entry = self.entry(cid)
        if code == 0 and (out / 'result.json').is_file():
            record = json.loads((out / 'result.json').read_text(encoding='utf-8')); d = record['diagnostics']
            with self.lock: entry.update(phase='completed', result_sha256=digest(out / 'result.json'), files={k: v for k, v in record['files'].items()}, elapsed_s=elapsed, completed_at_utc=now(),
                                         nodes=d.get('nodes', record['mesh'].get('nodes')), internal_steps=d['internal_steps'], energy_balance_relative=d['external_work_balance_relative'], max_strain=d['max_strain'])
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


    def converge(self, spec, case_ids=None, overrides=None, workers=None, progress=None):
        """Refinement reruns for completed cases; reports go to cases/<id>/convergence/ and the manifest."""
        from .convergence import settings as conv_settings, variants, compare_runs, report_summary, selected_cases, refined, control_pairs, increment_comparison, increment_summary
        conv = conv_settings(spec, overrides); entries = [c for c in self.manifest['cases'] if self.completed(c['case_id'])]
        require(entries, '没有已完成的用例可做收敛复核。'); selected = selected_cases(spec, entries, case_ids)
        def task(cid):
            case_dir = self.cases / cid; folder = case_dir / 'convergence'; folder.mkdir(exist_ok=True)
            scene = json.loads((case_dir / 'scene.json').read_text(encoding='utf-8')); base_settings = json.loads((case_dir / 'settings.json').read_text(encoding='utf-8'))
            frequency = (spec.get('analysis') or {}).get('frequency_mhz') or scene['source']['frequency_mhz']
            report = {'schema': 'workai.sweep.convergence.v1', 'case_id': cid, 'settings': conv, 'levels': [], 'started_at_utc': now()}
            for variant in variants(spec, scene, base_settings['degree'], base_settings.get('substeps', spec['substeps']), conv):
                row = {'level': variant['level'], 'skipped': variant.get('skipped')}
                if variant.get('skipped'): report['levels'].append(row); continue
                level_dir = folder / variant['level']; level_dir.mkdir(exist_ok=True); atomic(level_dir / 'scene.json', variant['scene']); atomic(level_dir / 'settings.json', variant['settings'])
                code = self.solve(spec, level_dir / 'scene.json', level_dir / 'out', variant['settings'], level_dir / 'run.log')
                if code == 0 and (level_dir / 'out' / 'result.json').is_file():
                    try:
                        row['comparison'] = compare_runs(case_dir / 'out', level_dir / 'out', frequency, conv['tolerance'], conv.get('window_us'))
                        if not refined(variant['level'], row['comparison']): row['error'] = 'refined run is not finer than the base run (check h_mm / cfl rounding)'
                    except Exception as error: row['error'] = str(error)
                else: row['error'] = f'refined run failed (exit {code}); see {level_dir / "run.log"}'
                row['settings'] = variant['settings']; row['numerics'] = variant['scene']['numerics']; report['levels'].append(row)
            report['summary'] = report_summary(report); report['completed_at_utc'] = now(); atomic(folder / 'report.json', report)
            with self.lock: self.entry(cid)['convergence'] = dict(report['summary'], report_sha256=digest(folder / 'report.json'))
            self.save()
            if progress: progress({'case_id': cid, 'convergence_pass': report['summary']['pass'], 'levels': {k: v for k, v in report['summary']['levels'].items()}})
            return report['summary']
        with ThreadPoolExecutor(max_workers=workers or spec['budget']['workers']) as pool: results = list(pool.map(task, selected))
        increments = []
        for linear, nonlinear in control_pairs([e for e in entries if e['case_id'] in selected]):
            frequency = (spec.get('analysis') or {}).get('frequency_mhz') or json.loads((self.cases / nonlinear / 'scene.json').read_text(encoding='utf-8'))['source']['frequency_mhz']; rows = []
            skipped = {r['level']: r['skipped'] for r in json.loads((self.cases / nonlinear / 'convergence' / 'report.json').read_text(encoding='utf-8'))['levels'] if r.get('skipped')}
            for level in conv['levels']:
                outs = [self.cases / cid / 'convergence' / level / 'out' for cid in (linear, nonlinear)]
                if level in skipped: rows.append({'level': level, 'skipped': skipped[level]}); continue
                if not all((o / 'result.json').is_file() for o in outs): rows.append({'level': level, 'error': 'refined run missing or failed for the nonlinear case or its linear control'}); continue
                try: rows.append({'level': level, 'comparison': increment_comparison(self.cases / linear / 'out', outs[0], self.cases / nonlinear / 'out', outs[1], frequency, conv['tolerance'], conv.get('window_us'))})
                except Exception as error: rows.append({'level': level, 'error': str(error)})
            report = {'schema': 'workai.sweep.convergence.increment.v1', 'nonlinear_case': nonlinear, 'linear_control': linear, 'levels': rows, 'summary': increment_summary(rows), 'completed_at_utc': now(),
                      'note': 'nonlinear law minus linear control on each discretisation; verdict on the increment A2 only'}
            path = self.cases / nonlinear / 'convergence' / 'increment.json'; atomic(path, report)
            with self.lock: self.entry(nonlinear)['convergence']['increment'] = dict(report['summary'], linear_control=linear, report_sha256=digest(path))
            self.save(); increments.append(report['summary']['pass'])
            if progress: progress({'case_id': nonlinear, 'linear_control': linear, 'increment_pass': report['summary']['pass'], 'levels': report['summary']['levels']})
        return {'cases': len(selected), 'passed': sum(1 for r in results if r['pass']), 'failed': sum(1 for r in results if not r['pass']),
                'increment_pairs': len(increments), 'increment_passed': sum(increments), 'increment_failed': len(increments) - sum(increments)}


def _spec_sha(spec):
    """Hash of what defines the cases. `convergence` only steers later checks; `engine`/`substeps` at their
    plane-strain defaults are dropped so campaigns created before those fields existed still resume."""
    defaults = {'engine': '2d_plane_strain', 'substeps': 4}
    clean = {k: v for k, v in spec.items() if k not in ('base_scene_path', 'convergence') and not (k in defaults and v == defaults[k] and spec.get('engine') == '2d_plane_strain')}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
