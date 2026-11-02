"""One immutable plane-strain web job: scene.json + settings.json in, result.json + binaries out."""
import argparse
import hashlib
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from solid.schema import require
from .run import simulate, browser_record


def now(): return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8'); temporary.replace(path)


def run(directory, time_budget_s=1800):
    directory = Path(directory); started = time.monotonic(); status = {'job_id': directory.name, 'phase': 'meshing', 'started_at_utc': now()}
    def publish(**values): status.update(values, updated_at_utc=now(), elapsed_s=time.monotonic() - started); atomic(directory / 'status.json', status)
    def timeout(*_): raise TimeoutError('二维计算超过时间预算，任务已停止。')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(time_budget_s)
    try:
        raw = json.loads((directory / 'scene.json').read_text(encoding='utf-8')); settings = json.loads((directory / 'settings.json').read_text(encoding='utf-8'))
        require(set(settings) == {'degree'} and type(settings['degree']) is int and 2 <= settings['degree'] <= 6, '二维阶次需为 2–6 的整数。')
        publish()
        def progress(value): publish(phase='running', step=value.get('frame', 0), steps_total=value.get('frames', 0) - 1)
        def guard(scene_mesh):
            mesh = scene_mesh['mesh']; frames = round(scene_mesh['scene']['acquisition']['duration_us'] / scene_mesh['scene']['acquisition']['sample_step_us']) + 1
            require(len(mesh['points']) <= 60000 and mesh['elements'].size <= 400000 and frames * len(mesh['points']) * 16 <= 768 * 1024**2, '二维网格或场存储超出网页预算；请增大网格尺寸、缩短记录或降低阶次。')
            publish(phase='running', step=0, steps_total=frames - 1, nodes=len(mesh['points']), elements=len(mesh['elements']))
        result = simulate(raw, settings['degree'], keep_fields=True, callback=progress, before_solve=guard)
        publish(phase='exporting')
        record, field_gz, lines_gz = browser_record(result, directory.name)
        record.update(scene_sha256=hashlib.sha256((directory / 'scene.json').read_bytes()).hexdigest(), settings_sha256=hashlib.sha256((directory / 'settings.json').read_bytes()).hexdigest())
        (directory / 'field.bin.gz').write_bytes(field_gz); (directory / 'lines.bin.gz').write_bytes(lines_gz); atomic(directory / 'result.json', record)
        publish(phase='completed', result_sha256=hashlib.sha256((directory / 'result.json').read_bytes()).hexdigest(), accepted_for_solver_routing=False,
                diagnostics={k: v for k, v in record['diagnostics'].items() if k != 'energy_samples'})
        print(json.dumps({'completed': True, 'job_id': directory.name, 'nodes': record['diagnostics']['nodes'], 'seconds': time.monotonic() - started}), flush=True)
    except BaseException as error:
        publish(phase='failed', error=str(error)); raise
    finally: signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--job-dir', type=Path, required=True); run(parser.parse_args().job_dir)
