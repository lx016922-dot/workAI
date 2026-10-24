"""Local browser application with bounded APIs and managed inference processes."""
import argparse
import io
import json
import mimetypes
import os
import re
import secrets
import signal
import shutil
import subprocess
import sys
import threading
import webbrowser
import zipfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs, unquote
from .domain import ROOT, PARTS, digest, model_info, templates, validate_request

REPOSITORY = ROOT.parent


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


class Application:
    def __init__(self, data):
        self.data = data.resolve()
        self.jobs = self.data / 'jobs'
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.info = model_info()
        self.token = secrets.token_hex(24)
        self.lock = threading.Lock()
        self.process = None
        self.current = None
        self.closed = False
        for p in self.jobs.glob('*/status.json'):
            status = json.loads(p.read_text(encoding='utf-8'))
            if status['phase'] not in ('completed', 'failed'):
                status.update(phase='failed', message='上次退出时任务未完成，可重新提交。')
                save(p, status)
        for p in (ROOT / 'examples').glob('*/status.json'):
            target = self.jobs / p.parent.name
            if not target.exists():
                shutil.copytree(p.parent, target)

    def path(self, job):
        if not re.fullmatch(r'local-[a-f0-9]{32}', job):
            raise ValueError('任务编号无效。')
        p = self.jobs / job
        if not p.is_dir() or p.is_symlink():
            raise FileNotFoundError('任务不存在。')
        return p

    def status(self, job):
        return json.loads((self.path(job) / 'status.json').read_text(encoding='utf-8'))

    def submit(self, value):
        request = validate_request(value)
        with self.lock:
            if self.closed or (self.process is not None and self.process.poll() is None):
                raise ValueError('已有任务正在运行，请等待完成。')
            job = 'local-' + secrets.token_hex(16)
            directory = self.jobs / job
            directory.mkdir()
            save(directory / 'request.json', request)
            save(directory / 'status.json', {'job_id': job, 'name': request['scene']['name'], 'phase': 'starting',
                 'started_at_utc': datetime.now(timezone.utc).isoformat(), 'controls_completed': 0, 'elapsed_s': 0})
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUTF8='1',
                       NUMBA_CACHE_DIR=str(self.data / 'cache'))
            self.current = job
            try:
                with (directory / 'worker.log').open('wb') as log:
                    self.process = subprocess.Popen([sys.executable, '-X', 'utf8', '-m', 'offline.worker', '--job-dir', str(directory)],
                        cwd=REPOSITORY, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            except OSError:
                self.fail(job, '无法启动计算进程。请检查软件包是否完整。')
                raise
            threading.Thread(target=self.monitor, args=(job, self.process), daemon=True).start()
            return {'job_id': job}

    def fail(self, job, message):
        status = self.status(job)
        if status['phase'] not in ('completed', 'failed'):
            status.update(phase='failed', message=message)
            save(self.path(job) / 'status.json', status)

    def monitor(self, job, process):
        try:
            process.wait(timeout=5400)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            self.fail(job, '计算超过 90 分钟预算，已停止。')
        self.fail(job, '计算进程退出，未生成完整结果。请查看任务目录中的 worker.log。')

    def close(self):
        with self.lock:
            self.closed = True
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                self.fail(self.current, '应用已退出，计算已停止。')


def handler(application):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, data, content_type='application/json; charset=utf-8', attachment=None):
            if isinstance(data, (dict, list)):
                data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; connect-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; object-src 'none'; frame-ancestors 'none'")
            if attachment:
                self.send_header('Content-Disposition', 'attachment; filename="' + attachment + '"')
            self.end_headers()
            self.wfile.write(data)

        def local_request(self):
            expected = '127.0.0.1:' + str(self.server.server_port)
            if self.headers.get('Host') != expected:
                raise PermissionError('请使用启动窗口显示的本机地址。')
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + expected:
                raise PermissionError('不接受来自其他网站的请求。')

        def do_GET(self):
            try:
                self.local_request()
                url = urlsplit(self.path)
                if url.path == '/api/info':
                    return self.reply(200, {'model': application.info, 'token': application.token, 'data_directory': str(application.data)})
                if url.path == '/api/templates':
                    return self.reply(200, templates())
                if url.path == '/api/jobs':
                    rows = [application.status(p.name) for p in application.jobs.iterdir() if re.fullmatch(r'local-[a-f0-9]{32}', p.name)]
                    return self.reply(200, sorted(rows, key=lambda x: x['started_at_utc'], reverse=True))
                match = re.fullmatch(r'/api/jobs/(local-[a-f0-9]{32})(?:/(result|field|export|request))?', url.path)
                if match:
                    job, kind = match.groups()
                    directory = application.path(job)
                    status = application.status(job)
                    if not kind:
                        return self.reply(200, status)
                    if kind == 'request':
                        return self.reply(200, (directory / 'request.json').read_bytes())
                    if status['phase'] != 'completed':
                        return self.reply(409, {'error': '任务尚未完成。'})
                    if kind == 'export':
                        target = directory / 'results.zip'
                        # The API whitelist never exposes process logs or arbitrary paths.
                        with application.lock:
                            if not target.exists():
                                with zipfile.ZipFile(target, 'w', zipfile.ZIP_STORED) as archive:
                                    for name in ['request.json', 'status.json', 'exports.json'] + [p + ext for p in PARTS for ext in ('.json', '.bin.gz')]:
                                        archive.write(directory / name, name)
                        return self.reply(200, target.read_bytes(), 'application/zip', job + '.zip')
                    part = parse_qs(url.query).get('part', ['full'])[0]
                    if part not in PARTS:
                        raise ValueError('响应类型无效。')
                    exports = json.loads((directory / 'exports.json').read_text(encoding='utf-8'))
                    path = directory / (part + ('.json' if kind == 'result' else '.bin.gz'))
                    if digest(path) != exports[part][kind + '_sha256']:
                        raise ValueError('结果文件校验失败。')
                    return self.reply(200, path.read_bytes(), 'application/json; charset=utf-8' if kind == 'result' else 'application/gzip')
                relative = unquote(url.path).lstrip('/') or 'index.html'
                path = (ROOT / 'web' / relative).resolve()
                if not path.is_relative_to((ROOT / 'web').resolve()) or not path.is_file():
                    raise FileNotFoundError()
                mime = {'.js': 'text/javascript', '.css': 'text/css', '.html': 'text/html'}.get(path.suffix, mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
                return self.reply(200, path.read_bytes(), mime)
            except (ValueError, KeyError, TypeError) as error:
                self.reply(422, {'error': str(error)})
            except PermissionError as error:
                self.reply(403, {'error': str(error)})
            except FileNotFoundError:
                self.reply(404, {'error': '文件或任务不存在。'})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                self.reply(500, {'error': '读取失败，请检查数据文件和磁盘空间。'})

        def do_POST(self):
            try:
                self.local_request()
                if self.path != '/api/jobs':
                    raise FileNotFoundError()
                if not secrets.compare_digest(self.headers.get('X-WorkAI-Token', ''), application.token):
                    raise PermissionError('请刷新应用页面后重试。')
                if self.headers.get_content_type() != 'application/json':
                    raise ValueError('只接受 JSON 场景。')
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 65536:
                    raise ValueError('场景超过 64 KiB 或为空。')
                self.reply(202, application.submit(json.loads(self.rfile.read(size))))
            except (ValueError, TypeError, KeyError) as error:
                self.reply(422, {'error': str(error)})
            except PermissionError as error:
                self.reply(403, {'error': str(error)})
            except FileNotFoundError:
                self.reply(404, {'error': '接口不存在。'})
            except OSError:
                self.reply(500, {'error': '不能写入数据目录或启动进程。'})
    return Handler


def main():
    # Embeddable Windows Python ignores PYTHONUTF8 under its isolated _pth mode.
    # Keep redirected logs and Chinese paths readable even outside the launcher.
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path(os.environ.get('WORKAI_DATA_DIR', REPOSITORY / 'data')))
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    lockfile = (args.data / '.application.lock').open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            lockfile.seek(0)
            if not lockfile.read(1):
                lockfile.write(b'0'); lockfile.flush()
            lockfile.seek(0)
            msvcrt.locking(lockfile.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit('此数据目录已有 WorkAI 实例运行，请使用已打开的窗口。')
    app = Application(args.data)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(app))
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'WorkAI Offline — {url}\n数据目录：{app.data}\n按 Ctrl+C 退出；退出将停止进行中的计算。', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    def stop(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
        server.server_close()
        lockfile.close()


if __name__ == '__main__':
    main()
