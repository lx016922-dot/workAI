"""Local server endpoints the desktop window relies on: the application version and the token-protected shutdown."""
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parents[1]


class DesktopEndpoints(unittest.TestCase):
    def test_version_and_shutdown(self):
        opener = build_opener(ProxyHandler({}))
        version = json.loads((ROOT / 'offline/version.json').read_text(encoding='utf-8'))
        self.assertRegex(version['version'], r'^\d+\.\d+\.\d+$')
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / 'server.log'
            with log.open('w', encoding='utf-8') as out:
                server = subprocess.Popen([sys.executable, '-X', 'utf8', '-m', 'offline.app', '--no-browser', '--data', str(Path(temp) / 'data')],
                                          cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
            try:
                address = None
                for _ in range(300):
                    text = log.read_text(encoding='utf-8')
                    if 'http://127.0.0.1:' in text:
                        address = 'http://127.0.0.1:' + text.split('http://127.0.0.1:')[1].split()[0]
                        break
                    self.assertIsNone(server.poll(), text)
                    time.sleep(.1)
                self.assertIsNotNone(address)
                info = json.load(opener.open(address + '/api/info', timeout=30))
                self.assertEqual(info['app']['version'], version['version'])
                post = lambda headers: opener.open(Request(address + '/api/shutdown', data=b'{}', headers=headers), timeout=30)
                with self.assertRaises(HTTPError) as refused:
                    post({'Content-Type': 'application/json'})   # no token
                self.assertEqual(refused.exception.code, 403)
                self.assertIsNone(server.poll())
                reply = json.load(post({'Content-Type': 'application/json', 'X-WorkAI-Token': info['token']}))
                self.assertIn('message', reply)
                self.assertEqual(server.wait(timeout=30), 0)
            finally:
                if server.poll() is None:
                    server.kill()
                    server.wait()


class JobQueue(unittest.TestCase):
    def test_a_job_right_after_a_finished_one_waits_for_the_old_worker_to_exit(self):
        # The worker publishes 'completed' and then still needs a moment to exit; a new job submitted in that moment
        # must not be refused as already running (the packaged smoke test hit this after 1.1.1 made defect jobs larger).
        from offline.app import Application
        with tempfile.TemporaryDirectory() as temp:
            app = Application(Path(temp) / 'data')
            job = 'local-' + '0' * 32; (app.jobs / job).mkdir()
            (app.jobs / job / 'status.json').write_text(json.dumps({'job_id': job, 'phase': 'completed'}), encoding='utf-8')
            app.current = job; app.process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(1.5)'])
            started = time.monotonic(); app.settle()
            self.assertIsNotNone(app.process.poll()); self.assertGreater(time.monotonic() - started, .5)
            (app.jobs / job / 'status.json').write_text(json.dumps({'job_id': job, 'phase': 'running'}), encoding='utf-8')
            app.process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
            try:
                started = time.monotonic(); app.settle()   # a running job is not waited for: submit refuses at once
                self.assertLess(time.monotonic() - started, 5); self.assertIsNone(app.process.poll())
            finally:
                app.process.kill(); app.process.wait()


if __name__ == '__main__':
    unittest.main()
