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


if __name__ == '__main__':
    unittest.main()
