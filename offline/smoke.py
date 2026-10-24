"""Run the packaged runtime through its HTTP API and a complete prediction."""
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen, build_opener, ProxyHandler
from urllib.error import HTTPError
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    opener = build_opener(ProxyHandler({}))  # Loopback only; never contacts an Internet proxy.
    with tempfile.TemporaryDirectory(prefix='workai-package-smoke-') as temp:
        directory = Path(temp)
        log = (directory / 'server.log').open('w+', encoding='utf-8')
        server = subprocess.Popen([sys.executable, '-X', 'utf8', '-m', 'offline.app', '--no-browser', '--data', str(directory / 'data')], cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            address = None
            while time.monotonic() < deadline:
                text = (directory / 'server.log').read_text(encoding='utf-8')
                import re
                match = re.search(r'http://127\.0\.0\.1:[0-9]+', text)
                if match:
                    address = match[0]
                    break
                if server.poll() is not None:
                    raise RuntimeError(text)
                time.sleep(.2)
            assert address, 'Server failed to start'
            def request(path, value=None, headers=None):
                req = Request(address + path, data=None if value is None else json.dumps(value).encode(), headers=headers or {})
                with opener.open(req, timeout=60) as response:
                    raw = response.read()
                    return json.loads(raw) if response.headers.get_content_type() == 'application/json' else raw
            info = request('/api/info')
            assert info['model']['original_model_restored'] is True
            assert info['model']['weights_sha256'] == 'abcf531d1c968786200e3a0a0bf944cd2b54000797fe7067bfef6f571b056a0d'
            scene = request('/api/templates')['plate']
            scene['geometry'].update(length_mm=8., width_mm=3., thickness_mm=1.5)
            scene['source']['center_mm'] = [0., 1.5, .75]
            for i, receiver in enumerate(scene['receivers']):
                receiver['point_mm'] = [2. if i == 0 else 6., 1.5, .75]
            body = {'scene': scene, 'degree': 4}
            headers = {'Content-Type': 'application/json', 'X-WorkAI-Token': info['token']}
            for bad in [{'Content-Type': 'application/json'}, dict(headers, Origin='https://unrelated.example')]:
                try:
                    request('/api/jobs', body, bad)
                    raise AssertionError('Untrusted mutation was accepted')
                except HTTPError as error:
                    assert error.code == 403
            invalid = json.loads(json.dumps(body))
            invalid['scene']['source']['frequency_mhz'] = 2
            try:
                request('/api/jobs', invalid, headers)
                raise AssertionError('Out-of-scope scene was accepted')
            except HTTPError as error:
                assert error.code == 422
            job = request('/api/jobs', body, headers)['job_id']
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                status = request('/api/jobs/' + job)
                if status['phase'] in ('completed', 'failed'):
                    break
                time.sleep(2)
            if status['phase'] != 'completed':
                raise AssertionError((status, (directory / 'data/jobs' / job / 'worker.log').read_text(encoding='utf-8')))
            fields = {}
            for part in ['full', 'linear', 'stvk', 'geometric', 'material']:
                result = request('/api/jobs/' + job + '/result?part=' + part)
                packed = request('/api/jobs/' + job + '/field?part=' + part)
                assert hashlib.sha256(packed).hexdigest() == result['field']['sha256']
                raw = gzip.decompress(packed)
                assert hashlib.sha256(raw).hexdigest() == result['field']['uncompressed_sha256']
                fields[part] = np.frombuffer(raw, dtype='<f8')
                assert np.isfinite(fields[part]).all()
                assert result['neural_model']['numerical_correction'] is False
                assert result['diagnostics']['external_work_balance_relative'] <= .01
            np.testing.assert_array_equal(fields['full'] - fields['stvk'], fields['material'])
            np.testing.assert_array_equal(fields['stvk'] - fields['linear'], fields['geometric'])
            assert np.max(abs(fields['full'])) > 1e-12
            assert np.max(abs(fields['material'])) > 1e-16
            archive = request('/api/jobs/' + job + '/export')
            assert archive[:2] == b'PK'
            print(json.dumps({'passed': True, 'python': sys.version, 'platform': sys.platform,
                              'weights_sha256': info['model']['weights_sha256'], 'elapsed_s': status['elapsed_s'],
                              'controls': 3, 'response_parts': 5, 'authentication_checks': 2, 'invalid_scene_rejected': True}), flush=True)
        finally:
            server.terminate()
            server.wait(timeout=15)
            log.close()


if __name__ == '__main__':
    main()
