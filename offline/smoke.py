"""Run the packaged runtime through its HTTP API and a complete prediction."""
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
import shutil
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
            scene['numerics']['h_mm']=2.5
            scene['boundary']['fixed_face']='xmax'
            assert scene['acquisition']['duration_us']==5000
            for i, receiver in enumerate(scene['receivers']):
                receiver['point_mm'] = [2. if i == 0 else 6., 1.5, .75]
            body = {'scene': scene, 'degree': 4}
            headers = {'Content-Type': 'application/json', 'X-WorkAI-Token': info['token']}
            def wait(job, limit_s):
                deadline = time.monotonic() + limit_s
                while time.monotonic() < deadline:
                    status = request('/api/jobs/' + job)
                    if status['phase'] in ('completed', 'failed'):
                        break
                    time.sleep(2)
                if status['phase'] != 'completed':
                    raise AssertionError((status, (directory / 'data/jobs' / job / 'worker.log').read_text(encoding='utf-8')))
                return status
            # Fast engine (learned coarse-graph operator), as the workbench's v2 top-face Gaussian template: every shipped
            # model must match its recorded evaluation (a CRLF checkout broke this in 1.0.0) and the default model must run.
            fast = request('/api/templates')['plate']
            fast.pop('signal'); fast.update(schema='workai.hyperelastic.solid.v2', name='fast engine smoke')
            fast['geometry'].update(length_mm=48., width_mm=2., thickness_mm=1.6)
            fast['source'] = {'name': 'top', 'face': 'zmax', 'center_mm': [16., 1., 1.6], 'radius_mm': 0., 'direction': [0., 0., -1.], 'amplitude_mpa': 1.,
                              'frequency_mhz': .4, 'cycles': 5., 'delay_us': 0., 'phase_deg': 0., 'signal': {'type': 'tone_burst', 'window': 'hann'},
                              'profile': {'type': 'gaussian', 'sigma_mm': 2.}}
            fast['sources'] = []
            fast['boundaries'] = [{'type': 'symmetry', 'face': 'ymax'}] + [{'type': 'absorbing', 'face': face, 'length_mm': 8., 'ramp': 'cubic', 'attenuation_db': 60.} for face in ('xmin', 'xmax')]
            fast['acquisition'] = {'duration_us': 20., 'sample_step_us': .05}
            fast['receivers'] = [{'name': 'top', 'point_mm': [22., 1., 1.6], 'direction': [0., 0., 1.]}]
            fast_models = ('p4-h3-harmonic', 'p4-h2-multisource')
            for name in fast_models:
                checked = request('/api/preview', {'scene': fast, 'degree': 4, 'engine': 'coarse', 'coarse_model': name}, headers)
                assert checked['ok'] and not checked['errors'], checked['errors']
                assert checked['engine']['name'] == name and len(checked['engine']['test_by_excitation']) == 4, checked['engine']
            fast_job = request('/api/jobs', {'scene': fast, 'degree': 4, 'engine': 'coarse'}, headers)['job_id']
            fast_status = wait(fast_job, 900)
            fast_result = request('/api/jobs/' + fast_job + '/result?part=full')
            assert fast_result['engine'] == 'workai_coarse_graph_offline' and fast_result['coarse_model']['name'] == fast_models[0]
            assert fast_result['recording']['frame_count'] == 401 and all(c['energy_balance_within_1_percent'] for c in fast_result['controls'])
            fast_field = np.frombuffer(gzip.decompress(request('/api/jobs/' + fast_job + '/field?part=full')), dtype='<f8')
            assert np.isfinite(fast_field).all() and np.max(abs(fast_field)) > 1e-12
            # Defects, automatic mesh, text box and one-click defects (1.1): preview, then a short job with a closed crack.
            cracked = json.loads(json.dumps(fast)); cracked['defects'] = [{'type': 'crack', 'x_mm': 24., 'depth_mm': .5, 'closed': True}]
            checked = request('/api/preview', {'scene': cracked, 'mesh': 'auto'}, headers)
            assert checked['ok'] and checked['mesh']['auto']['contact_pairs'] > 0, (checked['errors'], checked.get('mesh'))
            told = request('/api/interpret', {'text': '在 34 mm 处添加直径 0.6 mm 的通孔；在 8 到 14 mm 之间添加分层，距上表面 0.8 mm', 'scene': cracked}, headers)
            assert told['changed'] and [d['type'] for d in told['scene']['defects']] == ['crack', 'hole', 'delamination'], told['operations']
            assert request('/api/preview', {'scene': told['scene'], 'mesh': 'auto'}, headers)['ok']
            assert request('/api/defect', {'type': 'contact', 'scene': cracked}, headers)['defect']['type'] == 'contact'
            small = json.loads(json.dumps(cracked)); small['geometry']['length_mm'] = 16.; small['defects'][0]['x_mm'] = 8.
            small['source']['center_mm'] = [4., 1., 1.6]; small['source']['frequency_mhz'] = .3; small['source']['cycles'] = 2.; small['source']['profile'] = {'type': 'gaussian', 'sigma_mm': 1.}
            small['boundaries'] = [{'type': 'symmetry', 'face': 'ymax'}]; small['acquisition'] = {'duration_us': 10., 'sample_step_us': .05}
            small['receivers'] = [{'name': 'after', 'point_mm': [12., 1., 1.6], 'direction': [0., 0., 1.]}]
            defect_job = request('/api/jobs', {'scene': small, 'mesh': 'auto'}, headers)['job_id']
            defect_status = wait(defect_job, 1800)
            defect_result = request('/api/jobs/' + defect_job + '/result?part=full')
            assert defect_result['scene']['defects'][0]['closed'] is True and all(c['energy_balance_within_1_percent'] for c in defect_result['controls'])
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
            status = wait(job, 5400)
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
                assert result['recording']['frame_count']==25001 and result['times_us'][-1]==5000
                assert result['neural_model']['weights_sha256']==info['model']['long_model']['files_sha256']['continued-weights.npz']
                assert result['diagnostics']['external_work_balance_relative'] <= .01
            np.testing.assert_array_equal(fields['full'] - fields['stvk'], fields['material'])
            np.testing.assert_array_equal(fields['stvk'] - fields['linear'], fields['geometric'])
            assert np.max(abs(fields['full'])) > 1e-12
            assert np.max(abs(fields['material'])) > 1e-16
            trace=request('/api/jobs/'+job+'/traces',{'part':'full','points':[[2.,1.5,.75]]},headers)
            assert len(trace['times_us'])==25001 and trace['times_us'][-1]==5000
            assert np.isfinite(trace['points'][0]['xyz_m']).all()
            with opener.open(address+'/api/jobs/'+job+'/export',timeout=120) as response,(directory/'result.zip').open('wb') as output:
                shutil.copyfileobj(response,output,1024*1024)
            with zipfile.ZipFile(directory/'result.zip') as archive:
                assert 'record/recording.json' in archive.namelist()
                assert len([n for n in archive.namelist() if n.startswith('record/fields/')])>1
                assert archive.testzip() is None
            print(json.dumps({'passed': True, 'python': sys.version, 'platform': sys.platform,
                              'weights_sha256': result['neural_model']['weights_sha256'],
                              'original_weights_sha256': info['model']['weights_sha256'], 'elapsed_s': status['elapsed_s'],
                              'duration_us':5000,'full_samples':25001,'controls': 3, 'response_parts': 5, 'authentication_checks': 2, 'invalid_scene_rejected': True,
                              'fast_engine_models_checked': len(fast_models), 'fast_engine_elapsed_s': fast_status['elapsed_s'],
                              'defect_job_elapsed_s': defect_status['elapsed_s'], 'text_box_operations': len(told['operations'])}), flush=True)
        finally:
            server.terminate()
            server.wait(timeout=15)
            log.close()


if __name__ == '__main__':
    main()
