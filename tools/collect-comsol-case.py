"""Run/query one immutable COMSOL job and collect verified artifacts.

Network requests use the existing paired PowerShell signing client and proxy.
An uncertain POST is queried by its existing ID, never automatically retried.
"""
import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True)+'\n')
    tmp.replace(path)


def request_json(command, action, *options):
    """Retry only idempotent reads; an uncertain submission is never replayed."""
    attempts = 1 if action == 'submit' else 12
    transient = ['timeout', 'timed out', 'request was canceled', 'temporarily unavailable',
                 'http 502', 'http 503', 'http 504', 'connection reset']
    for attempt in range(attempts):
        try:
            result = subprocess.run(command + ['-Action', action, *map(str, options)],
                                    capture_output=True, text=True,
                                    timeout=1250 if action == 'download' else 45)
        except subprocess.TimeoutExpired as error:
            if action == 'submit':
                raise RuntimeError('Submission outcome is unknown; query its original job ID before any further action.') from error
            if attempt == attempts - 1:
                raise
            time.sleep(min(30, 3 * 2**attempt))
            continue
        if result.returncode:
            error = result.stderr.strip() or result.stdout[-2500:]
            if action == 'submit' or attempt == attempts - 1 or not any(s in error.lower() for s in transient):
                raise RuntimeError(error)
            time.sleep(min(30, 3 * 2**attempt))
            continue
        value = json.loads(result.stdout)
        if not value.get('server_signature_verified'):
            raise ValueError('Unverified server response')
        return value
    raise RuntimeError('Read request exhausted its bounded retry window')


def run(args):
    root, folder=args.root, args.case
    identity=json.loads((folder/'identity.json').read_text())
    job=identity['job_id']
    for name in ['scene','parameters']:
        if digest(folder/f'{name}.json') != identity[f'{name}_sha256']:
            raise ValueError(f'{name} changed after preparation')
    command=[str(root/'bin/workai-pwsh'),'-NoLogo','-NoProfile','-NonInteractive',
             '-File',str(Path(__file__).with_name('comsol-reference.ps1')),
             '-Root',str(root),'-JobId',job]

    def request(action, *options):
        return request_json(command, action, *options)

    if args.submit and not (root/'requests'/job).exists():
        if not args.source or digest(args.source)!=identity['source_sha256']:
            raise ValueError('Prepared COMSOL source changed')
        (folder/'UltraResearch.java').write_bytes(args.source.read_bytes())
        result=request('submit','-SourceFile',args.source,'-SceneFile',folder/'scene.json',
                       '-ParametersFile',folder/'parameters.json','-OutputFile',folder/'submitted.json')
        print(json.dumps({'job_id':job,'submitted':result['http_status']==202}),flush=True)
    deadline=time.monotonic()+args.timeout
    while True:
        status=request('status','-OutputFile',folder/'status.json')
        phase=status['body']['phase']
        live=request('log','-OutputFile',folder/'solver-live.json')
        body=live['body']
        print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'job_id':job,'phase':phase,
                          'log_bytes':body.get('total_bytes',0),'tail':body.get('tail_utf8','')[-700:]}),flush=True)
        if phase in ['completed','failed']:
            break
        if not args.wait:
            return
        if time.monotonic()>=deadline:
            raise TimeoutError('Job still running. Query the same ID to continue.')
        time.sleep(30)
    receipts=[]
    for item in status['body']['artifacts']:
        name=item['name']
        if name not in ['request.json','case.properties','completed.properties','coarse.csv','fine.csv','timecheck.csv',
                        'coarse.field.csv','fine.field.csv','timecheck.field.csv','coarse.mph','fine.mph','timecheck.mph',
                        'compile.log','solver.log','worker.log']:
            continue
        path=folder/name
        if path.exists():
            if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
                raise ValueError('Existing artifact differs from signed server manifest: '+name)
            receipts.append({**item,'verified_against_signed_manifest':True})
        else:
            receipt=request('download','-FileName',name,'-OutputFile',path)
            if receipt['sha256']!=item['sha256'] or receipt['bytes']!=item['bytes']:
                raise ValueError('Artifact identity changed during collection')
            receipts.append({**item,'server_signature_verified':True,'verified_against_signed_manifest':True})
        print(json.dumps({'job_id':job,'downloaded':name,'bytes':item['bytes']}),flush=True)
    save(folder/'collection.json',{'schema':'workai.comsol_verified_collection.v1','job_id':job,
        'collected_at_utc':datetime.now(timezone.utc).isoformat(),'phase':phase,'identity':identity,
        'signed_status_sha256':digest(folder/'status.json'),'artifacts':receipts})
    if phase!='completed':
        raise RuntimeError('COMSOL job failed; artifacts retained for diagnosis, excluded from training.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/workspace/workai-remote'))
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--source',type=Path)
    p.add_argument('--submit',action='store_true')
    p.add_argument('--wait',action='store_true')
    p.add_argument('--timeout',type=int,default=172800)
    run(p.parse_args())
