"""Resume the P1 queue, collect real COMSOL data, and push verified results.

Only the existing signed client accesses the server. POSTs are never repeated
after a request ID is recorded. The runner waits for the current pilot to end;
it cannot and does not kill remote processes. Git pushes are never forced.
"""
import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from collect_dataset import archive, collect, material_check, verified, write
from validate import validate

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
BRANCH='data/comsol-p1-20261024'
OLD_JOB='comsol-553288731cd144e18544c0106f381838'
SOURCE=HERE/'UltraResearch.java'
STATE=HERE/'campaign-status.json'


def utc():
    return datetime.now(timezone.utc).isoformat()


def state(phase, **extra):
    old=json.loads(STATE.read_text()) if STATE.exists() else {}
    old.update(status=phase,updated_at_utc=utc(),**extra)
    write(STATE,old)
    print(json.dumps(dict(status=phase,utc=old['updated_at_utc'],**extra)),flush=True)


def push(message):
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    if branch!=BRANCH:
        raise RuntimeError('Unexpected Git branch; do not alter or force-push it')
    subprocess.run(['git','add','--','docs/comsol','research/comsol-p1-20261024'],cwd=REPO,check=True)
    diff=subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO)
    if diff.returncode==1:
        subprocess.run(['git','-c','core.whitespace=trailing-space,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=REPO,check=True)
        subprocess.run(['git','commit','-m',message],cwd=REPO,check=True)
    elif diff.returncode:
        raise RuntimeError('Cannot inspect staged changes')
    subprocess.run(['git','push','origin',f'HEAD:refs/heads/{BRANCH}'],cwd=REPO,check=True)


def wait_old_job(root):
    command=[str(root/'bin/workai-pwsh'),'-NoLogo','-NoProfile','-NonInteractive','-File',str(REPO/'tools/comsol-reference.ps1'),
             '-Root',str(root),'-Action','status','-JobId',OLD_JOB]
    while True:
        result=subprocess.run(command,capture_output=True,text=True,timeout=45)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
        response=json.loads(result.stdout)
        if not response.get('server_signature_verified'):
            raise RuntimeError('Unverified remote status')
        if response['body']['phase'] in ['completed','failed']:
            print('Prior pilot is terminal; queue may start.',flush=True)
            return
        state('waiting_for_pilot_to_finish',active_pilot_job=OLD_JOB,
              message='Existing API has no cancellation endpoint. Waiting for the identified pilot to finish or be stopped on the server.')
        time.sleep(30)


def run_job(root, folder):
    if (folder/'collection.json').exists():
        verified(folder)
        return
    ident=json.loads((folder/'identity.json').read_text())
    query=[str(root/'bin/workai-pwsh'),'-NoLogo','-NoProfile','-NonInteractive','-File',str(HERE/'query-job.ps1'),
           '-Root',str(root),'-JobId',ident['job_id']]
    result=subprocess.run(query,capture_output=True,text=True,timeout=45)
    if result.returncode:raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
    response=json.loads(result.stdout)
    if not response.get('server_signature_verified') or response['http_status'] not in [200,404]:
        raise RuntimeError('Cannot establish signed remote job state')
    submit=response['http_status']==404
    if submit and (root/'requests'/ident['job_id']).exists():
        raise RuntimeError('Prior POST exists but job is missing; investigate without repeating POST')
    shutil.copyfile(SOURCE,folder/'UltraResearch.java')
    cmd=[sys.executable,str(REPO/'tools/collect-comsol-case.py'),'--root',str(root),'--case',str(folder),
         '--source',str(SOURCE),'--wait','--timeout','604800']
    if submit:cmd.append('--submit')
    log_path=folder/'controller.log'
    with log_path.open('a') as log:
        proc=subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,text=True)
        last_report=0
        while proc.poll() is None:
            if time.monotonic()-last_report>=60:
                ident=json.loads((folder/'identity.json').read_text())
                progress=dict(active_job=ident['job_id'],active_case=ident['case_id'],active_laws=ident['laws'],active_variant=ident['variant'])
                live=folder/'solver-live.json'
                if live.exists():
                    body=json.loads(live.read_text())['body']
                    progress.update(remote_phase=body.get('phase'),solver_log_bytes=body.get('total_bytes'),last_remote_update=body.get('last_write_utc'))
                state('running',**progress)
                last_report=time.monotonic()
            time.sleep(15)
        if proc.returncode:
            raise RuntimeError(f'Job collector exited {proc.returncode}; see {log_path}. Preserve identity and query it; never blindly repeat POST.')
    verified(folder)


def verify_plan(plan, runs):
    digest=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if plan['source_sha256']!=digest:
        raise RuntimeError('Campaign source changed; prepare a reviewed new plan')
    expected={(case,law,variant) for case in ['A-f0.40-T1.6','B-f0.40-T1.6','C-f0.25-T2.0']
              for variant,laws in [('base',[0,1,2]),('mesh2',[0,2]),('dt2',[0,2])] for law in laws}
    actual=set()
    for name in [plan['material_check'],*plan['runs']]:
        folder=runs/name
        if not folder.exists():
            folder.mkdir(parents=True,exist_ok=False)
            for filename in ['scene.json','parameters.json','identity.json']:
                shutil.copyfile(HERE/'inputs'/name/filename,folder/filename)
    for name in plan['runs']:
        folder=runs/name;identity=json.loads((folder/'identity.json').read_text())
        if identity['source_sha256']!=digest:
            raise RuntimeError('Prepared source identity mismatch')
        for filename in ['scene','parameters']:
            if hashlib.sha256((folder/f'{filename}.json').read_bytes()).hexdigest()!=identity[f'{filename}_sha256']:
                raise RuntimeError('Prepared input changed')
        for law in identity['laws']:actual.add((identity['case_id'],law,identity['variant']))
    if actual!=expected or len(plan['runs'])!=21:
        raise RuntimeError('P1 plan must contain exactly 21 distinct runs')
    print('Verified complete immutable P1 plan: 21 runs.',flush=True)


def main(args):
    plan=json.loads((HERE/'plan.json').read_text());verify_plan(plan,args.runs)
    if args.verify_plan:return
    args.runs.mkdir(parents=True,exist_ok=True)
    with (args.runs/'campaign.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            wait_old_job(args.root)
            check_folder=args.runs/plan['material_check']
            run_job(args.root,check_folder)
            check=material_check(check_folder)
            data=HERE/'data';data.mkdir(exist_ok=True)
            write(data/'material-check.json',check);archive(check_folder,HERE/'evidence/material-check')
            state('material_check_passed',material_check='passed',active_job=None,message='Dedicated P1 source passed the COMSOL one-element constitutive checks.')
            if args.push:push('Record verified COMSOL P1 material checks for final solver settings')
            for name in plan['runs']:
                folder=args.runs/name
                run_job(args.root,folder)
                collect(argparse.Namespace(material_check=check_folder,runs=[folder],output=data))
                result=validate(data,partial=True)
                state('collecting',completed_transient_runs=result['completed_runs'],expected_runs=21,
                      message='Only completed, hash-verified real COMSOL outputs appear in data/.')
                if args.push:push(f"Add verified COMSOL P1 results: {name} ({result['completed_runs']}/21)")
            result=validate(data,partial=False)
            if not result['complete']:raise RuntimeError('P1 incomplete')
            output=args.runs.parent/'WorkAI_COMSOL_P1_20261024.zip'
            with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as z:
                for path in sorted(data.iterdir()):
                    if path.is_file():z.write(path,'data/'+path.name)
                z.write(HERE/'README.md','README.md')
                for path in sorted((REPO/'docs/comsol').iterdir()):
                    if path.is_file() and not path.name.startswith('.'):z.write(path,'contract/'+path.name)
            state('completed',completed_transient_runs=21,active_job=None,archive_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  message='All 21 P1 COMSOL runs collected and checked. See validation.json for convergence warnings; completion alone is not an accuracy certificate.')
            if args.push:push('Complete the 21-run COMSOL P1 dataset and convergence checks')
        except Exception as exc:
            state('blocked',message=str(exc))
            if args.push:
                try:push('Record COMSOL P1 queue blocker without adding incomplete data')
                except Exception as push_error:print(f'Git upload also pending: {push_error}',flush=True)
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,default=Path('/workspace/comsol-data-20261024/runs'))
    p.add_argument('--root',type=Path,default=Path('/workspace/workai-remote'))
    p.add_argument('--push',action='store_true');p.add_argument('--verify-plan',action='store_true')
    main(p.parse_args())
