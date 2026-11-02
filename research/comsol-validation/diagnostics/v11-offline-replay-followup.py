import datetime,hashlib,json,shutil,stat,subprocess,time,zipfile
from pathlib import Path,PurePosixPath

repo=Path('/workspace/workAI');root=Path('/workspace/recovery');repository='lx016922-dot/workAI'
offline=Path('/workspace/workai-remote/graphwave/modal-v11/phase-driven-offline5ms/hann-01')
receipt=root/'v11-audit-dispatch.json';asset='WorkAI-GraphModal-v11-driven-hann-5ms.zip'
manifest_name='WorkAI-GraphModal-v11-driven-manifest.json'
py='/workspace/offline-v3-linux-check/WorkAI-Offline-linux-x64/runtime/bin/python3'
sources=['tools/check-modal-offline-replay.py','tools/check-modal-refinements.py',
         'tools/validate-comsol-linear-matrix.py','tools/compare-comsol-reference.py']
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
frozen={name:digest(repo/name) for name in sources}
plan={'schema':'workai.queued_offline_replay.v1','queued_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'prediction_case':'hann-01','duration_us':5000,'source_matrix_run':40204192440,
      'source_sha256':frozen,'cloud_asset':asset,'scientific_accuracy_certified':False}
plan_path=repo/'research/comsol-validation/diagnostics/v11-offline-replay-plan.json'
if plan_path.exists():raise RuntimeError('Existing replay plan; inspect original process instead of duplicating downloads')
plan_path.write_text(json.dumps(plan,indent=2)+'\n')
def check():
 if any(digest(repo/name)!=sha for name,sha in frozen.items()):raise RuntimeError('Frozen replay checker changed; explicit replanning required')
def api(path):return json.loads(subprocess.check_output(['gh','api',path],text=True,stderr=subprocess.PIPE,timeout=90))
deadline=time.monotonic()+18*3600
while True:
 if time.monotonic()>deadline:raise TimeoutError('Offline record or matrix publication remains incomplete')
 if (offline/'metrics.json').exists() and receipt.exists():
  try:
   audit=json.loads(receipt.read_text())
   if audit.get('status')=='run_identified':
    state=api(f'repos/{repository}/actions/runs/{audit["run_id"]}')
    if state['head_sha']!=audit['audit_head']:raise RuntimeError('Audit source changed')
    if state['status']=='completed':
     if state['conclusion']!='success':raise RuntimeError('Matrix audit/publication failed; inspect retained evidence')
     break
  except (subprocess.CalledProcessError,subprocess.TimeoutExpired,json.JSONDecodeError):pass
 time.sleep(45)
check();folder=root/'v11-offline-cloud-evidence';folder.mkdir(exist_ok=False)
def download(name):
 result=subprocess.run(['gh','release','download','v0.3.0','--repo',repository,'--pattern',name,'--dir',str(folder)],capture_output=True,text=True,timeout=1800)
 if result.returncode:raise RuntimeError('Release download failed; no accuracy or replay assertion made')
download(manifest_name)
release=api(f'repos/{repository}/releases/tags/v0.3.0')
published=next(x for x in release['assets'] if x['name']==manifest_name)
if published.get('digest')!='sha256:'+digest(folder/manifest_name):raise RuntimeError('Published manifest digest mismatch')
manifest=json.loads((folder/manifest_name).read_text());item=manifest['files'][asset]
if not 0<item['bytes']<1900*1024**2 or shutil.disk_usage(folder).free<item['bytes']+512*1024**2:raise RuntimeError('Evidence exceeds the existing download/disk budget')
download(asset)
archive=folder/asset
if archive.stat().st_size!=item['bytes'] or digest(archive)!=item['sha256']:raise RuntimeError('Complete cloud evidence ZIP identity mismatch')
with zipfile.ZipFile(archive) as z:
 entries=z.infolist()
 if len(entries)>64 or sum(x.file_size for x in entries)>1900*1024**2:raise RuntimeError('Unexpected evidence archive size')
 for entry in entries:
  p=PurePosixPath(entry.filename)
  if p.is_absolute() or '..' in p.parts or '\\' in entry.filename or stat.S_ISLNK(entry.external_attr>>16):raise RuntimeError('Unsafe archive entry')
  if p.parts[0]=='hann-01':z.extract(entry,folder/'extracted')
check();output=repo/'research/comsol-validation/diagnostics/v11-offline5ms-replay'
subprocess.run([py,'tools/check-modal-offline-replay.py','--offline',str(offline),'--cloud',str(folder/'extracted/hann-01'),'--output',str(output)],cwd=repo,check=True)
check();proof={'audit_run':audit['run_id'],'matrix_run':40204192440,'cloud_asset':asset,'cloud_asset_sha256':item['sha256'],'cloud_asset_bytes':item['bytes'],'manifest_sha256':digest(folder/manifest_name),'source_sha256':frozen,'scientific_accuracy_certified':False}
(output/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({'complete_offline_cloud_replay_checked':True,'report':str(output/'summary.json'),'scientific_accuracy_certified':False}),flush=True)
