import datetime,hashlib,json,subprocess,time
from pathlib import Path

repo=Path('/workspace/workAI');remote=Path('/workspace/workai-remote/graphwave/comsol')
data=repo/'research/comsol-validation/data';diagnostics=repo/'research/comsol-validation/diagnostics'
candidate=diagnostics/'v11-native500-refinement/time500'
py='/workspace/offline-v3-linux-check/WorkAI-Offline-linux-x64/runtime/bin/python3'
sources=['tools/recompare-native-modal-prediction.py','tools/check-modal-refinements.py',
         'tools/validate-comsol-linear-matrix.py','tools/compare-comsol-reference.py',
         'tools/comsol_artifacts.py','tools/export-comsol-evidence.py']
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
frozen={name:digest(repo/name) for name in sources}
candidate_files={name:digest(candidate/name) for name in ['identity.json','prediction.json','prediction.npz','metrics.json']}
jobs={'native-quarter-050-20us-v11':'comsol-ec59b2ae07c94042ab34ca6c54b71001',
      'native-quarter-050-time-20us-v11':'comsol-af378dafdc114defb1b8dfb2113c645d'}
plan={'schema':'workai.queued_native_recomparison.v1','queued_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'sources_sha256':frozen,'candidate_files_sha256':candidate_files,'reference_jobs':jobs,
      'candidate_retraining_or_inference':False,'duration_us':20,'scientific_accuracy_certified':False}
plan_path=diagnostics/'v11-native500-refinement-followup-plan.json'
if plan_path.exists():raise RuntimeError('Follow-up plan already exists; inspect original process')
plan_path.write_text(json.dumps(plan,indent=2)+'\n')
def check():
 if any(digest(repo/name)!=sha for name,sha in frozen.items()) or any(digest(candidate/name)!=sha for name,sha in candidate_files.items()):
  raise RuntimeError('Frozen follow-up source or prediction changed; explicit replanning required')
deadline=time.monotonic()+16*3600
references=[data/'native-quarter-075-20us-v11']
for name,job in jobs.items():
 folder=remote/name
 while not (folder/'collection.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('Native reference collection remains incomplete')
  time.sleep(30)
 check();collection=json.loads((folder/'collection.json').read_text())
 if collection['phase']!='completed' or collection['job_id']!=job:raise RuntimeError('Unexpected or failed reference collection')
 if not (data/name).exists():
  subprocess.run([py,'tools/export-comsol-evidence.py','--root',str(remote),'--cases',name,'--output',str(data)],cwd=repo,check=True)
 references.append(data/name)
 output=diagnostics/('v11-native500-vs-'+('050-time' if 'time' in name else '050-space'))
 subprocess.run([py,'tools/recompare-native-modal-prediction.py','--prediction',str(candidate),
                 '--references',*[str(p) for p in references],'--output',str(output)],cwd=repo,check=True)
 check();print(json.dumps({'reference_job':job,'recomparison_completed':str(output),'candidate_changed':False}),flush=True)
print('Both native spatial/time comparisons completed; inspect all gates before further training',flush=True)
