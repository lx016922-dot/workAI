"""Bundle a complete research candidate and its network-free prediction CLI."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(value):return hashlib.sha256(value).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    config=json.loads((args.model/'model.json').read_text());plan=json.loads(args.plan.read_text())
    if config.get('scientific_accuracy_certified') or plan.get('scientific_accuracy_certified'):
        raise ValueError('This package is an explicitly uncertified research candidate')
    for name,key in [('basis.npz','basis_sha256'),('weights.npz','weights_sha256'),('report.json','training_report_sha256')]:
        if digest((args.model/name).read_bytes())!=config[key]:raise ValueError('Candidate identity changed')
    files={}
    for name in ['model.json','basis.npz','weights.npz','identity.json','report.json','manufactured-states.npz']:
        files['model/'+name]=(args.model/name).read_bytes()
    for name,sha in config['compilation_source_sha256'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe source path')
        data=(args.model/'source_snapshot'/name).read_bytes()
        if digest(data)!=sha:raise ValueError('Training source snapshot changed')
        files['model/source_snapshot/'+name]=data
    runtime=set(plan['source_sha256'])
    for folder in ['graphwave','spectral3d']:
        runtime.update(str(path.relative_to(repo)) for path in (repo/folder).glob('*.py'))
    for name in sorted(runtime):
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe runtime path')
        data=(repo/name).read_bytes()
        if name in plan['source_sha256'] and digest(data)!=plan['source_sha256'][name]:
            raise ValueError('Frozen inference source changed: '+name)
        files['code/'+name]=data
    files['code/plan.json']=args.plan.read_bytes()
    launcher='''import argparse,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
plan=json.loads((root/'code/plan.json').read_text())
p=argparse.ArgumentParser(description='Offline research prediction: complete 5 ms, no accuracy certificate')
p.add_argument('--case',choices=[c['id'] for c in plan['cases']],required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--step-us',type=float,choices=[.05,.025,.0125],default=.05)
a=p.parse_args()
model=json.loads((root/'model/model.json').read_text())
raise SystemExit(subprocess.call([sys.executable,str(root/'code/tools/run-modal-nodal-case.py'),
 '--model',str(root/'model'),'--plan',str(root/'code/plan.json'),'--case',a.case,
 '--modes',str(model['exported_modes']),'--step-us',str(a.step_us),'--output',str(a.output.resolve())]))
'''
    files['run.py']=launcher.encode()
    files['requirements.txt']=b'numpy==2.2.6\nscipy==1.15.3\nnumba==0.63.1\nllvmlite==0.46.0\n'
    files['README.txt']=('''WorkAI nonlinear basis research candidate — offline prediction

Use Python 3.12 from the existing WorkAI Offline v0.3.0 runtime. That bundle
already contains the pinned packages in requirements.txt; prediction makes no
network requests and needs no COMSOL installation, server, credential or label.

Example (run the command with the existing offline Python executable):
  python run.py --case hann-05 --output results

Available signals: hann, gaussian, rectangular, chirp.
Available cycle counts: 01, 03, 05, 10, 20. Every case covers a real 5 ms interval
with 25,001 samples. Default internal step: 0.05 us. Calculations can take hours.
Each result preserves all three constitutive controls, coordinates and energy.

Scope: the single fixed rectangular aluminum plate and uniform axial traction
in code/plan.json. This is a research candidate, not a replacement for the
default model or a certificate of general/full-5-ms nonlinear accuracy.

The complete graph quadrature, learned material tensors and coupled 3D equations
are retained. Static solves occur only during basis training. No reference array
or numerical correction enters prediction. Source and training provenance are
included; manifest.json records every packaged file's SHA-256.
''').encode()
    manifest={'files_sha256':{name:digest(data) for name,data in files.items()},
              'basis_sha256':config['basis_sha256'],'weights_sha256':config['weights_sha256'],
              'requires_existing_pinned_offline_runtime':True,'network_required_for_prediction':False,
              'default_internal_step_us':.05,'duration_us':5000,'scientific_accuracy_certified':False}
    files['manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    if len(files)>256 or sum(map(len,files.values()))>512*1024**2:raise ValueError('Existing package bounds exceeded')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(files.items()):z.writestr(name,data)
    with zipfile.ZipFile(args.output) as z:
        for name,data in files.items():
            if digest(z.read(name))!=digest(data):raise ValueError('Package readback mismatch')
    receipt={'archive':args.output.name,'bytes':args.output.stat().st_size,'sha256':digest(args.output.read_bytes()),
             'archive_readback_verified':True,'files':len(files),'scientific_accuracy_certified':False}
    args.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
