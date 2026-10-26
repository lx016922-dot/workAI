"""Preserve bounded, independently audited complete-record research evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    summary=json.loads((args.audit/'summary.json').read_text())
    if summary['cases']!=20 or summary['scientific_accuracy_certified']:
        raise ValueError('Twenty independently audited research cases required')
    args.output.mkdir(parents=True,exist_ok=False)
    manifest={}
    for signal in ['hann','gaussian','rectangular','chirp']:
        target=args.output/f'WorkAI-GraphModal-v12-enriched264-{signal}-5ms.zip'
        with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
            for cycles in [1,3,5,10,20]:
                identifier=f'{signal}-{cycles:02d}';folder=args.root/('graph-modal-'+identifier)/identifier
                proof=json.loads((args.audit/(identifier+'.json')).read_text())
                if proof['case_id']!=identifier or proof['duration_us']!=5000 or proof['frames']!=25001:
                    raise ValueError('Incomplete audited case')
                for path in sorted(folder.iterdir()):
                    if not path.is_file() or path.is_symlink() or digest(path)!=proof['files_sha256'][path.name]:
                        raise ValueError('Audited evidence changed')
                    z.write(path,identifier+'/'+path.name)
                z.write(args.audit/(identifier+'.json'),identifier+'/independent-audit.json')
        if target.stat().st_size>=1900*1024**2:raise ValueError('Existing evidence archive bound exceeded')
        with zipfile.ZipFile(target) as z:
            if z.testzip() is not None:raise ValueError('Evidence archive readback failed')
        manifest[target.name]={'bytes':target.stat().st_size,'sha256':digest(target)}
    provenance={'run_id':os.environ.get('GITHUB_RUN_ID'),'source_commit':os.environ.get('GITHUB_SHA'),
                'model_archive_sha256':os.environ.get('MODEL_ARCHIVE_SHA256'),
                'scientific_accuracy_certified':False}
    target=args.output/'WorkAI-GraphModal-v12-enriched264-audit.zip'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(args.audit.iterdir()):
            if path.is_file():z.write(path,'audit/'+path.name)
        z.writestr('provenance.json',json.dumps(provenance,indent=2)+'\n')
        for name in ['tools/audit-modal-matrix.py','tools/package-enriched-evidence.py',
                     'research/comsol-validation/plans/v12-enriched-nodal-driven.json']:
            z.write(name,name)
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:raise ValueError('Audit archive readback failed')
    manifest[target.name]={'bytes':target.stat().st_size,'sha256':digest(target)}
    complete={'files':manifest,**provenance,'scope':'Complete nonlinear candidate trajectories and separate linear reference checks; no full 5 ms nonlinear COMSOL certificate'}
    (args.output/'WorkAI-GraphModal-v12-enriched264-manifest.json').write_text(json.dumps(complete,indent=2)+'\n')
    (args.output/'WorkAI-GraphModal-v12-enriched264-SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+k+'\n' for k,v in manifest.items()))
    print(json.dumps(complete),flush=True)


if __name__=='__main__':main()
