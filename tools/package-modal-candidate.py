"""Package an immutable research basis, weights and their verified source snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def package(model,output):
    config=json.loads((model/'model.json').read_text())
    files=['model.json','basis.npz','weights.npz']
    for name,key in [('basis.npz','basis_sha256'),('weights.npz','weights_sha256')]:
        if digest(model/name)!=config[key]:raise ValueError('Compiled candidate changed: '+name)
    for name,sha in config['compilation_source_sha256'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe snapshot path')
        snapshot=model/'source_snapshot'/path
        if snapshot.is_symlink() or digest(snapshot)!=sha:raise ValueError('Compilation snapshot changed: '+name)
        files.append('source_snapshot/'+name)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    before={name:digest(model/name) for name in files}
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in sorted(files):archive.write(model/name,name)
    with zipfile.ZipFile(output) as archive:
        for name,sha in before.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=sha:
                raise ValueError('Archive readback mismatch: '+name)
    if before!={name:digest(model/name) for name in files}:raise ValueError('Candidate changed during packaging')
    result={'schema':'workai.graph_modal_package.v1','archive':output.name,'bytes':output.stat().st_size,
        'sha256':digest(output),'files_sha256':before,'archive_readback_verified':True,
        'requires_pinned_offline_python_runtime':True,'scientific_accuracy_certified':False}
    output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['archive','bytes','sha256','archive_readback_verified']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();package(a.model,a.output)
