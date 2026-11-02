"""Copy only verified scientific case files into a reproducible public dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source, destination):
    collection=json.loads((source/'collection.json').read_text())
    if collection['phase'] not in ['completed','failed']:
        raise ValueError('Only finished jobs can be archived')
    identity=collection['identity']
    status=source/'status.json'
    if digest(status)!=collection['signed_status_sha256']:
        raise ValueError('Collected signed status changed')
    envelope=json.loads(status.read_text())
    if not envelope.get('server_signature_verified') or envelope['body']['job_id']!=identity['job_id']:
        raise ValueError('Missing transfer signature verification')
    files={'collection.json','identity.json','status.json','scene.json','parameters.json','input.json','UltraResearch.java'}
    for name,key in [('scene.json','scene_sha256'),('parameters.json','parameters_sha256'),('UltraResearch.java','source_sha256')]:
        if digest(source/name)!=identity[key]:
            raise ValueError('Changed scientific input: '+name)
    for row in collection['artifacts']:
        name=row['name'];path=source/name
        if Path(name).name!=name or path.is_symlink() or not row.get('verified_against_signed_manifest'):
            raise ValueError('Unverified artifact')
        if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
            raise ValueError('Changed artifact: '+name)
        files.add(name)
    destination.mkdir(parents=True,exist_ok=False)
    for name in sorted(files):
        path=source/name
        if path.is_symlink() or not path.is_file():raise ValueError('Invalid input file')
        shutil.copyfile(path,destination/name)
    report={'schema':'workai.comsol_public_evidence.v1','job_id':identity['job_id'],
            'phase':collection['phase'],'identity':identity,
            'transfer_signature_verified':True,'raw_signature_headers_archived':False,
            'verification_scope':'Original collection verified signed transfer and artifact hashes. This archive retains hash evidence and transfer attestation, not independently replayable raw signature headers.',
            'failed_jobs_excluded_from_training':True,'long_accuracy_certified':False,
            'files_sha256':{name:digest(destination/name) for name in sorted(files)}}
    (destination/'archive.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'case':destination.name,'phase':collection['phase'],'files':len(files),
                      'bytes':sum((destination/name).stat().st_size for name in files)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cases',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    for case in args.cases:
        if Path(case).name!=case or case in ['.','..']:raise ValueError('Invalid case name')
        export(args.root/case,args.output/case)
