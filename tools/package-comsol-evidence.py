"""Preserve audited neural records plus COMSOL and physics training evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    repo=Path(__file__).resolve().parents[1];research=repo/'research/comsol-validation'
    report=json.loads((research/'candidate-matrix20/summary.json').read_text())
    if report['completed_cases']!=20 or not report['metrics_independently_recomputed']:
        raise ValueError('The complete independent audit must be committed first')
    args.output.mkdir(parents=True,exist_ok=True)
    archive=args.output/'WorkAI-COMSOL-physics-evidence-0.3.0.zip';entries=[]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        def add(path,name):
            z.write(path,name);entries.append({'path':name,'bytes':path.stat().st_size,'sha256':sha(path)})
        for row in report['rows']:
            case=row['case_id']
            choices=[args.input/('comsol-linear-'+case)/case,args.input/case/case]
            present=[path for path in choices if (path/'metrics.json').exists()]
            if len(present)!=1:raise ValueError('Missing or ambiguous completed case: '+case)
            directory=present[0]
            for name,key in [('identity.json','identity_sha256'),('metrics.json','metrics_sha256'),('comsol-modal-reference.npz','comsol_modal_reference_sha256')]:
                if sha(directory/name)!=row[key]:raise ValueError('Downloaded evidence differs from audit')
            identity=json.loads((directory/'identity.json').read_text())
            metrics=json.loads((directory/'metrics.json').read_text());prediction=metrics['prediction']
            if identity['weights_sha256']!=report['weights_sha256'] or prediction['frames']!=25001 or prediction['duration_us']!=5000:
                raise ValueError('Wrong model or incomplete record')
            if json.loads((directory/'prediction.json').read_text())!=prediction:raise ValueError('Prediction receipt differs')
            progress=json.loads((directory/'progress.json').read_text())
            if progress['phase']!='completed' or progress['completed_frames']!=25001:raise ValueError('Incomplete record')
            names=['identity.json','metrics.json','prediction.json','progress.json','comsol-modal-reference.npz']
            first=0
            for block in prediction['probe_blocks']:
                name=block['file']
                if Path(name).name!=name or block['first_frame']!=first or sha(directory/name)!=block['sha256']:
                    raise ValueError('Changed or discontinuous probe data')
                first=block['last_frame']+1;names.append(name)
            if first!=25001:raise ValueError('Missing end of record')
            for name in names:add(directory/name,'candidate-cases/'+case+'/'+name)
        for path in sorted(research.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.name.endswith('.decoded.csv'):continue
            if path.suffix.lower() in ['.pt','.pth','.safetensors']:raise ValueError('No new optimizer checkpoints in this archive')
            add(path,'research/'+path.relative_to(research).as_posix())
        z.writestr('MANIFEST.json',json.dumps({'schema':'workai.comsol_physics_evidence.v1',
            'repository_source_commit':os.environ.get('WORKAI_EVIDENCE_COMMIT','working-tree'),
            'source_workflow_run_url':report['workflow_run_url'],'candidate_cases':20,
            'candidate_reference_kind':'COMSOL-sourced linear modal response; native transient cases are separately labeled',
            'scientific_accuracy_certified':False,'files':entries},indent=2)+'\n')
    digest=sha(archive)
    (args.output/'SHA256SUMS-comsol.txt').write_text(digest+'  '+archive.name+'\n')
    print(json.dumps({'archive':archive.name,'bytes':archive.stat().st_size,'sha256':digest,'verified_files':len(entries)}))


if __name__=='__main__':main()
