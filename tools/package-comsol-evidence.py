"""Preserve audited neural records plus COMSOL and physics training evidence."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import zipfile


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--supplement',action='store_true',help='Use a distinct asset name to retain the original release evidence')
    p.add_argument('--audit-subdir',default='candidate-matrix20',help='Committed audited candidate directory under research/comsol-validation')
    p.add_argument('--archive-label',default='',help='Optional distinct label for a later training evidence snapshot')
    p.add_argument('--baseline-input',type=Path,help='Optional complete same-plan baseline records')
    p.add_argument('--baseline-audit-subdir',default='',help='Independent audit of the paired baseline')
    args=p.parse_args()
    for value in [args.audit_subdir]+[v for v in [args.archive_label,args.baseline_audit_subdir] if v]:
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}',value):raise ValueError('Invalid evidence dataset or label')
    if args.audit_subdir!='candidate-matrix20' and not args.archive_label:
        raise ValueError('A different audited candidate requires a distinct archive label')
    repo=Path(__file__).resolve().parents[1];research=repo/'research/comsol-validation'
    report=json.loads((research/args.audit_subdir/'summary.json').read_text())
    if report['completed_cases']!=20 or not report['metrics_independently_recomputed']:
        raise ValueError('The complete independent audit must be committed first')
    if bool(args.baseline_input)!=bool(args.baseline_audit_subdir):
        raise ValueError('Baseline input and audit must be supplied together')
    baseline=None
    if args.baseline_input:
        if not args.archive_label:raise ValueError('Paired evidence requires a distinct archive label')
        baseline=json.loads((research/args.baseline_audit_subdir/'summary.json').read_text())
        if baseline['completed_cases']!=20 or not baseline['metrics_independently_recomputed']:
            raise ValueError('The complete baseline audit must be committed first')
        for key in ['plan_sha256','reference_identity','refinement_identity']:
            if baseline[key]!=report[key]:raise ValueError('Baseline and candidate comparisons differ: '+key)
    args.output.mkdir(parents=True,exist_ok=True)
    suffix='-'+args.archive_label if args.archive_label else '-supplement' if args.supplement else ''
    archive=args.output/('WorkAI-COMSOL-physics-evidence-0.3.0'+suffix+'.zip');entries=[]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        def add(path,name):
            z.write(path,name);entries.append({'path':name,'bytes':path.stat().st_size,'sha256':sha(path)})
        def add_records(audit,root,prefix):
            cases=set()
            for row in audit['rows']:
                case=row['case_id']
                if not re.fullmatch(r'(hann|gaussian|rectangular|chirp)-(01|03|05|10|20)',case) or case in cases:
                    raise ValueError('Invalid or duplicated case')
                cases.add(case)
                choices=[root/('comsol-linear-'+case)/case,root/case/case,root/case]
                present=[path for path in choices if (path/'metrics.json').exists()]
                if len(present)!=1:raise ValueError('Missing or ambiguous completed case: '+case)
                directory=present[0]
                for name,key in [('identity.json','identity_sha256'),('metrics.json','metrics_sha256'),('comsol-modal-reference.npz','comsol_modal_reference_sha256')]:
                    if sha(directory/name)!=row[key]:raise ValueError('Downloaded evidence differs from audit')
                identity=json.loads((directory/'identity.json').read_text())
                metrics=json.loads((directory/'metrics.json').read_text());prediction=metrics['prediction']
                if identity['weights_sha256']!=audit['weights_sha256'] or prediction['frames']!=25001 or prediction['duration_us']!=5000:
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
                for name in names:add(directory/name,prefix+'/'+case+'/'+name)
            if len(cases)!=20:raise ValueError('Incomplete case matrix')
        add_records(report,args.input,'candidate-cases')
        if baseline:add_records(baseline,args.baseline_input,'baseline-cases')
        for path in sorted(research.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.name.endswith('.decoded.csv'):continue
            if path.suffix.lower() in ['.pt','.pth','.safetensors']:raise ValueError('No new optimizer checkpoints in this archive')
            add(path,'research/'+path.relative_to(research).as_posix())
        # The evidence can be replayed with the already released offline
        # Python runtime, without downloading the current repository.
        for folder in ['graphwave','spectral3d','solid','tools']:
            for path in sorted((repo/folder).rglob('*.py')):
                if '__pycache__' not in path.parts:add(path,path.relative_to(repo).as_posix())
        add(repo/'offline/model/time-coefficients.json','offline/model/time-coefficients.json')
        # Locate labels by their immutable identity rather than assume a name.
        def reference_folder(identity):
            matches=[]
            for path in (research/'data').glob('*/identity.json'):
                if json.loads(path.read_text())==identity:matches.append(path.parent.name)
            if len(matches)!=1:raise ValueError('Missing or ambiguous reference source')
            return matches[0]
        ref=reference_folder(report['reference_identity']);fine=reference_folder(report['refinement_identity'])
        plans=[path for path in (repo/'research').rglob('*.json') if sha(path)==report['plan_sha256']]
        if not plans:raise ValueError('Frozen plan is missing')
        plan='frozen-plan.json';add(sorted(plans)[0],plan)
        instructions=['WorkAI research evidence: offline audit replay',
            'Use Python from the released Windows/Linux offline package (NumPy 2.2.6, SciPy 1.15.3, Numba 0.63.1).',
            'Run from the extracted evidence directory. No server or network connection is required.',
            'These checks compare complete neural records with COMSOL-sourced linear modal references.',
            'They do not certify native nonlinear transient accuracy. Original audit reports remain unchanged.','']
        candidate_weights_path=None
        for audit,folder,prefix in [(report,args.audit_subdir,'candidate-cases')]+([(baseline,args.baseline_audit_subdir,'baseline-cases')] if baseline else []):
            weights=[path for path in (research/'training').glob('*/candidate-weights.npz') if sha(path)==audit['weights_sha256']]
            if not weights:raise ValueError('Missing frozen weights')
            weights.sort()
            if prefix=='candidate-cases':candidate_weights_path='research/'+weights[0].relative_to(research).as_posix()
            instructions.append('python tools/summarize-comsol-linear-validation.py --input '+prefix+' --plan '+plan+
                ' --weights research/'+weights[0].relative_to(research).as_posix()+' --reference research/data/'+ref+
                ' --check-reference research/data/'+fine+' --output replay-'+folder+
                ' --workflow-run '+audit['workflow_run_url'].rsplit('/',1)[-1])
        instructions.extend(['','Optional: compute a new complete 5 ms research prediction locally (may take tens of minutes).',
            'This does not install the candidate as the default application model or certify its accuracy.',
            'python tools/run-comsol-linear-case.py --case hann-01 --plan '+plan+' --weights '+candidate_weights_path+
            ' --reference research/data/'+ref+' --check-reference research/data/'+fine+' --output new-research-predictions'])
        content=('\n'.join(instructions)+'\n').encode('utf-8');name='REPRODUCE.txt'
        z.writestr(name,content);entries.append({'path':name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
        z.writestr('MANIFEST.json',json.dumps({'schema':'workai.comsol_physics_evidence.v1',
            'repository_source_commit':os.environ.get('WORKAI_EVIDENCE_COMMIT','working-tree'),
            'supplemental_snapshot':bool(args.supplement or args.archive_label),
            'audited_candidate_subdirectory':args.audit_subdir,'archive_label':args.archive_label,
            'source_workflow_run_url':report['workflow_run_url'],'candidate_cases':20,
            'baseline_cases':20 if baseline else 0,
            'audited_baseline_subdirectory':args.baseline_audit_subdir or None,
            'baseline_workflow_run_url':baseline['workflow_run_url'] if baseline else None,
            'offline_reproduction_source_included':True,
            'candidate_reference_kind':'COMSOL-sourced linear modal response; native transient cases are separately labeled',
            'scientific_accuracy_certified':False,'files':entries},indent=2)+'\n')
    with zipfile.ZipFile(archive) as verified:
        for entry in entries:
            if verified.getinfo(entry['path']).file_size!=entry['bytes']:
                raise ValueError('Archived file length differs from manifest')
            with verified.open(entry['path']) as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=entry['sha256']:
                    raise ValueError('Archived bytes differ from manifest: '+entry['path'])
    digest=sha(archive)
    (args.output/('SHA256SUMS-comsol'+suffix+'.txt')).write_text(digest+'  '+archive.name+'\n')
    print(json.dumps({'archive':archive.name,'bytes':archive.stat().st_size,'sha256':digest,'verified_files':len(entries),
                      'archive_contents_read_back_and_verified':True}))


if __name__=='__main__':main()
