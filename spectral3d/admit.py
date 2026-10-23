"""Audited numerical admission and bounded refinement; never changes old jobs."""
import argparse,json,os,shutil,subprocess,sys,time,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .campaign import atomic,now
from .compare import spectral,compare,digest
from .field_compare import compare_fields


def run(args):
    reuse_roots=[];pending=([args.reuse] if args.reuse else [])+args.extra_reuse
    while pending:
        directory=pending.pop(0)
        if directory in reuse_roots:continue
        prior=json.loads((directory/'manifest.json').read_text())
        if prior['phase'] not in ['completed','superseded']:raise ValueError('The reused admission run must have stopped scheduling jobs.')
        reuse_roots.append(directory)
        if prior.get('reused_admission'):pending.append(Path(prior['reused_admission']))
    root=args.output;root.mkdir(parents=True,exist_ok=False);source=Path(__file__).resolve().parents[1];hashes={}
    for module in ['solid','spectral3d']:
        target=root/'source_snapshot'/module;target.mkdir(parents=True)
        for p in (source/module).glob('*.py'):shutil.copyfile(p,target/p.name);hashes[module+'/'+p.name]=digest(p)
    inputs=args.dataset/'planned-inputs.json';plan=json.loads(inputs.read_text());lock=threading.RLock();manifest={'schema':'workai.spectral_admission.v1','phase':'running','started_at_utc':now(),'source_manifest_sha256':None,'planned_inputs_sha256':digest(inputs),'source_sha256':hashes,'cases':[{'case_id':c['case_id'],'split':c['split'],'phase':'waiting'} for c in plan['cases']],'accepted_for_solver_routing':False,'refinement_rule':'Original p3 h2; if any 5% check fails, base p3 h1.4; if still failing, base p4 h1.4. No thresholds relaxed; all earlier results retained.'}
    def publish():
        with lock:manifest['updated_at_utc']=now();atomic(root/'manifest.json',manifest)
    manifest.update(workers=args.workers,reused_admission=str(args.reuse) if args.reuse else None,reused_admission_manifest_sha256=digest(args.reuse/'manifest.json') if args.reuse else None)
    manifest['reuse_sources']=[{'directory':str(p),'manifest_sha256':digest(p/'manifest.json')} for p in reuse_roots]
    if args.prefer_degree:manifest['refinement_rule']='For holes, try p4/h2 with p4/h1.4, p5/h2 and same-mesh time refinement before p4/h1.4 fallback. Other families retain p3/h1.4 then p4/h1.4. All 5% probe, component, harmonic and full-field checks retained.'
    def solve(case,label,p,h,cfl,substeps,reuse_labels=()):
        for previous in reuse_roots:
            for candidate in [label,*reuse_labels]:
                prior=previous/case['case_id']/candidate/'manifest.json'
                if prior.exists() and json.loads(prior.read_text())['phase']=='completed':
                    metadata=json.loads(prior.read_text());assert metadata['degree']==p and metadata['h_mm']==h and metadata['cfl']==cfl and metadata['substeps']==substeps
                    spectral(prior)
                    assert len(metadata['cases'])==1 and metadata['cases'][0]['case_id']==case['case_id'] and metadata['cases'][0]['scene']==case['scene']
                    for item in metadata['cases']:
                        for control in item['controls']:assert digest(prior.parent/control['directory']/'field.npz')==control['field_sha256']
                    return prior
        directory=root/case['case_id']/label
        cmd=[sys.executable,'-m','spectral3d.campaign','--campaign',str(inputs),'--output',str(directory),'--cases',case['case_id'],'--degree',str(p),'--h',str(h),'--cfl',str(cfl),'--integrator','newmark' if case['scene']['geometry']['kind']=='hole' else 'verlet','--substeps',str(substeps),'--fields']
        with (directory.parent/(label+'.log')).open('x') as log:subprocess.run(cmd,cwd=root/'source_snapshot',env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
        return directory/'manifest.json'
    def check(paths,output):
        baseline=spectral(paths['base']);reports={}
        for name in ['space','degree','time']:
            reports[name]=compare(baseline,spectral(paths[name]));reports[name].update(left_manifest_sha256=digest(paths['base']),right_manifest_sha256=digest(paths[name]))
        reports['field']=compare_fields(paths['base'],paths['space']);reports['comparison_source_sha256']=hashes;atomic(output,reports)
        guards=all(d['energy_balance_within_1_percent'] and d['max_strain']<=.001 and d['min_J']>0 for c in baseline.values() for d in c['diagnostics'])
        return guards and reports['field']['all_checks_pass'] and all(reports[k]['all_probe_checks_pass'] for k in ['space','degree','time'])
    def work(case,row):
        directory=root/case['case_id'];directory.mkdir();deadline=time.monotonic()+7200
        try:
            paths={key:args.dataset/case['case_id']/key/'manifest.json' for key in ['base','space','degree','time']}
            while True:
                ready=all(p.exists() and json.loads(p.read_text())['phase']=='completed' for p in paths.values())
                if ready:break
                if time.monotonic()>deadline:raise TimeoutError('Reference generation did not complete.')
                state=json.loads((args.dataset/'manifest.json').read_text());item=next(c for c in state['cases'] if c['case_id']==case['case_id'])
                if item['phase']=='failed':raise ValueError('Numerical job failed: '+item.get('error','unknown'))
                time.sleep(3)
            row['phase']='checking';publish();quality=directory/'original-quality.json';passed=check(paths,quality);row['attempts']=[{'quality_path':str(quality),'quality_sha256':digest(quality),'pass':passed}]
            degree_first=args.prefer_degree and case['scene']['geometry']['kind']=='hole'
            if not passed and degree_first:
                row.update(phase='refining',tier='degree4_h2');publish()
                paths={'base':solve(case,'degree-base',4,2.,.3,4),'space':solve(case,'degree-space',4,1.4,.3,4,reuse_labels=['tier1-degree']),'degree':solve(case,'degree-higher',5,2.,.3,4),'time':solve(case,'degree-time',4,2.,.15,8)}
                quality=directory/'degree-quality.json';passed=check(paths,quality);row['attempts'].append({'quality_path':str(quality),'quality_sha256':digest(quality),'pass':passed})
                if not passed:
                    base=paths['space'];row['tier']='degree4_h14';publish()
                    paths={'base':base,'space':solve(case,'tier2-space',4,1.,.3,4),'degree':solve(case,'tier2-degree',5,1.4,.3,4),'time':solve(case,'tier2-time',4,1.4,.15,8)}
                    quality=directory/'degree-fallback-quality.json';passed=check(paths,quality);row['attempts'].append({'quality_path':str(quality),'quality_sha256':digest(quality),'pass':passed})
            for tier,p in ([] if degree_first else [(1,3),(2,4)]):
                if passed:break
                row['phase']='refining';row['tier']=tier;publish()
                base=paths['space'] if tier==1 else paths['degree']
                paths={'base':base,'space':solve(case,f'tier{tier}-space',p,1.,.3,4),'degree':solve(case,f'tier{tier}-degree',p+1,1.4,.3,4),'time':solve(case,f'tier{tier}-time',p,1.4,.15,8)}
                quality=directory/f'tier{tier}-quality.json';passed=check(paths,quality);row['attempts'].append({'quality_path':str(quality),'quality_sha256':digest(quality),'pass':passed})
            row.update(phase='completed',reference_checks_passed=passed,base_manifest=str(paths['base']),base_manifest_sha256=digest(paths['base']),quality_path=str(quality),quality_sha256=digest(quality),levels={k:{'manifest':str(p),'sha256':digest(p)} for k,p in paths.items()})
            publish();print(json.dumps({'case_id':case['case_id'],'reference_checks_passed':passed,'tier':row.get('tier',0)}),flush=True)
        except BaseException as error:row.update(phase='failed',error=str(error),reference_checks_passed=False);publish();print(json.dumps({'case_id':case['case_id'],'error':str(error)}),flush=True)
    publish()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(lambda pair:work(*pair),zip(plan['cases'],manifest['cases'])))
    # Bind the final generation manifest only once the independent producer stops.
    while json.loads((args.dataset/'manifest.json').read_text())['phase']=='generating':time.sleep(3)
    manifest.update(phase='completed',source_manifest_sha256=digest(args.dataset/'manifest.json'),all_reference_checks_passed=all(c.get('reference_checks_passed',False) for c in manifest['cases']),completed_at_utc=now());publish()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,choices=[1,2,3],default=2);p.add_argument('--reuse',type=Path);p.add_argument('--extra-reuse',type=Path,action='append',default=[]);p.add_argument('--prefer-degree',action='store_true');run(p.parse_args())
