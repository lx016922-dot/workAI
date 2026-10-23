"""Predeclared geometry-separated high-order dataset with per-case admission.

Every geometry gets h, p and time checks using identical physical inputs.
Failed or interrupted numerical jobs are retained and are never auto-retried.
"""
import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from solid.schema import default_scene,validate_scene
from .campaign import atomic,now
from .compare import compare,spectral,digest
from .field_compare import compare_fields


def plan(seed=2026102311):
    rng=np.random.default_rng(seed);cases=[]
    for kind in ['plate','pipe','hole','notch']:
        for variant,split in enumerate(['train','train','validation','test']):
            s=default_scene(kind);g=s['geometry'];length=float(rng.uniform(8.,12.));width=float(rng.uniform(3.,4.));thick=float(rng.uniform(1.5,2.1));ro=float(rng.uniform(2.2,2.8));ri=ro-float(rng.uniform(.85,1.1))
            g.update(length_mm=length,width_mm=width,thickness_mm=thick,outer_radius_mm=ro,inner_radius_mm=ri,hole_radius_mm=float(rng.uniform(.55,.8)),hole_x_mm=length*float(rng.uniform(.44,.56)),hole_y_mm=width*float(rng.uniform(.43,.57)),notch_x_mm=length*float(rng.uniform(.44,.57)),notch_width_mm=float(rng.uniform(.7,1.2)),notch_depth_mm=thick*float(rng.uniform(.22,.4)))
            right=bool(rng.integers(2));fixed=variant%2==1;y=(ro+ri)/2 if kind=='pipe' else width/2;z=0. if kind=='pipe' else thick/2
            s['source'].update(face='xmax' if right else 'xmin',center_mm=[length if right else 0,0. if kind=='pipe' else width/2,0. if kind=='pipe' else thick/2],direction=[-1. if right else 1.,float(rng.uniform(-.15,.15)),float(rng.uniform(-.12,.12))],amplitude_mpa=float(rng.uniform(.6,1.5)),frequency_mhz=float(rng.uniform(.18,.28)),cycles=float(rng.uniform(1.5,2.1)),delay_us=float(rng.uniform(.05,.5)),phase_deg=float(rng.uniform(-75,75)))
            s['boundary']['fixed_face']=('xmin' if right else 'xmax') if fixed else 'none';s['acquisition'].update(duration_us=14.,sample_step_us=.05);s['numerics']['h_mm']=.55
            for rx,fraction in zip(s['receivers'],[.23,.79]):rx['point_mm']=[length*fraction,y,z]
            cid=f'spectral-{kind}-{variant}';s['name']=cid;s=validate_scene(s);identity=hashlib.sha256(json.dumps(s['geometry'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            cases.append({'case_id':cid,'split':split,'geometry_group_sha256':identity,'scene':s,'phase':'planned'})
    assert len({c['geometry_group_sha256'] for c in cases})==16
    return {'schema':'workai.spectral_dataset.v1','phase':'planned','seed':seed,'created_at_utc':now(),'split_counts':{'train':8,'validation':4,'test':4},'threshold':.05,'cases':cases,
            'test_scope':'Four new geometries withheld from network fitting and checkpoint selection; numerical-reference checks use a fixed admission rule.',
            'accepted_for_solver_routing':False,'scope':'Bounded aluminum 8–12 mm, .18–.28 MHz, 14 us; no arbitrary shape accuracy claim.'}


def run(args):
    root=args.output;lock=threading.RLock();source=Path(__file__).resolve().parents[1]
    if args.resume:
        manifest=json.loads((root/'manifest.json').read_text());input_manifest=json.loads((root/'planned-inputs.json').read_text())
        assert digest(root/'planned-inputs.json')==manifest['planned_inputs_sha256']
        for key,sha in manifest['source_sha256'].items():assert digest(root/'source_snapshot'/key)==sha
    else:
        root.mkdir(parents=True,exist_ok=False);manifest=plan();atomic(root/'planned-inputs.json',manifest);manifest['planned_inputs_sha256']=digest(root/'planned-inputs.json');hashes={}
        for module in ['solid','spectral3d']:
            target=root/'source_snapshot'/module;target.mkdir(parents=True)
            for file in (source/module).glob('*.py'):shutil.copyfile(file,target/file.name);hashes[module+'/'+file.name]=digest(file)
        manifest['source_sha256']=hashes;manifest['workers']=args.workers;atomic(root/'manifest.json',manifest)
    def publish():
        with lock:manifest['updated_at_utc']=now();atomic(root/'manifest.json',manifest)
    def work(case):
        if case['phase'] in ['completed','failed']:return
        case['phase']='running';publish();kind=case['scene']['geometry']['kind'];integrator='newmark' if kind=='hole' else 'verlet'
        try:
            levels={}
            for name,degree,h,substeps,cfl,fields in [('base',3,2.,4,.3,True),('space',3,1.4,4,.3,True),('degree',4,2.,4,.3,False),('time',3,2.,8,.15,False)]:
                directory=root/case['case_id']/name;path=directory/'manifest.json';levels[name]=path
                if directory.exists():
                    if path.exists() and json.loads(path.read_text())['phase']=='completed':spectral(path);continue
                    raise RuntimeError('Prior incomplete job needs inspection: '+str(directory))
                directory.parent.mkdir(parents=True,exist_ok=True)
                command=[sys.executable,'-m','spectral3d.campaign','--campaign',str(root/'planned-inputs.json'),'--output',str(directory),'--cases',case['case_id'],'--degree',str(degree),'--h',str(h),'--integrator',integrator,'--substeps',str(substeps),'--cfl',str(cfl)]
                if fields:command.append('--fields')
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
                with (directory.parent/(name+'.log')).open('x') as stream:subprocess.run(command,cwd=root/'source_snapshot',env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                case.setdefault('completed_levels',[]).append(name);publish()
            reports={}
            baseline=spectral(levels['base'])
            for name in ['space','degree','time']:reports[name]=compare(baseline,spectral(levels[name]));reports[name].update(left_manifest_sha256=digest(levels['base']),right_manifest_sha256=digest(levels[name]))
            reports['field']=compare_fields(levels['base'],levels['space']);atomic(root/case['case_id']/'quality.json',reports)
            guards=all(d['energy_balance_within_1_percent'] and d['max_strain']<=.001 and d['min_J']>0 for d in baseline[case['case_id']]['diagnostics'])
            case.update(phase='completed',integrator=integrator,reference_checks_passed=guards and reports['field']['all_checks_pass'] and all(reports[k]['all_probe_checks_pass'] for k in ['space','degree','time']),quality_sha256=digest(root/case['case_id']/'quality.json'),levels={name:{'manifest':str(path.relative_to(root)),'sha256':digest(path)} for name,path in levels.items()})
            publish();print(json.dumps({'case_id':case['case_id'],'split':case['split'],'reference_checks_passed':case['reference_checks_passed']}),flush=True)
        except BaseException as error:case.update(phase='failed',error=str(error));publish();print(json.dumps({'case_id':case['case_id'],'error':str(error)}),flush=True)
    manifest['phase']='generating';publish()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(work,manifest['cases']))
    manifest['phase']='completed' if all(c['phase']=='completed' for c in manifest['cases']) else 'completed_with_failures'
    manifest['all_reference_checks_passed']=all(c.get('reference_checks_passed',False) for c in manifest['cases']);publish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,choices=[1,2,3],default=3);p.add_argument('--resume',action='store_true');run(p.parse_args())
