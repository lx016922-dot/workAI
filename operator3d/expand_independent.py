"""Predeclared independent excitation/geometry combinations and fresh test set.

V2 tests are explicitly consumed development data. Its 8 validation geometries
remain validation. The 8 new tests must not be loaded until weights are frozen.
"""
import argparse,copy,hashlib,json,shutil,subprocess,sys,uuid
from pathlib import Path
import numpy as np
from solid.schema import validate_scene
from .campaign import geometry_identity,write_json

def expand(source):
    m=copy.deepcopy(source);assert m['complete'];rng=np.random.default_rng(2026102304)
    for c in m['cases']:
        c['previous_v2_split']=c['split']
        if c['split']=='test':c['split']='train'
    new=[]
    for family,kind in enumerate(['plate','pipe','hole','notch']):
        template=next(c['scene'] for c in m['cases'] if c['scene']['geometry']['kind']==kind)
        combinations=[('train',fixed,end) for fixed in [False,True] for end in [False,True]]
        combinations += [('test',False,bool(family%2)),('test',True,not bool(family%2))]
        for split,fixed,source_max in combinations:
            s=copy.deepcopy(template);g=s['geometry'];length=float(rng.uniform(10,14));width=float(rng.uniform(3.5,4.5));thickness=float(rng.uniform(1.6,2.4));radius=float(rng.uniform(2.3,3))
            g.update(length_mm=length,width_mm=width,thickness_mm=thickness,outer_radius_mm=radius,inner_radius_mm=radius-float(rng.uniform(.95,1.1)),hole_radius_mm=float(rng.uniform(.45,.75)),hole_x_mm=length*float(rng.uniform(.46,.51)),hole_y_mm=width/2,notch_x_mm=length*float(rng.uniform(.5,.6)),notch_width_mm=float(rng.uniform(.7,1.15)),notch_depth_mm=float(rng.uniform(.35,.7)))
            s['source'].update(face='xmax' if source_max else 'xmin',center_mm=[length if source_max else 0.,0. if kind=='pipe' else width/2,0. if kind=='pipe' else thickness/2],direction=[-1. if source_max else 1.,float(rng.uniform(-.15,.15)),float(rng.uniform(-.08,.08))],frequency_mhz=float(rng.uniform(.18,.25)),cycles=float(rng.uniform(1.5,2)),amplitude_mpa=float(rng.uniform(.75,1.5)),delay_us=float(rng.uniform(0,.45)),phase_deg=float(rng.uniform(-10,20)))
            s['boundary']['fixed_face']=('xmin' if source_max else 'xmax') if fixed else 'none'
            y=(g['inner_radius_mm']+radius)/2 if kind=='pipe' else width/2;z=0. if kind=='pipe' else thickness/2
            for rx,fraction in zip(s['receivers'],[.25,.8]):rx['point_mm']=[fraction*length,y,z]
            cid=f'{kind}-v3-{split}-'+('fixed' if fixed else 'free')+('-right' if source_max else '-left');s['name']='Independent geometry/excitation · '+cid;s=validate_scene(s)
            new.append({'case_id':cid,'split':split,'geometry_group_sha256':geometry_identity(s),'scene':s})
    m.update(complete=False,case_count=56,split_counts={'train':40,'validation':8,'test':8},cases=m['cases']+new,seed=2026102304,development_only=False,scope='56 parameterized aluminum geometries. Old V2 tests become development training; V2 validation is retained. Additional training covers either loading end with free or opposite fixed boundary. Eight new tests reserved until model freeze. No arbitrary-CAD or label-convergence claim.')
    assert len({c['geometry_group_sha256'] for c in m['cases']})==56
    return m

def main(args):
    root=args.root;directory=args.campaign;source=args.source
    for kind in ['jobs','studies']:
        for p in (root/'solid'/kind).glob('*/status.json'):
            if json.loads(p.read_text())['phase'] not in ['completed','failed']:raise ValueError('An existing solve is active; inspect it instead of duplicating.')
    expected=expand(json.loads(source.read_text()))
    if args.resume:
        m=json.loads((directory/'manifest.json').read_text())
        assert [{k:c[k] for k in ['case_id','split','geometry_group_sha256','scene']} for c in m['cases']]==[{k:c[k] for k in ['case_id','split','geometry_group_sha256','scene']} for c in expected['cases']]
    else:
        directory.mkdir(parents=True,exist_ok=False);m=expected;m['parent_campaign_sha256']=hashlib.sha256(source.read_bytes()).hexdigest();shutil.copytree(source.parent/'source_snapshot',directory/'source_snapshot');shutil.copyfile(Path(__file__),directory/'source_snapshot/expand_independent.py');write_json(directory/'manifest.json',m)
    for case in m['cases']:
        reference=None
        for role,law in [('linear','linear'),('geometric','stvk'),('material','murnaghan')]:
            controls=case.setdefault('controls',[]);saved=next((c for c in controls if c['role']==role),None)
            if saved:
                job=root/'solid/jobs'/saved['job_id'];status=json.loads((job/'status.json').read_text());assert status['phase']=='completed'
                digest=hashlib.sha256((job/'result.json').read_bytes()).hexdigest();assert digest==status['result_sha256'];assert saved.get('result_sha256',digest)==digest;saved['result_sha256']=digest
            else:
                job=root/'solid/jobs'/('solid-'+uuid.uuid4().hex);job.mkdir();s=copy.deepcopy(case['scene']);s['constitutive']=law;write_json(job/'scene.json',s);saved={'role':role,'job_id':job.name};controls.append(saved);write_json(directory/'manifest.json',m)
                command=[sys.executable,'-c','from solid.worker import run;import sys;run(sys.argv[1],sys.argv[2] or None,"controls","float64")',str(job),str(reference) if reference else '']
                with (job/'worker.log').open('x') as log:subprocess.run(command,cwd=directory/'source_snapshot',stdout=log,stderr=subprocess.STDOUT,check=True)
                status=json.loads((job/'status.json').read_text());assert status['phase']=='completed';saved['result_sha256']=hashlib.sha256((job/'result.json').read_bytes()).hexdigest();assert saved['result_sha256']==status['result_sha256']
            if role=='linear':reference=job
            write_json(directory/'manifest.json',m)
        print(json.dumps({'case_complete':case['case_id'],'split':case['split']}),flush=True)
    m['complete']=True;write_json(directory/'manifest.json',m);print(json.dumps({'complete':True,'new_solves':72,'total_cases':56}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/workspace/workai-remote'));p.add_argument('--source',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--resume',action='store_true');main(p.parse_args())
