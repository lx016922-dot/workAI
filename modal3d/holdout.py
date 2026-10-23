"""Predeclared wider-domain holdouts; do not inspect labels during fitting."""
import argparse,copy,hashlib,json,shutil,subprocess,sys,uuid
from pathlib import Path
import numpy as np
from solid.schema import validate_scene
from operator3d.campaign import geometry_identity,write_json

def plan(source):
    result=copy.deepcopy(source);assert result['complete'];rng=np.random.default_rng(2026102307)
    for c in result['cases']:
        c['previous_v3_split']=c['split']
        if c['split']=='test':c['split']='train'
    cases=[]
    for k,kind in enumerate(['plate','pipe','hole','notch']):
        base=next(c['scene'] for c in result['cases'] if c['scene']['geometry']['kind']==kind)
        for fixed in [False,True]:
            s=copy.deepcopy(base);g=s['geometry'];length=float(rng.uniform(9,16));width=float(rng.uniform(3,5));thickness=float(rng.uniform(1.4,2.7));radius=float(rng.uniform(2.1,3.4));hole=float(rng.uniform(.4,.8));right=bool(k%2)^fixed
            g.update(length_mm=length,width_mm=width,thickness_mm=thickness,outer_radius_mm=radius,inner_radius_mm=radius-float(rng.uniform(.85,1.2)),hole_radius_mm=hole,hole_x_mm=length*float(rng.uniform(.42,.58)),hole_y_mm=float(rng.uniform(hole+.25,width-hole-.25)),notch_x_mm=length*float(rng.uniform(.4,.65)),notch_width_mm=float(rng.uniform(.65,1.3)),notch_depth_mm=float(rng.uniform(.3,.9)))
            s['source'].update(face='xmax' if right else 'xmin',center_mm=[length if right else 0.,0. if kind=='pipe' else width/2,0. if kind=='pipe' else thickness/2],direction=[-1. if right else 1.,float(rng.uniform(-.2,.2)),float(rng.uniform(-.12,.12))],frequency_mhz=float(rng.uniform(.15,.3)),cycles=float(rng.uniform(1.5,2.5)),amplitude_mpa=float(rng.uniform(.5,2)),delay_us=float(rng.uniform(0,.7)),phase_deg=float(rng.uniform(-80,80)))
            s['boundary']['fixed_face']=('xmin' if right else 'xmax') if fixed else 'none';s['acquisition']['duration_us']=20.;s['numerics']['h_mm']=.55
            y=(radius+g['inner_radius_mm'])/2 if kind=='pipe' else width/2;z=0. if kind=='pipe' else thickness/2
            for rx,fraction in zip(s['receivers'],[.23,.81]):rx['point_mm']=[fraction*length,y,z]
            cid='modal-test-'+kind+'-'+('fixed' if fixed else 'free');s['name']='Predeclared modal holdout · '+cid;s=validate_scene(s);cases.append({'case_id':cid,'split':'test','geometry_group_sha256':geometry_identity(s),'scene':s})
    result.update(complete=False,case_count=64,split_counts={'train':48,'validation':8,'test':8},cases=result['cases']+cases,seed=2026102307,scope='Fresh 8-case wider-parameter modal/energy holdout, 9-16 mm, .15-.30 MHz, .5-2 MPa, 20 us. Existing 56 cases are consumed development data; prior validation retained. No arbitrary-shape reliability claim.')
    assert len({c['geometry_group_sha256'] for c in result['cases']})==64;return result

def main(args):
    root=args.root;out=args.output
    for p in (root/'solid/jobs').glob('*/status.json'):
        if json.loads(p.read_text())['phase'] not in ['completed','failed']:raise ValueError('An existing solve is active.')
    out.mkdir(parents=True,exist_ok=False);m=plan(json.loads(args.source.read_text()));m['parent_campaign_sha256']=hashlib.sha256(args.source.read_bytes()).hexdigest();shutil.copytree(args.source.parent/'source_snapshot',out/'source_snapshot');shutil.copyfile(Path(__file__),out/'source_snapshot/modal-holdout.py');write_json(out/'manifest.json',m)
    for case in m['cases']:
        if case.get('controls'):continue
        reference=None;case['controls']=[]
        for role,law in [('linear','linear'),('geometric','stvk'),('material','murnaghan')]:
            job=root/'solid/jobs'/('solid-'+uuid.uuid4().hex);job.mkdir();s=copy.deepcopy(case['scene']);s['constitutive']=law;write_json(job/'scene.json',s);control={'role':role,'job_id':job.name};case['controls'].append(control);write_json(out/'manifest.json',m)
            command=[sys.executable,'-c','from solid.worker import run;import sys;run(sys.argv[1],sys.argv[2] or None,"controls","float64")',str(job),str(reference) if reference else '']
            with (job/'worker.log').open('x') as log:subprocess.run(command,cwd=out/'source_snapshot',stdout=log,stderr=subprocess.STDOUT,check=True)
            status=json.loads((job/'status.json').read_text());assert status['phase']=='completed';control['result_sha256']=hashlib.sha256((job/'result.json').read_bytes()).hexdigest();assert control['result_sha256']==status['result_sha256'];write_json(out/'manifest.json',m)
            if role=='linear':reference=job
        print(json.dumps({'case_completed':case['case_id']}),flush=True)
    m['complete']=True;write_json(out/'manifest.json',m);print(json.dumps({'complete':True,'new_solves':24}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/workspace/workai-remote'));p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
