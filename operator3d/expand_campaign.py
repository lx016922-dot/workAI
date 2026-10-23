"""A new development split after the first pilot has been inspected.

All 16 previously inspected cases become development training cases. New
validation/test geometries are generated before any of their fields are read.
Both sets contain free/fixed boundaries and both loading ends in every family.
"""
import argparse,copy,hashlib,json,shutil,subprocess,sys,uuid
from pathlib import Path
import numpy as np
from solid.schema import validate_scene
from .campaign import geometry_identity,write_json

def expand(source):
    manifest=copy.deepcopy(source);assert manifest['complete'];old=manifest['cases'];new=[];rng=np.random.default_rng(2026102302)
    for case in old:case['previous_pilot_split']=case['split'];case['split']='train'
    for kind in ['plate','pipe','hole','notch']:
        template=next(c['scene'] for c in old if c['scene']['geometry']['kind']==kind)
        for split in ['validation','test']:
            for fixed in [False,True]:
                s=copy.deepcopy(template);g=s['geometry'];length=float(rng.uniform(10,14));width=float(rng.uniform(3.5,4.5));thickness=float(rng.uniform(1.6,2.4));radius=float(rng.uniform(2.3,3))
                g.update(length_mm=length,width_mm=width,thickness_mm=thickness,outer_radius_mm=radius,inner_radius_mm=radius-float(rng.uniform(.95,1.1)),hole_radius_mm=float(rng.uniform(.45,.75)),hole_x_mm=length*float(rng.uniform(.46,.51)),hole_y_mm=width/2,notch_x_mm=length*float(rng.uniform(.5,.6)),notch_width_mm=float(rng.uniform(.7,1.15)),notch_depth_mm=float(rng.uniform(.35,.7)))
                s['source'].update(face='xmax' if fixed else 'xmin',center_mm=[length if fixed else 0.,0. if kind=='pipe' else width/2,0. if kind=='pipe' else thickness/2],direction=[-1. if fixed else 1.,float(rng.uniform(-.15,.15)),float(rng.uniform(-.08,.08))],frequency_mhz=float(rng.uniform(.18,.25)),cycles=float(rng.uniform(1.5,2)),amplitude_mpa=float(rng.uniform(.75,1.5)),delay_us=float(rng.uniform(0,.45)),phase_deg=float(rng.uniform(-10,20)))
                s['boundary']['fixed_face']='xmin' if fixed else 'none';y=(g['inner_radius_mm']+radius)/2 if kind=='pipe' else width/2;z=0. if kind=='pipe' else thickness/2
                for receiver,fraction in zip(s['receivers'],[.25,.8]):receiver['point_mm']=[fraction*length,y,z]
                id=f'{kind}-{split}-'+('fixed' if fixed else 'free');s['name']='3-D second pilot · '+id;s=validate_scene(s)
                new.append({'case_id':id,'split':split,'geometry_group_sha256':geometry_identity(s),'scene':s})
    assert len({c['geometry_group_sha256'] for c in old+new})==32
    manifest.update(complete=False,case_count=32,split_counts={'train':16,'validation':8,'test':8},cases=old+new,seed=2026102302,previous_pilot_cases_now_development_training=True,fresh_validation_and_test_groups=True,scope='32-case bounded aluminum pilot. Prior inspected test cases are now training; 16 fresh balanced validation/test cases. Both source end and fixity co-vary; no claim of independent coverage of every parameter combination.')
    return manifest

def main(args):
    root=args.root;source_path=args.source;old=json.loads(source_path.read_text());directory=args.campaign;directory.mkdir(parents=True,exist_ok=False)
    for kind in ['jobs','studies']:
        for p in (root/'solid'/kind).glob('*/status.json'):
            if json.loads(p.read_text())['phase'] not in ['completed','failed']:raise ValueError('Existing active solve; do not duplicate it.')
    manifest=expand(old);manifest['parent_campaign_sha256']=hashlib.sha256(source_path.read_bytes()).hexdigest();frozen=directory/'source_snapshot';shutil.copytree(source_path.parent/'source_snapshot',frozen);shutil.copyfile(Path(__file__),frozen/'expand_campaign.py');write_json(directory/'manifest.json',manifest)
    for case in manifest['cases']:
        if case.get('controls'):
            for control in case['controls']:
                job=root/'solid/jobs'/control['job_id'];status=json.loads((job/'status.json').read_text());assert status['phase']=='completed' and hashlib.sha256((job/'result.json').read_bytes()).hexdigest()==control['result_sha256']==status['result_sha256']
            continue
        reference=None;case['controls']=[]
        for role,law in [('linear','linear'),('geometric','stvk'),('material','murnaghan')]:
            job=root/'solid/jobs'/('solid-'+uuid.uuid4().hex);job.mkdir();scene=copy.deepcopy(case['scene']);scene['constitutive']=law;write_json(job/'scene.json',scene);control={'role':role,'job_id':job.name};case['controls'].append(control);write_json(directory/'manifest.json',manifest)
            command=[sys.executable,'-c','from solid.worker import run;import sys;run(sys.argv[1],sys.argv[2] or None,"controls","float64")',str(job),str(reference) if reference else '']
            with (job/'worker.log').open('x') as log:subprocess.run(command,cwd=frozen,stdout=log,stderr=subprocess.STDOUT,check=True)
            status=json.loads((job/'status.json').read_text());assert status['phase']=='completed';control['result_sha256']=hashlib.sha256((job/'result.json').read_bytes()).hexdigest();assert control['result_sha256']==status['result_sha256'];write_json(directory/'manifest.json',manifest)
            if role=='linear':reference=job
        print(json.dumps({'case_completed':case['case_id'],'split':case['split']}),flush=True)
    manifest['complete']=True;write_json(directory/'manifest.json',manifest);print(json.dumps({'campaign_complete':True,'new_solves':48,'cases':32}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/workspace/workai-remote'));p.add_argument('--source',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);main(p.parse_args())
