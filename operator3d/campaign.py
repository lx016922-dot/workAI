"""Create a fixed case-level split, then run three immutable controls per case.

This is explicitly invoked data generation, never an environment startup step.
Existing simulations, paired identities and the earlier 2-D training stay intact.
"""
import argparse,copy,hashlib,json,shutil,subprocess,sys,uuid
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from solid.schema import default_scene,validate_scene

def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False));temporary.replace(path)

def geometry_identity(scene):
    g=scene['geometry'];keys={'plate':['length_mm','width_mm','thickness_mm'],'pipe':['length_mm','outer_radius_mm','inner_radius_mm'],'hole':['length_mm','width_mm','thickness_mm','hole_radius_mm','hole_x_mm','hole_y_mm'],'notch':['length_mm','width_mm','thickness_mm','notch_x_mm','notch_width_mm','notch_depth_mm']}[g['kind']]
    return hashlib.sha256(json.dumps({k:g[k] for k in ['kind']+keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()

def case_plan():
    cases=[]
    for kind in ['plate','pipe','hole','notch']:
        for variant in range(4):
            s=default_scene(kind);g=s['geometry'];length=[10.,11.2,12.4,14.][variant];width=[3.5,3.8,4.2,4.5][variant];thickness=[1.6,1.8,2.1,2.4][variant]
            g.update(length_mm=length,width_mm=width,thickness_mm=thickness,outer_radius_mm=[2.3,2.5,2.7,3.][variant],inner_radius_mm=[1.3,1.45,1.6,1.9][variant],hole_radius_mm=[.45,.55,.65,.75][variant],hole_x_mm=length*(.46+.015*variant),hole_y_mm=width*.5,notch_x_mm=length*.55,notch_width_mm=[.7,.85,1.,1.15][variant],notch_depth_mm=[.35,.48,.58,.7][variant])
            source_max=variant in [2,3];center=[length if source_max else 0.,0. if kind=='pipe' else width/2,0. if kind=='pipe' else thickness/2]
            s['source'].update(face='xmax' if source_max else 'xmin',center_mm=center,frequency_mhz=[.18,.205,.225,.25][variant],cycles=[1.5,1.75,2.,2.][variant],amplitude_mpa=[.75,1.,1.25,1.5][variant],delay_us=[0.,.15,.3,.45][variant],phase_deg=[0.,10.,-10.,20.][variant],direction=[[1.,.1,0.],[1.,0.,.08],[-1.,.12,-.04],[-1.,-.15,.06]][variant])
            s['boundary']['fixed_face']='xmin' if source_max else 'none';s['numerics']['h_mm']=.65
            y=(g['inner_radius_mm']+g['outer_radius_mm'])/2 if kind=='pipe' else width/2;z=0. if kind=='pipe' else thickness/2
            s['receivers']=[{'name':'P1','point_mm':[.25*length,y,z],'direction':[1.,0.,0.]},{'name':'P2','point_mm':[.8*length,y,z],'direction':[1.,2.,-1.]}]
            s['name']=f'3-D operator pilot · {kind} · {variant}';s=validate_scene(s)
            cases.append({'case_id':f'{kind}-{variant}','split':{0:'train',1:'validation',2:'test',3:'train'}[variant],'geometry_group_sha256':geometry_identity(s),'scene':s})
    assert len({c['geometry_group_sha256'] for c in cases})==16
    return cases

def run(args):
    root=args.root;directory=args.campaign
    for kind in ['jobs','studies']:
        for p in (root/'solid'/kind).glob('*/status.json'):
            if json.loads(p.read_text())['phase'] not in ['completed','failed']:raise ValueError('An existing solid calculation is active; do not duplicate it.')
    if args.resume:
        manifest=json.loads((directory/'manifest.json').read_text());expected=case_plan()
        assert [{k:c[k] for k in ['case_id','split','geometry_group_sha256','scene']} for c in manifest['cases']]==expected
    else:
        directory.mkdir(parents=True,exist_ok=False);frozen=directory/'source_snapshot';(frozen/'solid').mkdir(parents=True)
        for p in (REPO/'solid').glob('*.py'):shutil.copyfile(p,frozen/'solid'/p.name)
        shutil.copyfile(Path(__file__),frozen/'campaign.py')
        manifest={'schema':'workai.operator3d.campaign.v1','complete':False,'case_count':16,'split_counts':{'train':8,'validation':4,'test':4},'split_unit':'whole geometry; all amplitudes, laws, mesh/time repeats must remain in this group','seed':20261023,'discretization_verified':False,'scope':'Bounded 4-family aluminum pilot, not arbitrary-CAD generalization or an independently converged full-field dataset.','cases':case_plan()}
        write_json(directory/'manifest.json',manifest)
    frozen=directory/'source_snapshot'
    for case in manifest['cases']:
        reference=None
        for role,law in [('linear','linear'),('geometric','stvk'),('material','murnaghan')]:
            controls=case.setdefault('controls',[]);saved=next((r for r in controls if r['role']==role),None)
            if saved:
                job=root/'solid/jobs'/saved['job_id'];status=json.loads((job/'status.json').read_text())
                if status['phase']!='completed':raise ValueError('Recorded control is not completed; inspect its existing status before any retry.')
                data=(job/'result.json').read_bytes();digest=hashlib.sha256(data).hexdigest();assert digest==status['result_sha256'];saved['result_sha256']=digest
            else:
                job=root/'solid/jobs'/('solid-'+uuid.uuid4().hex);job.mkdir(parents=True);scene=copy.deepcopy(case['scene']);scene['constitutive']=law
                write_json(job/'scene.json',scene);saved={'role':role,'job_id':job.name};controls.append(saved);write_json(directory/'manifest.json',manifest)
                command=[sys.executable,'-c','from solid.worker import run;import sys;run(sys.argv[1],sys.argv[2] or None,"controls","float64")',str(job),str(reference) if reference else '']
                with (job/'worker.log').open('x') as log:subprocess.run(command,cwd=frozen,stdout=log,stderr=subprocess.STDOUT,check=True)
                status=json.loads((job/'status.json').read_text());data=(job/'result.json').read_bytes();assert status['phase']=='completed';saved['result_sha256']=hashlib.sha256(data).hexdigest();assert saved['result_sha256']==status['result_sha256']
            if role=='linear':reference=job
            write_json(directory/'manifest.json',manifest)
        print(json.dumps({'case_completed':case['case_id'],'split':case['split'],'controls':3}),flush=True)
    manifest['complete']=True;write_json(directory/'manifest.json',manifest);print(json.dumps({'campaign_complete':True,'new_solves':48,'independent_cases':16,'manifest':str(directory/'manifest.json')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/workspace/workai-remote'));p.add_argument('--campaign',type=Path,required=True);p.add_argument('--resume',action='store_true');run(p.parse_args())
