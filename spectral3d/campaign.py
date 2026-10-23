"""Immutable reference jobs with explicit solver version, no neural fitting."""
import argparse
import copy
import hashlib
import json
import shutil
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from solid.schema import validate_scene
from .mesh import make_mesh
from .solver import solve


def now():return datetime.now(timezone.utc).isoformat()
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2));tmp.replace(path)


def run(args):
    source=json.loads(args.campaign.read_text());cases=[c for c in source['cases'] if c['case_id'] in args.cases]
    if len(cases)!=len(set(args.cases)):raise ValueError('Case selection did not match.')
    args.output.mkdir(parents=True,exist_ok=False);snapshot=args.output/'source_snapshot';snapshot.mkdir()
    hashes={}
    for module in ['solid','spectral3d']:
        (snapshot/module).mkdir()
        for file in (Path(__file__).resolve().parents[1]/module).glob('*.py'):
            shutil.copyfile(file,snapshot/module/file.name);hashes[module+'/'+file.name]=digest(file)
    manifest={'schema':'workai.spectral_reference_campaign.v1','phase':'running','started_at_utc':now(),
        'source_campaign_sha256':digest(args.campaign),'source_sha256':hashes,'degree':args.degree,'h_mm':args.h,'cfl':args.cfl,'integrator':args.integrator,'substeps':args.substeps,
        'test_scope':source.get('test_scope','Previously evaluated cases, now development for reference discretization; no fresh generalization claim.'),
        'accepted_for_solver_routing':False,'cases':[]}
    atomic(args.output/'manifest.json',manifest)
    for original in cases:
        scene=validate_scene(original['scene']);mesh=make_mesh(scene,args.degree,args.h)
        row={'case_id':original['case_id'],'scene':scene,'phase':'running','controls':[]};manifest['cases'].append(row)
        for key in ['split','geometry_group_sha256']:
            if key in original:row[key]=original[key]
        np.savez_compressed(args.output/(row['case_id']+'-mesh.npz'),**mesh)
        row['mesh_file']=row['case_id']+'-mesh.npz';row['mesh_sha256']=digest(args.output/row['mesh_file'])
        atomic(args.output/'manifest.json',manifest)
        prepared=None
        if args.integrator=='newmark':
            from .implicit import prepare
            prepared=prepare(scene,mesh,args.substeps)
        for law in args.laws:
            name=row['case_id']+'-'+law;job=args.output/name;job.mkdir();control={'law':law,'directory':name,'phase':'running','started_at_utc':now()};row['controls'].append(control);atomic(args.output/'manifest.json',manifest)
            current=copy.deepcopy(scene);current['constitutive']=law;atomic(job/'scene.json',current)
            def callback(progress):atomic(job/'status.json',dict(phase='running',updated_at_utc=now(),**progress))
            try:
                if args.integrator=='newmark':
                    from .implicit import solve as implicit_solve
                    result=implicit_solve(current,mesh,args.substeps,args.fields,callback,prepared)
                else:result=solve(current,mesh,args.cfl,args.fields,callback)
                diagnostics=result.pop('diagnostics');fields=result.pop('fields_m')
                if fields is not None:np.savez_compressed(job/'field.npz',displacement_m=fields)
                np.savez_compressed(job/'probes.npz',**result);atomic(job/'diagnostics.json',diagnostics)
                control.update(phase='completed',completed_at_utc=now(),probes_sha256=digest(job/'probes.npz'),diagnostics_sha256=digest(job/'diagnostics.json'),scene_sha256=digest(job/'scene.json'))
                if fields is not None:control['field_sha256']=digest(job/'field.npz')
                atomic(job/'status.json',control);print(json.dumps({'case_id':row['case_id'],'law':law,**{k:v for k,v in diagnostics.items() if k!='energy_samples'}}),flush=True)
            except BaseException as error:
                control.update(phase='failed',error=str(error));row['phase']='failed';manifest['phase']='failed';atomic(job/'status.json',control);atomic(args.output/'manifest.json',manifest);raise
            atomic(args.output/'manifest.json',manifest)
        row['phase']='completed';atomic(args.output/'manifest.json',manifest)
    manifest.update(phase='completed',completed_at_utc=now());atomic(args.output/'manifest.json',manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--degree',type=int,default=3);p.add_argument('--h',type=float,default=1.6);p.add_argument('--cfl',type=float,default=.3);p.add_argument('--fields',action='store_true');p.add_argument('--integrator',choices=['verlet','newmark'],default='verlet');p.add_argument('--substeps',type=int,default=4);p.add_argument('--laws',nargs='+',choices=['linear','stvk','murnaghan'],default=['linear','stvk','murnaghan']);run(p.parse_args())
