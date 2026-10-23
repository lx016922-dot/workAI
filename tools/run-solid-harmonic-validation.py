"""Explicitly run three five-control studies for harmonic discretization checks."""
import argparse,hashlib,json,re,shutil,subprocess,sys,uuid
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from solid.schema import default_scene,validate_scene
from solid.worker import write_json

def checked_coarse(root,id,scene):
    """Reuse completed controls only after binding their inputs and physics code."""
    if not re.fullmatch(r'study-[a-f0-9]{32}',id):raise ValueError('Invalid coarse study identity.')
    directory=root/'solid/studies'/id;data=(directory/'result.json').read_bytes();record=json.loads(data);status=json.loads((directory/'status.json').read_text())
    assert status['phase']=='completed' and status['study_id']==id and hashlib.sha256(data).hexdigest()==status['result_sha256']
    assert record['study_id']==id and record['schema']=='workai.solid.nonlinear-study.v1'
    assert hashlib.sha256((directory/'study.json').read_bytes()).hexdigest()==record['config_sha256']
    physical=lambda s:{k:v for k,v in s.items() if k!='name'}
    assert physical(record['scene'])==physical(scene),'Coarse study scene differs from the requested physical/numerical baseline.'
    assert [r['role'] for r in record['controls']]==['linear_A','geometric_A','material_A','geometric_2A','material_2A']
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((REPO/'solid').glob('*.py'))}
    for i,ref in enumerate(record['controls']):
        assert re.fullmatch(r'solid-[a-f0-9]{32}',ref['job_id']);job=root/'solid/jobs'/ref['job_id'];raw=(job/'result.json').read_bytes();result=json.loads(raw);completed=json.loads((job/'status.json').read_text())
        assert completed['phase']=='completed' and hashlib.sha256(raw).hexdigest()==ref['result_sha256']==completed['result_sha256']
        expected=json.loads(json.dumps(scene));expected['constitutive']=['linear','stvk','murnaghan','stvk','murnaghan'][i];expected['source']['amplitude_mpa']*=2 if i>=3 else 1
        assert physical(result['scene'])==physical(expected) and result['source_sha256']==hashes
        assert hashlib.sha256((job/'scene.json').read_bytes()).hexdigest()==result['scene_sha256']
        assert result['field']['dtype']=='little_endian_float64' and hashlib.sha256((job/'field.bin.gz').read_bytes()).hexdigest()==result['field']['sha256']
    return {'level':'coarse','study_id':id,'result_sha256':status['result_sha256']}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/workspace/workai-remote'));parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--scene',type=Path);parser.add_argument('--mesh-sizes',type=float,nargs=2,default=[.5,.35]);parser.add_argument('--reuse-coarse');args=parser.parse_args()
    if not args.mesh_sizes[0]>args.mesh_sizes[1]>0:parser.error('Mesh sizes must be positive and strictly decrease.')
    base=validate_scene(json.loads(args.scene.read_text()) if args.scene else default_scene('plate'))
    if not args.scene:base['receivers'][1]['direction']=[1,0,0]
    baseline=json.loads(json.dumps(base));baseline['numerics'].update(h_mm=args.mesh_sizes[0],cfl=.3)
    reused=checked_coarse(args.root,args.reuse_coarse,validate_scene(baseline)) if args.reuse_coarse else None
    evidence=args.evidence;evidence.mkdir(parents=True,exist_ok=False);frozen=evidence/'source_snapshot';(frozen/'solid').mkdir(parents=True)
    for path in (REPO/'solid').glob('*.py'):shutil.copyfile(path,frozen/'solid'/path.name)
    references=[];fine_mesh=None
    for level,h,cfl in [('coarse',args.mesh_sizes[0],.3),('fine',args.mesh_sizes[1],.3),('timecheck',args.mesh_sizes[1],.15)]:
        if level=='coarse' and reused:
            references.append(reused);write_json(evidence/'studies.json',{'references':references,'complete':False});print(json.dumps({**reused,'reused_completed_controls':True}),flush=True);continue
        scene=json.loads(json.dumps(base));scene['numerics'].update(h_mm=h,cfl=cfl);scene['name']=base['name'][:70]+' · '+level;scene=validate_scene(scene)
        id='study-'+uuid.uuid4().hex;directory=args.root/'solid/studies'/id;directory.mkdir(parents=True)
        write_json(directory/'study.json',{'schema':'workai.solid.study-input.v1','study_id':id,'scene':scene})
        shutil.copytree(frozen,directory/'source_snapshot')
        command=[sys.executable,'-m','solid.study','--study-dir',str(directory)]
        if level=='timecheck':command+=['--mesh-reference',str(fine_mesh)]
        with (directory/'worker.log').open('x') as log:subprocess.run(command,cwd=directory/'source_snapshot',stdout=log,stderr=subprocess.STDOUT,check=True)
        data=(directory/'result.json').read_bytes();r=json.loads(data);status=json.loads((directory/'status.json').read_text());assert status['phase']=='completed' and hashlib.sha256(data).hexdigest()==status['result_sha256']
        if level=='fine':fine_mesh=args.root/'solid/jobs'/r['controls'][0]['job_id']
        references.append({'level':level,'study_id':id,'result_sha256':status['result_sha256']})
        write_json(evidence/'studies.json',{'references':references,'complete':len(references)==3});print(json.dumps({'level':level,'study_id':id,'completed':True}),flush=True)

if __name__=='__main__':main()
