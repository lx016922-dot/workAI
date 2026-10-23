"""Five constitutive/amplitude controls on one exact mesh, with immutable jobs."""
import argparse,copy,hashlib,json,time,uuid
from pathlib import Path
from .schema import validate_scene,exact,require
from .worker import run,write_json,now

CONTROLS=[('linear_A','linear',1),('geometric_A','stvk',1),('material_A','murnaghan',1),('geometric_2A','stvk',2),('material_2A','murnaghan',2)]

def validate_study(value):
    exact(value,['scene'],'非线性对照');scene=validate_scene(value['scene'])
    require(scene['source']['amplitude_mpa']<=5,'五组对照需要双幅值计算，基准幅值不能超过 5 MPa。')
    require(scene['acquisition']['duration_us']*scene['source']['frequency_mhz']>=2,'谐波分析记录至少需要两个基频周期。')
    scene['constitutive']='murnaghan'
    return {'scene':scene}

def execute(directory,mesh_reference=None):
    directory=Path(directory);started=time.monotonic();root=directory.parents[1];config_bytes=(directory/'study.json').read_bytes()
    config=json.loads(config_bytes);scene=validate_study({'scene':config['scene']})['scene'];jobs=root/'jobs'
    status={'study_id':directory.name,'phase':'running','started_at_utc':now(),'completed_controls':0,'total_controls':5,'jobs':[]}
    def publish(**items):status.update(items,updated_at_utc=now(),elapsed_s=time.monotonic()-started);write_json(directory/'status.json',status)
    publish();references=[];first=None;shared_nodes=None;shared_tetra=None;step=None;physics=None
    try:
        for role,law,factor in CONTROLS:
            require(time.monotonic()-started<2700,'五组对照超过 45 分钟预算，停止启动后续计算。')
            current=copy.deepcopy(scene);current['constitutive']=law;current['source']['amplitude_mpa']*=factor;current['name']=scene['name'][:70]+' · '+role
            job=jobs/('solid-'+uuid.uuid4().hex);job.mkdir();write_json(job/'scene.json',current)
            status['jobs'].append({'role':role,'job_id':job.name});publish(active_role=role,active_job_id=job.name)
            remaining=max(1,min(1800,int(2700-(time.monotonic()-started))))
            run(job,mesh_reference if first is None else first,'time' if first is None else 'controls',output_precision='float64',time_budget_s=remaining)
            result_bytes=(job/'result.json').read_bytes();record=json.loads(result_bytes)
            if first is None:
                first=job;shared_nodes=record['nodes_mm'];shared_tetra=record['tetra'];step=record['diagnostics']['internal_step_us'];physics=record['source_sha256']
            else:
                require(record['nodes_mm']==shared_nodes and record['tetra']==shared_tetra,'本构对照网格不同。')
                require(record['diagnostics']['internal_step_us']==step and record['source_sha256']==physics,'本构对照时间步或源码不同。')
            references.append({'role':role,'job_id':job.name,'result_sha256':hashlib.sha256(result_bytes).hexdigest()})
            publish(completed_controls=len(references));print(json.dumps({'study_id':directory.name,'completed_control':role,'job_id':job.name}),flush=True)
        result={'schema':'workai.solid.nonlinear-study.v1','study_id':directory.name,'scene':scene,'config_sha256':hashlib.sha256(config_bytes).hexdigest(),
                'controls':references,'same_mesh_and_time_step':True,'field_precision':'float64','internal_step_us':step,'completed_at_utc':now(),
                'harmonic_convergence_verified':False,'comsol_agreement_verified':False,'surrogate_used':False}
        write_json(directory/'result.json',result);publish(phase='completed',active_job_id=None,active_role=None,result_sha256=hashlib.sha256((directory/'result.json').read_bytes()).hexdigest())
        return result
    except BaseException as error:publish(phase='failed',error=str(error));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study-dir',required=True);parser.add_argument('--mesh-reference');args=parser.parse_args();execute(args.study_dir,args.mesh_reference)
