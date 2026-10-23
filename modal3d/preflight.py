"""Measure refinement costs against the unchanged reference solver limits."""
import argparse,copy,hashlib,json,math
from pathlib import Path
from solid.mesh import make_mesh
from solid.schema import validate_scene

def main(args):
    campaign=json.loads(args.campaign.read_text());rows=[]
    for row in campaign['cases']:
        if row['case_id'] not in args.cases:continue
        for h in args.h:
            scene=copy.deepcopy(row['scene']);scene['numerics']['h_mm']=h;result={'case_id':row['case_id'],'h_mm':h,'reference_solver_limits_unchanged':True}
            try:
                mesh=make_mesh(validate_scene(scene));material=scene['material'];speed=math.sqrt((material['lambda_pa']+2*material['mu_pa'])/material['rho_kg_m3']);sample=scene['acquisition']['sample_step_us']*1e-6;frames=round(scene['acquisition']['duration_us']/scene['acquisition']['sample_step_us'])+1;costs=[]
                for cfl in [scene['numerics']['cfl'],scene['numerics']['cfl']/2]:
                    step=sample/max(1,math.ceil(sample/(cfl*mesh['min_altitude']/(speed*1.05))));steps=round(scene['acquisition']['duration_us']*1e-6/step);product=steps*len(mesh['tetra']);field_bytes=frames*len(mesh['points'])*3*8
                    costs.append({'cfl':cfl,'steps':steps,'elements_times_steps':product,'float64_field_bytes':field_bytes,'within_current_limits':steps<=50000 and product<=350000000 and field_bytes<=192*1024**2})
                result.update(nodes=len(mesh['points']),elements=len(mesh['tetra']),min_altitude_mm=mesh['min_altitude']*1000,costs=costs)
            except ValueError as error:result.update(mesh_rejected=True,error=str(error))
            rows.append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
    output={'schema':'workai.reference_budget_preflight.v1','campaign_sha256':hashlib.sha256(args.campaign.read_bytes()).hexdigest(),'reference_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).resolve().parents[1]/'solid').glob('*.py')},'new_transient_solves':0,'cases':rows,'scope':'Measured costs against current software limits; exceeding them does not prove a larger machine or larger budget cannot solve the case.'}
    with args.output.open('x') as f:json.dump(output,f,indent=2,ensure_ascii=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--cases',nargs='+',required=True);p.add_argument('--h',type=float,nargs='+',default=[.28,.2]);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
