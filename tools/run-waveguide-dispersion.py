"""Independent SAFE meshes: no transient simulation or field fitting."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from waveguide.safe import rectangle,assemble,extensional_branch

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);args=p.parse_args()
    scene=json.loads(args.scene.read_text());g=scene['geometry'];m=scene['material'];f=scene['source']['frequency_mhz']*1e6
    if g['kind']!='plate':raise ValueError('Only a free rectangular cross-section is supported.')
    args.evidence.mkdir(parents=True,exist_ok=False);snapshot=args.evidence/'source';snapshot.mkdir()
    hashes={}
    for source in [REPO/'waveguide/safe.py',Path(__file__)]:
        data=source.read_bytes();(snapshot/source.name).write_bytes(data);hashes[source.name]=hashlib.sha256(data).hexdigest()
    lam=m['lambda_pa'];mu=m['mu_pa'];young=mu*(3*lam+2*mu)/(lam+mu);levels=[]
    for h in [.25,.125,.0625]:
        started=time.monotonic();points,tri=rectangle(g['width_mm']*.001,g['thickness_mm']*.001,h*.001);system=assemble(points,tri,m['rho_kg_m3'],lam,mu);branches=[];vectors=[]
        for frequency in [f,2*f]:
            result=extensional_branch(system,frequency,young);vectors.append(result.pop('vector'));branches.append(result)
        name='mesh-'+str(h).replace('.','p')+'.npz';np.savez_compressed(args.evidence/name,points_m=points,triangles=tri,fundamental=vectors[0],second=vectors[1])
        row={'h_mm':h,'nodes':len(points),'triangles':len(tri),'branches':branches,'mode_file':name,'mode_file_sha256':hashlib.sha256((args.evidence/name).read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-started};levels.append(row)
        print(json.dumps({'h_mm':h,'nodes':len(points),'k_rad_m':[r['wavenumber_rad_m'] for r in branches],'seconds':row['elapsed_s']}),flush=True)
    errors=[[abs(a['branches'][i]['wavenumber_rad_m']/b['branches'][i]['wavenumber_rad_m']-1) for i in range(2)] for a,b in zip(levels,levels[1:])]
    fine=levels[-1]['branches'];delta=fine[1]['wavenumber_rad_m']-2*fine[0]['wavenumber_rad_m']
    result={'schema':'workai.rectangular-waveguide.dispersion.v1','cross_section_mm':{'width':g['width_mm'],'thickness':g['thickness_mm']},'material':{k:m[k] for k in ['rho_kg_m3','lambda_pa','mu_pa']},'source_sha256':hashes,'levels':levels,'successive_relative_k_differences':errors,'finest_pair_within_one_percent':all(v<=.01 for v in errors[-1]),'mismatch_rad_m':delta,'coherence_length_mm':np.pi/abs(delta)*1000,'mode_tracking':'mass MAC continuation from the long-wavelength extensional branch','other_second_harmonic_modes_excluded':False,'modal_purity_verified':False,'nonlinear_coupling_integral_computed':False,'time_convention':'Re[phi(y,z) exp(i omega t - i k x)]'}
    (args.evidence/'dispersion.json').write_text(json.dumps(result,indent=2));print(json.dumps({'finest_pair_k_errors':errors[-1],'mismatch_rad_m':delta,'coherence_length_mm':result['coherence_length_mm']}))

if __name__=='__main__':main()
