"""Compare the differentiable residual to independently generated solver forces."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from spectralop.data import tensors
from spectralop.physics import star,internal
from operator3d.physics import U0

p=argparse.ArgumentParser();p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);rows=[];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for kind in ['plate','pipe','hole']:
 path=a.fixtures/(kind+'.npz')
 with np.load(path,allow_pickle=False) as f:mesh=dict(f)
 parts=mesh.pop('parts');expected=mesh.pop('force');scene=json.loads((a.fixtures/(kind+'.json')).read_text());case=tensors(scene,mesh);patch=star(case,np.arange(len(parts)));actual=internal(torch.as_tensor(parts/U0),case,patch)[0].numpy();relative=[]
 for branch in range(3):
  value=float(np.linalg.norm(actual[:,branch]-expected[:,branch])/np.linalg.norm(expected[:,branch]));assert value<2e-9,(kind,branch,value);relative.append(value)
 rows.append({'geometry':kind,'relative_force_difference_by_branch':relative,'fixture_sha256':sha(path)})
result={'passed':True,'cases':rows,'source_sha256':{str(p.relative_to(REPO)):sha(p) for p in [REPO/'spectralop/physics.py',REPO/'operator3d/physics.py',Path(__file__)]}}
with a.output.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
