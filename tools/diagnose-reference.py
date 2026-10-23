"""Audit retained P1 refinement pairs without changing or re-solving them."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from solid.mesh import prepare_mesh, locate, nodal_load

ROOT = Path('/workspace/workai-remote')
manifest_path = ROOT/'modal3d/refinements/holdout64-h040-20261023/manifest.json'
manifest = json.loads(manifest_path.read_text())
rows = []
for case in manifest['cases']:
    scene = case['original']['scene']; g = scene['geometry']; levels = []
    exact_volume = (g['length_mm']*np.pi*(g['outer_radius_mm']**2-g['inner_radius_mm']**2)
                    if g['kind']=='pipe' else g['length_mm']*g['width_mm']*g['thickness_mm'])
    if g['kind']=='hole': exact_volume -= np.pi*g['hole_radius_mm']**2*g['thickness_mm']
    if g['kind']=='notch': exact_volume -= g['notch_width_mm']*g['width_mm']*g['notch_depth_mm']
    for level in [case['original'], case['levels'][0]]:
        probes = []; record = None
        for control in level['controls']:
            directory = ROOT/'solid/jobs'/control['job_id']
            raw = (directory/'result.json').read_bytes()
            assert hashlib.sha256(raw).hexdigest()==control['result_sha256']
            result = json.loads(raw)
            if record is None:
                record = result
                with np.load(directory/'mesh.npz', allow_pickle=False) as mesh_file:
                    mesh = prepare_mesh(mesh_file['points'], mesh_file['tetra'])
                receivers = [locate(mesh, r['point_mm']) for r in scene['receivers']]
            raw = gzip.decompress((directory/'field.bin.gz').read_bytes())
            assert hashlib.sha256(raw).hexdigest()==result['field']['uncompressed_sha256']
            fields = np.frombuffer(raw, dtype='<f8').reshape(result['field']['shape'])
            probes.append(np.stack([np.einsum('tac,a->tc', fields[:,mesh['tetra'][c]],w) for c,w in receivers]))
        force, selected = nodal_load(mesh, scene['source'])
        points = mesh['points']*1000
        levels.append({'h_mm':level['scene']['numerics']['h_mm'], 'nodes':len(points),
            'elements':len(mesh['tetra']), 'volume_mm3':float(mesh['volume'].sum()*1e9),
            'volume_relative_error':float(mesh['volume'].sum()*1e9/exact_volume-1),
            'loaded_area_mm2':record['load_area_mm2'], 'total_nominal_force_n':force.sum(0).tolist(),
            'receiver_reconstruction_error_mm':max(float(np.linalg.norm(w@points[mesh['tetra'][c]]-r['point_mm'])) for (c,w),r in zip(receivers,scene['receivers'])),
            'full':probes[2], 'material':probes[2]-probes[1]})
    comparisons=[]
    for i, receiver in enumerate(scene['receivers']):
        item={'name':receiver['name'], 'point_mm':receiver['point_mm']}
        for part in ['full','material']:
            a,b=levels[0][part][i],levels[1][part][i]
            norm=np.linalg.norm(b,axis=0)
            item[part+'_xyz_relative_l2']=(np.linalg.norm(a-b,axis=0)/np.maximum(norm,1e-30)).tolist()
            item[part+'_xyz_rms_nm']=(np.sqrt(np.mean(b*b,axis=0))*1e9).tolist()
        comparisons.append(item)
    for level in levels:
        del level['full'],level['material']
    rows.append({'case_id':case['case_id'], 'exact_volume_mm3':exact_volume,
        'remaining_ligament_mm':g['thickness_mm']-g['notch_depth_mm'] if g['kind']=='notch' else None,
        'levels':levels, 'probes':comparisons})
out={'schema':'workai.reference_diagnosis.v1','reference_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
     'new_transient_solves':0, 'cases':rows}
path=ROOT/'validation/20261023-spectral-reference/p1-diagnosis.json'
with path.open('x') as f: json.dump(out,f,indent=2)
for row in rows: print(json.dumps(row),flush=True)
