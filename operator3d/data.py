"""SHA-bound float64 full fields, with strictly separate geometry groups."""
import gzip,hashlib,json
from pathlib import Path
import numpy as np
import torch
from solid.mesh import prepare_mesh,nodal_load,face_selection,locate
from .campaign import geometry_identity
from .physics import L0,U0

def digest(data):return hashlib.sha256(data).hexdigest()
def check_splits(manifest,require_test=True):
    if not manifest.get('complete') or manifest.get('schema')!='workai.operator3d.campaign.v1':raise ValueError('A completed campaign is required.')
    groups={};ids=set();counts={'train':0,'validation':0,'test':0}
    for case in manifest['cases']:
        if case['case_id'] in ids:raise ValueError('Duplicate physical case.')
        ids.add(case['case_id']);group=geometry_identity(case['scene'])
        if group!=case['geometry_group_sha256']:raise ValueError('Geometry identity changed.')
        if group in groups:raise ValueError('Geometry repeats cannot cross splits or count as independent cases.')
        groups[group]=case['split'];counts[case['split']]+=1
    if counts!=manifest['split_counts'] or min(counts['train'],counts['validation'])==0 or (require_test and counts['test']==0):raise ValueError('Incomplete grouped split.')
    return counts

def tensors(scene,mesh):
    points=mesh['points'];surface=mesh['surface'];load,selected=nodal_load(mesh,scene['source']);fixed=np.zeros(len(points),dtype=bool);fixed_faces=np.zeros(len(surface),dtype=bool)
    if scene['boundary']['fixed_face']!='none':fixed_faces=face_selection(mesh,scene['boundary']['fixed_face']);fixed[np.unique(surface[fixed_faces])]=True
    mass=np.zeros(len(points))
    for a in range(4):np.add.at(mass,mesh['tetra'][:,a],scene['material']['rho_kg_m3']*mesh['volume']/4)
    boundary=np.zeros(len(points),dtype=bool);boundary[np.unique(surface)]=True
    features=np.column_stack([points[surface].mean(1)/L0,mesh['normal'],selected.astype(float),fixed_faces.astype(float)])
    s=scene['source'];context=np.r_[np.ptp(points,axis=0)/L0,np.array(s['center_mm'])*.001/L0,s['frequency_mhz']/.2,s['cycles']/2,s['delay_us'],np.sin(np.deg2rad(s['phase_deg'])),np.cos(np.deg2rad(s['phase_deg'])),s['amplitude_mpa'],s['direction'],scene['acquisition']['duration_us']/14,float(fixed.any())]
    result={'scene':scene,'points':points/L0,'tetra':mesh['tetra'].astype(np.int64),'grad':mesh['grad'],'volume':mesh['volume'],'mass':mass,'load':load,'fixed':fixed,'boundary_nodes':boundary,'surface_features':features,'surface_weights':mesh['area']/mesh['area'].sum(),'context':context}
    return {k:torch.as_tensor(v,dtype=torch.bool if v.dtype==bool else torch.int64 if k=='tetra' else torch.float64) if isinstance(v,np.ndarray) else v for k,v in result.items()}

def load_dataset(root,campaign,include_splits=None,require_test=True):
    root=Path(root);campaign=Path(campaign);raw=campaign.read_bytes();manifest=json.loads(raw);counts=check_splits(manifest,require_test=require_test);cases=[];source_hashes=None;references=[]
    if include_splits is not None and (not set(include_splits)<={'train','validation','test'} or 'train' not in include_splits):raise ValueError('Training-only normalization requires the train split.')
    for item in manifest['cases']:
        # Checkpoint selection must not read final-test labels, even into memory.
        if include_splits is not None and item['split'] not in include_splits:continue
        if [r['role'] for r in item['controls']]!=['linear','geometric','material']:raise ValueError('Three ordered physical controls required.')
        fields=[];records=[];identity=[]
        for control,law in zip(item['controls'],['linear','stvk','murnaghan']):
            import re
            if not re.fullmatch(r'solid-[a-f0-9]{32}',control['job_id']):raise ValueError('Invalid control identity.')
            directory=root/'solid/jobs'/control['job_id'];result_bytes=(directory/'result.json').read_bytes();r=json.loads(result_bytes);status=json.loads((directory/'status.json').read_text())
            assert status['phase']=='completed' and digest(result_bytes)==status['result_sha256']==control['result_sha256']
            expected=dict(item['scene']);expected['constitutive']=law;assert r['scene']==expected and r['job_id']==control['job_id']
            assert digest((directory/'scene.json').read_bytes())==r['scene_sha256']
            if source_hashes is None:source_hashes=r['source_sha256']
            assert r['source_sha256']==source_hashes
            for name,sha in source_hashes.items():assert digest((campaign.parent/'source_snapshot/solid'/name).read_bytes())==sha
            packed=(directory/'field.bin.gz').read_bytes();assert digest(packed)==r['field']['sha256'] and len(packed)==r['field']['compressed_bytes'];field_bytes=gzip.decompress(packed)
            assert r['field']['dtype']=='little_endian_float64' and digest(field_bytes)==r['field']['uncompressed_sha256'] and len(field_bytes)==r['field']['uncompressed_bytes']
            field=np.frombuffer(field_bytes,dtype='<f8').reshape(r['field']['shape']);assert np.isfinite(field).all()
            if records:assert r['nodes_mm']==records[0]['nodes_mm'] and r['tetra']==records[0]['tetra'] and r['times_us']==records[0]['times_us'] and r['diagnostics']['internal_step_us']==records[0]['diagnostics']['internal_step_us']
            fields.append(field);records.append(r);identity.append({**control,'field_sha256':r['field']['sha256'],'raw_field_sha256':r['field']['uncompressed_sha256']})
        directory=root/'solid/jobs'/item['controls'][0]['job_id'];mesh_bytes=(directory/'mesh.npz').read_bytes();assert digest(mesh_bytes)==records[0]['mesh_sha256']
        with np.load(directory/'mesh.npz',allow_pickle=False) as saved:points=saved['points'];tetra=saved['tetra']
        assert np.array_equal(points*1000,records[0]['nodes_mm']) and np.array_equal(tetra,records[0]['tetra'])
        mesh=prepare_mesh(points,tetra);case=tensors(item['scene'],mesh);parts=np.stack([fields[0],fields[1]-fields[0],fields[2]-fields[1]],2)/U0
        case.update(case_id=item['case_id'],split=item['split'],group=item['geometry_group_sha256'],parts=parts,times=np.array(records[0]['times_us']),receivers=[locate(mesh,rx['point_mm']) for rx in item['scene']['receivers']])
        cases.append(case);references.append({'case_id':item['case_id'],'split':item['split'],'geometry_group_sha256':item['geometry_group_sha256'],'controls':identity,'nodes':len(points),'elements':len(tetra),'frames':len(parts)})
    # Equal case weighting; volume-weighted spatial scale, from training only.
    scales=[]
    for case in cases:
        if case['split']=='train':
            weight=case['mass'].numpy().copy();weight/=weight.sum();scales.append(np.einsum('tnkc,n->k',case['parts']**2,weight)/(len(case['times'])*3))
    scale=np.maximum(np.sqrt(np.mean(scales,axis=0)),[1e-2,5e-7,5e-7])
    description={'schema':'workai.operator3d.dataset.v1','campaign_sha256':digest(raw),'physical_case_count':manifest['case_count'],'split_counts':counts,'normalization_from_training_only':True,'branch_scales_in_100nm':scale.tolist(),'all_fields_float64':True,'same_mesh_controls_verified':True,'geometry_group_split_verified':True,'discretization_verified':manifest['discretization_verified'],'cases':references,'physics_source_sha256':source_hashes,'scope':manifest['scope']}
    if include_splits is not None:description.update(loaded_splits=sorted(include_splits),withheld_cases=[{'case_id':c['case_id'],'split':c['split'],'geometry_group_sha256':c['geometry_group_sha256']} for c in manifest['cases'] if c['split'] not in include_splits],final_test_fields_loaded='test' in include_splits)
    return cases,scale,description
