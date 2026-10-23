"""High-order fields and geometry, without importing solver runtime dependencies."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from spectral3d.mesh import boundary_data,interpolation
from spectral3d.curved_mesh import boundary_faces
from spectral3d.compare import digest,spectral
from spectral3d.field_compare import field
from operator3d.campaign import geometry_identity
from operator3d.physics import L0,U0


def tensors(scene,mesh):
    points=mesh['points'];elements=mesh['elements'];load,fixed_ids,_=boundary_data(mesh,scene)
    fixed=np.zeros(len(points),bool);fixed[fixed_ids]=True;boundary=np.zeros(len(points),bool);features=[];areas=[]
    for e,axis,end,ids in boundary_faces(mesh):
        boundary[ids]=True;nodes=points[ids];a=nodes[-1,0]-nodes[0,0];b=nodes[0,-1]-nodes[0,0];normal=np.cross(a,b);area=np.linalg.norm(normal);normal/=area
        center=nodes.mean((0,1));inside=points[elements[e]].mean((0,1,2))
        if normal@(center-inside)<0:normal=-normal
        features.append(np.r_[center/L0,normal,float(np.any(load[ids])),float(fixed[ids].all())]);areas.append(area)
    s=scene['source'];context=np.r_[np.ptp(points,axis=0)/L0,np.array(s['center_mm'])*.001/L0,s['frequency_mhz']/.2,s['cycles']/2,s['delay_us'],np.sin(np.deg2rad(s['phase_deg'])),np.cos(np.deg2rad(s['phase_deg'])),s['amplitude_mpa'],s['direction'],scene['acquisition']['duration_us']/14,float(fixed.any())]
    data={'scene':scene,'mesh':mesh,'points':points/L0,'mass':mesh['mass'],'load':load,'fixed':fixed,'boundary_nodes':boundary,'surface_features':np.asarray(features),'surface_weights':np.array(areas)/sum(areas),'context':context}
    return {k:torch.as_tensor(v,dtype=torch.bool if v.dtype==bool else torch.float64) if isinstance(v,np.ndarray) else v for k,v in data.items()}


def verify_manifest(path):
    manifest=json.loads(path.read_text());groups=set();counts={'train':0,'validation':0,'test':0};ids=set()
    if manifest['phase'] not in ['completed','completed_with_failures']:raise ValueError('Finish the numerical dataset first.')
    for c in manifest['cases']:
        key=geometry_identity(c['scene'])
        if key in groups or c['case_id'] in ids:raise ValueError('Repeated geometry or case identity.')
        groups.add(key);ids.add(c['case_id']);counts[c['split']]+=1
    if counts!=manifest['split_counts']:raise ValueError('Split counts changed.')
    return manifest


def checked_reference(path):
    cases=spectral(path)
    for case in cases.values():
        for record in case['diagnostics']:
            if not (record['energy_balance_within_1_percent'] and record['max_strain']<=.001 and record['min_J']>0):
                raise ValueError('Reference energy, strain or Jacobian guard failed: '+str(path))
    return cases


def load_dataset(path,splits=('train','validation'),admission_path=None):
    path=Path(path);manifest=verify_manifest(path);admission=json.loads(Path(admission_path).read_text()) if admission_path else None
    if admission:
        assert admission['source_manifest_sha256']==digest(path) and admission['phase']=='completed'
        for name,sha in admission['source_sha256'].items():assert digest(Path(admission_path).parent/'source_snapshot'/name)==sha
    cases=[];records=[]
    for item in manifest['cases']:
        if item['split'] not in splits:continue
        chosen=next(c for c in admission['cases'] if c['case_id']==item['case_id']) if admission else item
        if not chosen['reference_checks_passed']:raise ValueError('Unresolved reference checks: '+item['case_id'])
        quality_path=Path(chosen['quality_path']) if admission else path.parent/item['case_id']/'quality.json'
        assert digest(quality_path)==chosen['quality_sha256'];quality=json.loads(quality_path.read_text())
        if admission:assert quality['comparison_source_sha256']==admission['source_sha256']
        if not quality['field']['all_checks_pass'] or not all(quality[k]['all_probe_checks_pass'] for k in ['space','degree','time']):raise ValueError('Reference admission is not supported by its checks.')
        base=Path(chosen['base_manifest']) if admission else path.parent/item['levels']['base']['manifest']
        expected=chosen['base_manifest_sha256'] if admission else item['levels']['base']['sha256'];assert digest(base)==expected
        for k in ['space','degree','time']:assert quality[k]['left_manifest_sha256']==expected
        assert quality['field']['left_manifest_sha256']==expected
        for level in ['space','degree','time']:
            reference=Path(chosen['levels'][level]['manifest']) if admission else path.parent/chosen['levels'][level]['manifest']
            assert digest(reference)==chosen['levels'][level]['sha256']==quality[level]['right_manifest_sha256']
            checked_reference(reference)
            if level=='space':
                assert quality['field']['right_manifest_sha256']==digest(reference)
                reference_case=next(c for c in json.loads(reference.read_text())['cases'] if c['case_id']==item['case_id'])
                for control in reference_case['controls']:assert digest(reference.parent/control['directory']/'field.npz')==control['field_sha256']
        summary=checked_reference(base);m=json.loads(base.read_text());row=next(c for c in m['cases'] if c['case_id']==item['case_id']);assert row['scene']==item['scene']
        with np.load(base.parent/row['mesh_file'],allow_pickle=False) as data:mesh=dict(data)
        values=[field(base.parent,row,law) for law in ['linear','stvk','murnaghan']]
        case=tensors(item['scene'],mesh);parts=np.stack([values[0],values[1]-values[0],values[2]-values[1]],2)/U0
        if not np.isfinite(parts).all():raise ValueError('Nonfinite labels.')
        case.update(case_id=item['case_id'],split=item['split'],group=geometry_identity(item['scene']),parts=parts,times=summary[item['case_id']]['times'],receivers=[interpolation(mesh,r['point_mm']) for r in item['scene']['receivers']])
        cases.append(case);records.append({'case_id':item['case_id'],'split':item['split'],'geometry_group_sha256':case['group'],'base_manifest':str(base),'base_manifest_sha256':expected,'quality_sha256':chosen['quality_sha256'],'nodes':len(mesh['points']),'frames':len(parts),'controls':row['controls']})
    scales=[]
    for case in cases:
        if case['split']=='train':
            weight=case['mass'].numpy();scales.append(np.einsum('tnkc,n->k',case['parts']**2,weight/weight.sum())/(len(case['times'])*3))
    scale=np.maximum(np.sqrt(np.mean(scales,axis=0)),[1e-2,5e-7,5e-7]) if scales else None
    description={'schema':'workai.spectral_operator_dataset.v1','campaign_sha256':digest(path),'admission_sha256':digest(Path(admission_path)) if admission_path else None,'split_counts':manifest['split_counts'],'loaded_splits':list(splits),'final_test_fields_loaded':'test' in splits,'normalization_from_training_only':True,'branch_scales_in_100nm':scale.tolist() if scale is not None else None,'cases':records,'reference_checks_passed':True,'scope':manifest['scope']}
    return cases,scale,description
