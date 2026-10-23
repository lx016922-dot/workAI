"""Offline high-order reference field with exact element receiver interpolation."""
import argparse
import base64
import gzip
import hashlib
import html
import json
import re
import secrets
import sys
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from spectral3d.compare import spectral,digest


def surface_triangles(elements):
    faces={}
    for element in elements:
        for axis in range(3):
            for end in [0,-1]:
                selection=[slice(None)]*3;selection[axis]=end;face=element[tuple(selection)]
                key=tuple(sorted([int(face[0,0]),int(face[-1,0]),int(face[-1,-1]),int(face[0,-1])]))
                if key in faces:faces[key]=None
                else:faces[key]=face
    triangles=[]
    for face in faces.values():
        if face is None:continue
        for i in range(len(face)-1):
            for j in range(len(face)-1):
                a,b,c,d=face[i,j],face[i+1,j],face[i+1,j+1],face[i,j+1]
                triangles.extend([[a,b,c],[a,c,d]])
    return np.array(triangles,dtype=int)


def main(args):
    manifest=json.loads(args.campaign.read_text());spectral(args.campaign)
    case=next(c for c in manifest['cases'] if c['case_id']==args.case);root=args.campaign.parent
    mesh=np.load(root/case['mesh_file'],allow_pickle=False);values=[];diagnostics=None
    laws=['murnaghan'] if args.part=='full' else ['murnaghan','stvk']
    for law in laws:
        control=next(c for c in case['controls'] if c['law']==law);directory=root/control['directory']
        assert digest(directory/'field.npz')==control['field_sha256']
        with np.load(directory/'field.npz',allow_pickle=False) as field:values.append(field['displacement_m'])
        if diagnostics is None:diagnostics=json.loads((directory/'diagnostics.json').read_text())
    field=values[0] if args.part=='full' else values[0]-values[1];raw=np.asarray(field,dtype='<f8').tobytes();packed=gzip.compress(raw,compresslevel=5,mtime=0)
    assert np.isfinite(field).all() and len(raw)<=192*1024**2 and field.shape[1]<=30000
    points=mesh['points']*1000;surface=surface_triangles(mesh['elements']);scene=case['scene'];axis='xyz'.index(scene['source']['face'][0]);bound=points[:,axis].min() if scene['source']['face'].endswith('min') else points[:,axis].max()
    loaded=np.flatnonzero(np.all(abs(points[surface,axis]-bound)<1e-8,axis=1));sha=lambda b:hashlib.sha256(b).hexdigest()
    record={'schema':'workai.spectral_reference.result.v1','engine':'workai_spectral_reference_3d','job_id':args.case+'-spectral-'+args.part,
        'response_part':args.part,'scene':scene,'accepted_for_solver_routing':False,'nodes_mm':points.tolist(),'tetra':[],'surface':surface.tolist(),
        'loaded_faces':loaded.tolist(),'times_us':(np.arange(len(field))*scene['acquisition']['sample_step_us']).tolist(),'diagnostics':diagnostics,
        'spectral':{'degree':int(mesh['degree']),'gll':mesh['gll'].tolist(),'elements':mesh['elements'].reshape(len(mesh['elements']),-1).tolist(),
                    **({'mapping_kind':'isoparametric'} if 'mapping_kind' in mesh else {'mapping_kind':'cartesian','origins_mm':(mesh['origin']*1000).tolist(),'sizes_mm':(mesh['size']*1000).tolist()})},
        'evidence':{'campaign_sha256':digest(args.campaign),'mesh_sha256':case['mesh_sha256'],'field_sha256':{c['law']:c.get('field_sha256') for c in case['controls']},'dataset_split':case.get('split','development'),'export_runs_training':False},
        'field':{'layout':'time,node,xyz','dtype':'little_endian_float64','unit':'m','shape':list(field.shape),'compressed_bytes':len(packed),'uncompressed_bytes':len(raw),'sha256':sha(packed),'uncompressed_sha256':sha(raw)}}
    if args.admission:
        admission=json.loads(args.admission.read_text());assert admission['phase']=='completed'
        admitted=next(c for c in admission['cases'] if c['case_id']==args.case);assert admitted['reference_checks_passed'] and admitted['base_manifest_sha256']==digest(args.campaign)
        path=Path(admitted['quality_path']);assert digest(path)==admitted['quality_sha256'];checks=json.loads(path.read_text())
        record['spectral_reference_checks']={'threshold':.05,'h_passed':checks['space']['all_probe_checks_pass'],'p_passed':checks['degree']['all_probe_checks_pass'],'time_passed':checks['time']['all_probe_checks_pass'],'field_passed':checks['field']['all_checks_pass'],'field_checks':checks['field']['cases'][0]['checks']}
        record['evidence'].update(admission_sha256=digest(args.admission),quality_sha256=digest(path))
    source='\n'.join(re.sub(r'^export ','',re.sub(r'^import .*;\n','',(REPO/name).read_text(),flags=re.M),flags=re.M) for name in ['public/solid-data.js','public/spectral-data.js','public/solid-viewer.js'])
    source+="\nconst record=JSON.parse(document.getElementById('spectral-record').textContent);const packed=Uint8Array.from(atob(document.getElementById('spectral-field').textContent.trim()),c=>c.charCodeAt(0));const container=document.getElementById('spectral-view');try{const field=await decodeSpectralField(record,packed);container.innerHTML=solidResultMarkup(record);mountSolidResult(container,record,field,spectralPointSeries);document.getElementById('spectral-load').textContent='已核对高阶参考场 SHA-256 · 点位提取不重新求解';}catch(e){document.getElementById('spectral-load').textContent=e.message;}"
    assert '</script' not in source.lower();nonce=secrets.token_hex(18);body=json.dumps(record,separators=(',',':'),ensure_ascii=False,allow_nan=False).replace('<','\\u003c');css=(REPO/'public/styles.css').read_text();part='完整位移' if args.part=='full' else '材料非线性增量'
    kind_label={'plate':'铝板','pipe':'铝管','hole':'带孔铝板','notch':'缺口铝板'}[scene['geometry']['kind']]
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none';script-src 'nonce-{nonce}';style-src 'unsafe-inline';img-src data: blob:;connect-src 'none';base-uri 'none';form-action 'none'"><title>workAI · 高阶参考 · {part}</title><style>{css}\nbody{{min-width:0}}main{{max-width:1400px;margin:auto}}.solid-layout{{grid-template-columns:minmax(0,1.2fr) minmax(0,1fr)}}@media(max-width:1000px){{.solid-layout{{grid-template-columns:1fr}}}}</style></head><body><main><div class="page-heading"><div><div class="eyebrow">WORKAI · HIGH-ORDER REFERENCE</div><h1>{kind_label} · {part}</h1><p>三维动画、实体内点位的方向位移与频谱</p></div></div><p>{html.escape(args.case)} · {manifest['degree']} 阶 / 网格尺寸参数 {manifest['h_mm']} mm</p><p>该页展示真实高阶数值计算。参考网格检查与训练验收分别记录；此次参考计算不会改变已有模型的验收结论。</p><p id="spectral-load">正在核对高阶参考场…</p><div id="spectral-view"></div></main><script type="application/json" id="spectral-record">{body}</script><script type="application/octet-stream" id="spectral-field">{base64.b64encode(packed).decode()}</script><script type="module" nonce="{nonce}">{source}</script></body></html>'''
    with args.output.open('x') as f:f.write(page)
    print(json.dumps({'path':str(args.output),'bytes':args.output.stat().st_size,'sha256':digest(args.output),'nodes':len(points),'surface_triangles':len(surface),'frames':len(field),'part':args.part}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--part',choices=['full','material'],default='full');p.add_argument('--output',type=Path,required=True);p.add_argument('--admission',type=Path);main(p.parse_args())
