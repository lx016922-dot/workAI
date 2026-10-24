import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,rm,symlink} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {createHash} from 'node:crypto';
import {PlaneService} from '../server/plane.mjs';
import {createApp} from '../server/app.mjs';
const sha=b=>createHash('sha256').update(b).digest('hex');

test('plane-strain results require completed identity, scene/settings hashes and binary digests',async t=>{
 const root=await mkdtemp(join(tmpdir(),'workai-plane-service-'));t.after(()=>rm(root,{recursive:true,force:true}));const service=new PlaneService(root,{});await service.init();
 assert.throws(()=>service.path('../private'));assert.throws(()=>service.path('spectral-'+'a'.repeat(32)));const id='plane-'+'a'.repeat(32),path=service.path(id);await mkdir(path);
 const scene=Buffer.from('{}'),settings=Buffer.from('{"degree":4}'),field=Buffer.from('field fixture'),lines=Buffer.from('lines fixture');
 await writeFile(join(path,'scene.json'),scene);await writeFile(join(path,'settings.json'),settings);await writeFile(join(path,'field.bin.gz'),field);await writeFile(join(path,'lines.bin.gz'),lines);
 const record={schema:'workai.plane_strain.result.v1',engine:'workai_spectral_plane_strain_2d',job_id:id,accepted_for_solver_routing:false,scene_sha256:sha(scene),settings_sha256:sha(settings),
  field:{name:'field.bin.gz',compressed_bytes:field.length,sha256:sha(field)},lines:{name:'lines.bin.gz',compressed_bytes:lines.length,sha256:sha(lines)}};
 const bytes=JSON.stringify(record);await writeFile(join(path,'result.json'),bytes);await writeFile(join(path,'status.json'),JSON.stringify({job_id:id,phase:'running'}));
 await assert.rejects(()=>service.result(id),/尚未完成/);await assert.rejects(()=>service.assertIdle(),/正在运行/);
 await writeFile(join(path,'status.json'),JSON.stringify({job_id:id,phase:'completed',result_sha256:sha(bytes)}));await service.assertIdle();
 assert.deepEqual(await service.binary(id,'field'),field);assert.deepEqual(await service.binary(id,'lines'),lines);await assert.rejects(()=>service.binary(id,'scene'),/没有此数据文件/);
 await writeFile(join(path,'lines.bin.gz'),'changed');await assert.rejects(()=>service.binary(id,'lines'),/摘要不匹配/);
 await writeFile(join(path,'settings.json'),'changed');await assert.rejects(()=>service.result(id),/阶次设置摘要/);
 await rm(join(path,'result.json'));await symlink(join(path,'scene.json'),join(path,'result.json'));await assert.rejects(()=>service.result(id),/文件无效/);
 assert.equal((await service.list())[0].job_id,id);
});

test('plane-strain submission rejects wrong shapes and non-plane scenes before creating jobs',async t=>{
 const root=await mkdtemp(join(tmpdir(),'workai-plane-input-'));t.after(()=>rm(root,{recursive:true,force:true}));
 let scene={schema:'workai.hyperelastic.solid.v2',dimension:'3d',geometry:{kind:'plate'},numerics:{h_mm:1}};const input={numerics:{h_mm:1}};let validated=null;
 const service=new PlaneService(root,{exclusive:fn=>fn(),assertIdle:async()=>{},command:async(action,value)=>{validated=value;return action==='templates'?{plate:{schema:'workai.hyperelastic.solid.v1',name:'铝板',geometry:{kind:'plate',length_mm:12,width_mm:4,thickness_mm:2},source:{face:'xmin'},numerics:{h_mm:.7,cfl:.3,max_elements:40000},receivers:[]}}:scene;}});
 await service.init();
 await assert.rejects(()=>service.submit({scene:input,degree:7}),/阶次/);await assert.rejects(()=>service.submit({scene:input}),/阶次/);
 await assert.rejects(()=>service.submit({scene:{numerics:{h_mm:0}},degree:4}),/单元尺寸/);
 await assert.rejects(()=>service.submit({scene:input,degree:4}),/2d_plane_strain/);assert.equal(validated.numerics.h_mm,.25);assert.deepEqual(await service.list(),[]);
 const template=(await service.templates()).plate;assert.equal(template.dimension,'2d_plane_strain');assert.equal(template.schema,'workai.hyperelastic.solid.v2');assert.equal(template.receiver_lines[0].count,111);assert.equal(template.boundaries[0].type,'absorbing');
 await assert.rejects(()=>service.dispersion('abc'),/板厚/);await assert.rejects(()=>service.dispersion(0),/板厚/);
});

test('plane-strain routes share session and origin checks and serve the viewer modules',async t=>{
 let calls=0;const id='plane-'+'b'.repeat(32),plane={list:async()=>[],submit:async()=>{calls++;return {job_id:id};},templates:async()=>({plate:{name:'t'}}),dispersion:async thickness=>({thickness_mm:Number(thickness),modes:[]}),status:async()=>({job_id:id,phase:'running'})};
 const server=createApp({plane});await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));t.after(async()=>{server.closeAllConnections();await new Promise(resolve=>server.close(resolve));});const origin='http://127.0.0.1:'+server.address().port;
 for(const file of ['/plane-data.js','/plane-viewer.js','/plane-ui.js'])assert.equal((await fetch(origin+file)).status,200,file);
 assert.deepEqual(await (await fetch(origin+'/api/plane/jobs')).json(),{jobs:[]});assert.deepEqual(await (await fetch(origin+'/api/plane/templates')).json(),{templates:{plate:{name:'t'}}});
 assert.deepEqual(await (await fetch(origin+'/api/plane/dispersion?thickness_mm=2')).json(),{thickness_mm:2,modes:[]});
 assert.equal((await fetch(origin+'/api/plane/jobs/'+id)).status,200);assert.equal((await fetch(origin+'/api/plane/jobs/not-an-id')).status,404);
 const {csrf_token}=await (await fetch(origin+'/api/bootstrap')).json();const post=token=>fetch(origin+'/api/plane/jobs',{method:'POST',headers:{Origin:origin,'Content-Type':'application/json','X-WorkAI-CSRF':token},body:'{}'});
 assert.equal((await post('')).status,403);assert.equal(calls,0);assert.equal((await post(csrf_token)).status,202);assert.equal(calls,1);
 const page=await (await fetch(origin+'/')).text();assert.match(page,/二维平面应变/);
});
