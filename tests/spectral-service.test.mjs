import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,rm,symlink} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {createHash} from 'node:crypto';
import {SpectralService} from '../server/spectral.mjs';
import {createApp} from '../server/app.mjs';
const sha=b=>createHash('sha256').update(b).digest('hex');

test('high-order results require completed identity, scene/settings and field hashes',async t=>{
 const root=await mkdtemp(join(tmpdir(),'workai-spectral-service-'));t.after(()=>rm(root,{recursive:true,force:true}));const service=new SpectralService(root,{});await service.init();
 assert.throws(()=>service.path('../private'));const id='spectral-'+'a'.repeat(32),path=service.path(id);await mkdir(path);
 const scene=Buffer.from('{}'),settings=Buffer.from('{"degree":3}'),field=Buffer.from('compressed fixture');
 await writeFile(join(path,'scene.json'),scene);await writeFile(join(path,'settings.json'),settings);await writeFile(join(path,'field.bin.gz'),field);
 const record={schema:'workai.spectral_reference.result.v1',engine:'workai_spectral_reference_3d',job_id:id,accepted_for_solver_routing:false,scene_sha256:sha(scene),settings_sha256:sha(settings),field:{compressed_bytes:field.length,sha256:sha(field)}};
 const bytes=JSON.stringify(record);await writeFile(join(path,'result.json'),bytes);await writeFile(join(path,'status.json'),JSON.stringify({job_id:id,phase:'running'}));await assert.rejects(()=>service.result(id),/尚未完成/);await assert.rejects(()=>service.assertIdle(),/正在运行/);
 await writeFile(join(path,'status.json'),JSON.stringify({job_id:id,phase:'completed',result_sha256:sha(bytes)}));assert.deepEqual(await service.field(id),field);await service.assertIdle();
 await writeFile(join(path,'field.bin.gz'),'changed');await assert.rejects(()=>service.field(id),/位移场摘要/);
 await writeFile(join(path,'settings.json'),'changed');await assert.rejects(()=>service.result(id),/阶次设置摘要/);
 await rm(join(path,'result.json'));await symlink(join(path,'scene.json'),join(path,'result.json'));await assert.rejects(()=>service.result(id),/文件无效/);
});

test('high-order submission rejects unsupported inputs before creating jobs',async t=>{
 const root=await mkdtemp(join(tmpdir(),'workai-spectral-input-'));t.after(()=>rm(root,{recursive:true,force:true}));let scene={geometry:{kind:'cad'},source:{face:'xmin',radius_mm:0},boundary:{fixed_face:'none'},numerics:{h_mm:2}};const input={numerics:{h_mm:2}};
 const service=new SpectralService(root,{exclusive:fn=>fn(),assertIdle:async()=>{},command:async()=>scene});await service.init();
 await assert.rejects(()=>service.submit({scene:input,degree:6}),/阶设置/);await assert.rejects(()=>service.submit({scene:input,degree:3}),/参数化/);
 scene.geometry.kind='plate';scene.source.radius_mm=1;await assert.rejects(()=>service.submit({scene:input,degree:3}),/完整 X 端面/);assert.deepEqual(await service.list(),[]);
});

test('high-order routes share session and origin checks and expose the decoder',async t=>{
 let calls=0;const id='spectral-'+'b'.repeat(32),spectral={list:async()=>[],submit:async()=>{calls++;return {job_id:id};}};const server=createApp({spectral});await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));t.after(async()=>{server.closeAllConnections();await new Promise(resolve=>server.close(resolve));});const origin='http://127.0.0.1:'+server.address().port;
 assert.equal((await fetch(origin+'/spectral-data.js')).status,200);assert.deepEqual(await (await fetch(origin+'/api/spectral/jobs')).json(),{jobs:[]});
 const {csrf_token}=await (await fetch(origin+'/api/bootstrap')).json();const post=token=>fetch(origin+'/api/spectral/jobs',{method:'POST',headers:{Origin:origin,'Content-Type':'application/json','X-WorkAI-CSRF':token},body:'{}'});
 assert.equal((await post('')).status,403);assert.equal(calls,0);assert.equal((await post(csrf_token)).status,202);assert.equal(calls,1);
});
