import test from 'node:test';
import assert from 'node:assert/strict';
import { amplitudeSpectrum, pointSeries, locateSolidPoint, unitVector, validateSolidResult,decodeSolidField } from '../public/solid-data.js';
import {gzipSync} from 'node:zlib';
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {SolidService} from '../server/solid.mjs';
import {validateModalResult,decodeModalField} from '../public/modal-data.js';

test('3-D barycentric interpolation preserves affine displacements and arbitrary signed directions',()=>{
  const result={nodes_mm:[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],tetra:[[0,1,2,3]],times_us:[0,1,2]};
  const data=[];for(const t of result.times_us)for(const [x,y,z] of result.nodes_mm)data.push(t*(x+2*y)*1e-9,t*z*1e-9,t*(2-x)*1e-9);
  const positive=pointSeries(result,new Float64Array(data),[.2,.3,.1],[1,2,-1]);
  const exact=(.8+2*.1-1.8)/Math.sqrt(6)*1e-9;
  assert.ok(Math.abs(positive.projected_m[1]-exact)<1e-20);
  const negative=pointSeries(result,new Float64Array(data),[.2,.3,.1],[-1,-2,1]);
  assert.ok(Math.abs(positive.projected_m[2]+negative.projected_m[2])<1e-20);
  assert.throws(()=>locateSolidPoint(result,[1,1,1]));assert.throws(()=>unitVector([0,0,0]));
});

test('Hann single-sided amplitude spectrum has physical units and correct Nyquist scaling',()=>{
  const count=1024,step=.05,frequency=40/(count*step),amplitude=7e-9;
  const values=Array.from({length:count},(_,i)=>2e-9+amplitude*Math.sin(2*Math.PI*40*i/count));
  const s=amplitudeSpectrum(values,step);
  assert.ok(Math.abs(s.frequency_mhz[40]-frequency)<1e-12);assert.ok(Math.abs(s.amplitude_m[40]/amplitude-1)<1e-5);
  assert.ok(s.amplitude_m[0]<1e-22);assert.equal(s.nyquist_mhz,10);
  const nyquist=amplitudeSpectrum(values.map((_,i)=>amplitude*(-1)**i),step);
  assert.ok(Math.abs(nyquist.amplitude_m[512]/amplitude-1)<1e-8);
  assert.throws(()=>amplitudeSpectrum([0,NaN,1,2],step));
});

test('solid metadata rejects mismatched field dimensions and invalid mesh indices',()=>{
  const r={schema:'workai.hyperelastic.result.v1',engine:'workai_explicit_hyperelastic_3d',nodes_mm:[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],tetra:[[0,1,2,3]],surface:[[0,1,2]],times_us:Array.from({length:21},(_,i)=>i*.1),scene:{acquisition:{sample_step_us:.1}},field:{dtype:'little_endian_float32',unit:'m',layout:'time,node,xyz',shape:[21,4,3],uncompressed_bytes:1008}};
  assert.equal(validateSolidResult(r),r);
  const bad=structuredClone(r);bad.tetra[0][3]=7;assert.throws(()=>validateSolidResult(bad));
  assert.throws(()=>validateSolidResult({...r,field:{...r.field,unit:'nm'}}));
});

test('verified float64 field transport preserves weak increments hidden by float32 rounding',async()=>{
  const values=new Float64Array(21*4*3);values.fill(1e-7);values[1]+=1e-15;
  assert.equal(Math.fround(values[0]),Math.fround(values[1]));
  const raw=Buffer.from(values.buffer),packed=gzipSync(raw),sha=b=>createHash('sha256').update(b).digest('hex');
  const r={schema:'workai.hyperelastic.result.v1',engine:'workai_explicit_hyperelastic_3d',nodes_mm:[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],tetra:[[0,1,2,3]],surface:[[0,1,2]],times_us:Array.from({length:21},(_,i)=>i*.1),scene:{acquisition:{sample_step_us:.1}},field:{dtype:'little_endian_float64',unit:'m',layout:'time,node,xyz',shape:[21,4,3],uncompressed_bytes:raw.length,compressed_bytes:packed.length,sha256:sha(packed),uncompressed_sha256:sha(raw)}};
  const field=await decodeSolidField(r,packed);assert.ok(field instanceof Float64Array);assert.deepEqual(field,values);
  await assert.rejects(()=>decodeSolidField({...r,field:{...r.field,uncompressed_sha256:'0'.repeat(64)}},packed));
});

test('hybrid predictions cannot be accepted as FEM fields or promote themselves',async()=>{
  const values=new Float64Array(21*4*3);values.fill(1e-12);const raw=Buffer.from(values.buffer),packed=gzipSync(raw),sha=b=>createHash('sha256').update(b).digest('hex');
  const r={schema:'workai.modal_prediction.result.v1',engine:'workai_modal_energy_3d',accepted_for_solver_routing:false,response_part:'material',evidence:{run_id:'test',checkpoint_sha256:'a'.repeat(64),transient_labels_used_for_prediction:false},nodes_mm:[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],tetra:[[0,1,2,3]],surface:[[0,1,2]],times_us:Array.from({length:21},(_,i)=>i*.1),scene:{acquisition:{sample_step_us:.1}},field:{dtype:'little_endian_float64',unit:'m',layout:'time,node,xyz',shape:[21,4,3],uncompressed_bytes:raw.length,compressed_bytes:packed.length,sha256:sha(packed),uncompressed_sha256:sha(raw)}};
  assert.equal(validateModalResult(r),r);assert.throws(()=>validateSolidResult(r));assert.deepEqual(await decodeModalField(r,packed),values);assert.throws(()=>validateModalResult({...r,accepted_for_solver_routing:true}));assert.throws(()=>validateModalResult({...r,evidence:{...r.evidence,transient_labels_used_for_prediction:true}}));await assert.rejects(()=>decodeModalField({...r,field:{...r.field,sha256:'0'.repeat(64)}},packed));
});

test('CAD storage is content-addressed and rejects code uploads and changed geometry bytes',async t=>{
  const root=await mkdtemp(join(tmpdir(),'workai-solid-'));t.after(()=>rm(root,{recursive:true,force:true}));const service=new SolidService(root);await service.init();
  const data=Buffer.from('ISO-10303-21;\nHEADER;\nENDSEC;\nEND-ISO-10303-21;\n');
  const body={filename:'test.step',data_base64:data.toString('base64')},asset=await service.upload(body);
  assert.equal(asset.sha256,createHash('sha256').update(data).digest('hex'));
  assert.deepEqual(await service.upload(body),asset);
  await assert.rejects(()=>service.upload({...body,filename:'program.py'}));
  await assert.rejects(()=>service.upload({...body,path:'/tmp/untrusted'}));
  await writeFile(join(root,'solid/assets',asset.asset_id,'geometry.step'),'changed');
  await assert.rejects(()=>service.upload(body));
});

test('solid result API binds completed status, original scene and field bytes',async t=>{
  const root=await mkdtemp(join(tmpdir(),'workai-solid-result-'));t.after(()=>rm(root,{recursive:true,force:true}));const service=new SolidService(root);await service.init();
  const id='solid-'+'a'.repeat(32),dir=join(root,'solid/jobs',id);await mkdir(dir);const scene='{"input":"retained"}',field=Buffer.from('field fixture'),hash=b=>createHash('sha256').update(b).digest('hex');
  const result=JSON.stringify({schema:'workai.hyperelastic.result.v1',job_id:id,scene_sha256:hash(scene),field:{sha256:hash(field),compressed_bytes:field.length}});
  await writeFile(join(dir,'scene.json'),scene);await writeFile(join(dir,'result.json'),result);await writeFile(join(dir,'field.bin.gz'),field);await writeFile(join(dir,'status.json'),JSON.stringify({job_id:id,phase:'completed',result_sha256:hash(result)}));
  assert.deepEqual(await service.field(id),field);
  await writeFile(join(dir,'field.bin.gz'),'bad');await assert.rejects(()=>service.field(id));
  await writeFile(join(dir,'scene.json'),'changed');await assert.rejects(()=>service.result(id));
  assert.throws(()=>service.path('../private'));
});
