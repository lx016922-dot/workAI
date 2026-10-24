import test from 'node:test';
import assert from 'node:assert/strict';
import {gzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {validatePlaneResult,decodePlaneResult,harmonicCoefficients,lineHarmonics,accumulationFit,lineVelocities,fkSpectrum,fft} from '../public/plane-data.js';
const sha=b=>createHash('sha256').update(b).digest('hex');

function fixture({frames=64,points=8,degree=2}={}) {
  const step=.05,times=Array.from({length:frames},(_,i)=>i*step),size=degree+1;
  const nodes=[];for(let i=0;i<size;i++)for(let j=0;j<size;j++)nodes.push([i*.5,j*.5]);
  const elements=[Array.from({length:size*size},(_,i)=>i)];
  const cp=5000,f=.5,k=2*Math.PI*f*1e6/cp,distance=Array.from({length:points},(_,i)=>20+2*i);
  const lines=new Float64Array(frames*points*2);
  for(let t=0;t<frames;t++)for(let p=0;p<points;p++){const x=distance[p]*1e-3,tt=times[t]*1e-6,env=Math.exp(-(((tt-x/4500-.8e-6)/.6e-6)**2));lines[(t*points+p)*2]=1e-9*env*Math.cos(k*x-2*Math.PI*f*1e6*tt);lines[(t*points+p)*2+1]=0;}
  const field=new Float32Array(2*nodes.length*2).map((_,i)=>i*1e-12);
  const lineRaw=Buffer.from(lines.buffer),fieldRaw=Buffer.from(field.buffer),lineGz=gzipSync(lineRaw),fieldGz=gzipSync(fieldRaw);
  const result={schema:'workai.plane_strain.result.v1',engine:'workai_spectral_plane_strain_2d',dimension:'2d_plane_strain',accepted_for_solver_routing:false,job_id:'plane-'+'c'.repeat(32),degree,
    scene:{acquisition:{sample_step_us:step},source:{frequency_mhz:f,cycles:2,delay_us:0},geometry:{thickness_mm:2},name:'fixture'},
    mesh:{nodes_mm:nodes,elements,degree,h_mm:1,boundary_lines:[[0,1],[1,2]]},times_us:times,diagnostics:{},
    receivers:[{name:'P',point_mm:[1,0,1],direction:[1,0,0],u_x_m:times.map(t=>1e-9*Math.sin(t)),u_z_m:times.map(()=>0)}],
    receiver_lines:[{name:'top',direction:[1,0,0],count:points,columns:Array.from({length:points},(_,i)=>i),points_mm:distance.map(x=>[x,0,2]),distance_mm:distance}],
    lines:{name:'lines.bin.gz',layout:'time,point,xz',dtype:'little_endian_float64',unit:'m',shape:[frames,points,2],compressed_bytes:lineGz.length,uncompressed_bytes:lineRaw.length,sha256:sha(lineGz),uncompressed_sha256:sha(lineRaw)},
    field:{name:'field.bin.gz',layout:'frame,node,xz',dtype:'little_endian_float32',unit:'m',shape:[2,nodes.length,2],frame_indices:[0,frames-1],every_n_frames:frames-1,compressed_bytes:fieldGz.length,uncompressed_bytes:fieldRaw.length,sha256:sha(fieldGz),uncompressed_sha256:sha(fieldRaw)}};
  return {result,lineGz,fieldGz,cp,f};
}

test('plane-strain records validate, decode with digests, and reject tampering',async()=>{
  const {result,lineGz,fieldGz}=fixture();
  const decoded=await decodePlaneResult(result,fieldGz.buffer.slice(fieldGz.byteOffset,fieldGz.byteOffset+fieldGz.byteLength),lineGz.buffer.slice(lineGz.byteOffset,lineGz.byteOffset+lineGz.byteLength));
  assert.equal(decoded.lines.length,result.lines.shape[0]*result.lines.shape[1]*2);assert.equal(decoded.field.length,2*result.mesh.nodes_mm.length*2);
  const broken=structuredClone(result);broken.lines.sha256='0'.repeat(64);await assert.rejects(()=>decodePlaneResult(broken,null,lineGz.buffer.slice(lineGz.byteOffset,lineGz.byteOffset+lineGz.byteLength)),/SHA-256/);
  for(const mutate of [r=>r.engine='other',r=>r.mesh.elements[0][0]=999,r=>r.receiver_lines[0].columns[0]=99,r=>r.field.frame_indices=[0,0],r=>r.times_us[3]+=1e-3,r=>r.lines.shape=[1,1,2]]){const bad=structuredClone(result);mutate(bad);assert.throws(()=>validatePlaneResult(bad));}
});

test('harmonic coefficients, line velocities and f–k peak recover the synthetic wave',async()=>{
  const {result,lineGz,cp,f}=fixture({frames:800,points:16});   // 40 us record: fundamental leakage into 2f stays below 1e-4
  const {lines}=await decodePlaneResult(result,null,lineGz.buffer.slice(lineGz.byteOffset,lineGz.byteOffset+lineGz.byteLength));
  const times=result.times_us,values=times.map(t=>3e-9*Math.cos(2*Math.PI*f*t+.4)+5e-11*Math.cos(2*Math.PI*2*f*t-1));
  const h=harmonicCoefficients(times,values,f).harmonics;
  assert.ok(Math.abs(h[0].amplitude_m/3e-9-1)<1e-3);assert.ok(Math.abs(h[0].phase_deg*Math.PI/180-.4)<2e-3);assert.ok(Math.abs(h[1].amplitude_m/5e-11-1)<1e-2,String(h[1].amplitude_m));
  const rows=lineHarmonics(result,lines,result.receiver_lines[0],f,{mode:'moving',group_velocity:4500,burst_us:2.4,delay_us:0});
  assert.equal(rows.length,16);const v=lineVelocities(rows,f);assert.ok(Math.abs(v.phase_velocity_m_s-cp)/cp<.01,String(v.phase_velocity_m_s));assert.ok(v.phase_fit_r_squared>.999);
  const fit=accumulationFit(rows);assert.equal(fit.available,true);assert.equal(fit.a1.length,16);
  const fk=fkSpectrum(result,lines,result.receiver_lines[0],2);let best=[-1,0,0];for(let j=0;j<fk.rows;j++)for(let i=0;i<fk.columns;i++){const m=fk.magnitude(i,j);if(m>best[0])best=[m,i,j];}
  assert.ok(Math.abs(fk.frequency_mhz[best[1]]-f)<.08);const kTheory=2*Math.PI*f*1e6/cp*1e-3;assert.ok(fk.wavenumber_rad_per_mm[best[2]]>0&&Math.abs(fk.wavenumber_rad_per_mm[best[2]]-kTheory)<.08,String(fk.wavenumber_rad_per_mm[best[2]]));
  const re=new Float64Array([1,0,0,0]),im=new Float64Array(4);fft(re,im);assert.deepEqual(Array.from(re),[1,1,1,1]);
});
