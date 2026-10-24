import test from 'node:test';
import assert from 'node:assert/strict';
import {analyzeTimeWindow,amplitudeSpectrum} from '../offline/web/solid-data.js';
import {chart} from '../offline/web/solid-viewer.js';
import {analyzePointPath,pointPathCsv} from '../offline/web/solid-path.js';

test('5 ms analysis uses every sample; a 200001 point chart remains bounded',()=>{
 const times=Array.from({length:200001},(_,i)=>i*.025),values=times.map(t=>2e-9*Math.sin(2*Math.PI*.2*t));
 const a=analyzeTimeWindow(values,times,.2,0,5000);
 assert.equal(a.sample_count,200001);assert.ok(Math.abs(a.harmonics[0].amplitude_m-2e-9)<1e-18);
 assert.ok(chart(times,values,'time','u').length<180000);
 assert.throws(()=>analyzeTimeWindow(values,times,.2,0,5000.1));
});
test('chirp reports integrated frequency bands and omits fixed carrier coefficients',()=>{
 const times=Array.from({length:10001},(_,i)=>i*.2),values=times.map(t=>1e-9*Math.sin(2*Math.PI*.23*t));
 const a=analyzeTimeWindow(values,times,.2,0,2000,{kind:'chirp',end_frequency_mhz:.28});
 assert.deepEqual(a.harmonics,[]);assert.equal(a.bands.length,2);
 assert.ok(Math.abs(a.bands[0].windowed_rms_m-1e-9/Math.sqrt(2))<1e-16);
 assert.ok(a.bands[1].windowed_rms_m<1e-16);
});
test('multi-point analysis awaits complete records and exports their time grid',async()=>{
 const times=Array.from({length:25001},(_,i)=>i*.2);
 const result={job_id:'test',response_part:'full',times_us:[0,5000],recording:{},field:{sha256:'x'},scene:{source:{frequency_mhz:.2},signal:{kind:'hann'}}};
 const extract=async(_r,_f,point,direction)=>({times_us:times,xyz_m:times.map(t=>[Math.sin(t)*1e-9,0,0]),projected_m:times.map(t=>Math.sin(t)*1e-9)});
 const a=await analyzePointPath(result,null,extract,{mode:'segment',start:[1,0,0],end:[2,0,0],count:2,direction:[1,0,0],start_us:0,end_us:5000});
 assert.equal(a.times_us.length,25001);assert.equal(a.rows[0].sample_count,25001);
 assert.match(pointPathCsv(a),/2,5000,/);
 await assert.rejects(analyzePointPath(result,null,extract,{mode:'segment',start:[1,0,0],end:[2,0,0],count:9,direction:[1,0,0],start_us:0,end_us:5000}),/8/);
});
