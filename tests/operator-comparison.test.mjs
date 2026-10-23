import test from 'node:test';
import assert from 'node:assert/strict';
import {comparisonSeries} from '../public/operator-comparison.js';

function signalFixture(){
  const times=Array.from({length:401},(_,i)=>i*.05),frequency=.5;
  const field=(shift=0)=>times.map(t=>{const s=Math.sin(2*Math.PI*frequency*t),h=Math.sin(4*Math.PI*frequency*t+shift);return [[s*1e-9,2*s*1e-9,-s*1e-9],[0,0,0],[h*1e-11,2*h*1e-11,-h*1e-11]];});
  return {record:{times_us:times,frequency_mhz:frequency},receiver:{predicted_parts_m:field(),fem_parts_m:field()}};
}

test('AI comparison projects all components in metres and distinguishes response increments',()=>{
  const {record,receiver}=signalFixture();const direction=[1,2,-1],norm=Math.sqrt(6),full=comparisonSeries(record,receiver,direction,'full'),material=comparisonSeries(record,receiver,direction,'material');
  assert.ok(full.relative_l2<1e-15);assert.ok(material.second_complex_relative_error<1e-14);assert.ok(Math.abs(material.fem_2f.amplitude_m-1e-11*norm)<1e-18);
  assert.ok(Math.abs(full.fem_m[13]-norm*(Math.sin(2*Math.PI*.5*record.times_us[13])*1e-9+Math.sin(4*Math.PI*.5*record.times_us[13])*1e-11))<1e-22);
  const opposite=comparisonSeries(record,receiver,direction.map(x=>-x),'material');assert.ok(opposite.fem_m.every((v,i)=>Math.abs(v+material.fem_m[i])<1e-25));assert.equal(opposite.fem_spectrum.fft_size,512);
});

test('AI comparison detects phase error even when 2f magnitudes agree',()=>{
  const {record,receiver}=signalFixture();receiver.predicted_parts_m=receiver.predicted_parts_m.map((parts,i)=>[parts[0],parts[1],[Math.sin(4*Math.PI*.5*record.times_us[i]+Math.PI/2)*1e-11,0,0]]);
  const s=comparisonSeries(record,receiver,[1,0,0],'material');assert.ok(Math.abs(s.predicted_2f.amplitude_m/s.fem_2f.amplitude_m-1)<1e-5);assert.ok(Math.abs(s.second_complex_relative_error-Math.sqrt(2))<1e-5);
});

test('AI comparison refuses empty direction, mismatched fields and nonuniform times',()=>{
  const {record,receiver}=signalFixture();assert.throws(()=>comparisonSeries(record,receiver,[0,0,0],'full'),/零向量/);assert.throws(()=>comparisonSeries(record,receiver,[1,0,0],'unknown'));
  assert.throws(()=>comparisonSeries(record,{...receiver,fem_parts_m:receiver.fem_parts_m.slice(1)},[1,0,0],'full'),/不完整/);
  const broken=structuredClone(record);broken.times_us[20]+=.01;assert.throws(()=>comparisonSeries(broken,receiver,[1,0,0],'full'),/采样/);
});
