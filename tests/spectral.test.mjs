import test from 'node:test';
import assert from 'node:assert/strict';
import {spectralPointSeries,validateSpectralResult} from '../public/spectral-data.js';

test('high-order receiver reproduces a quadratic rather than displaying linear interpolation',()=>{
  const nodes=[],ids=[];for(let i=0;i<3;i++)for(let j=0;j<3;j++)for(let k=0;k<3;k++){ids.push(nodes.length);nodes.push([i,j,k]);}
  const result={nodes_mm:nodes,times_us:[0,1],spectral:{degree:2,gll:[-1,0,1],elements:[ids],origins_mm:[[0,0,0]],sizes_mm:[[2,2,2]]}};
  const field=Float64Array.from([0,1].flatMap(t=>nodes.flatMap(([x,y,z])=>[t*(x*x+2*y-z*z),t*y*y,t*x*z])));
  const point=[.31,1.42,.79],probe=spectralPointSeries(result,field,point,[2,-3,1]),expected=[point[0]**2+2*point[1]-point[2]**2,point[1]**2,point[0]*point[2]];
  expected.forEach((v,i)=>assert.ok(Math.abs(v-probe.xyz_m[1][i])<1e-12));assert.ok(Math.abs(probe.projected_m[1]-(2*expected[0]-3*expected[1]+expected[2])/Math.sqrt(14))<1e-12);
  assert.throws(()=>spectralPointSeries(result,field,[2.1,1,1],[1,0,0]),/实体之外/);
  assert.throws(()=>spectralPointSeries(result,field,point,[0,0,0]),/零向量/);
  assert.throws(()=>validateSpectralResult({...result,schema:'workai.hyperelastic.result.v1'}),/格式/);
});

test('curved isoparametric receiver inverts geometry and preserves affine physical displacement',()=>{
  const nodes=[],ids=[],gll=[-1,0,1];
  for(const x of gll)for(const y of gll)for(const z of gll){ids.push(nodes.length);nodes.push([2+x,.8+.2*y+.04*x*x,.5+.2*z+.02*y*y]);}
  const result={nodes_mm:nodes,times_us:[0,1],spectral:{degree:2,gll,elements:[ids],mapping_kind:'isoparametric'}};
  const field=Float64Array.from([0,1].flatMap(t=>nodes.flatMap(([x,y,z])=>[t*(x+2*y),t*(3*z-x),t*y]))),q=[.27,-.42,.61],point=[2+q[0],.8+.2*q[1]+.04*q[0]**2,.5+.2*q[2]+.02*q[1]**2];
  const probe=spectralPointSeries(result,field,point,[2,-3,1]),expected=[point[0]+2*point[1],3*point[2]-point[0],point[1]];
  expected.forEach((value,i)=>assert.ok(Math.abs(value-probe.xyz_m[1][i])<1e-11));
  assert.throws(()=>spectralPointSeries(result,field,[2,.1,.5],[1,0,0]),/管腔或孔洞/);
  assert.throws(()=>spectralPointSeries(result,field,[5,.8,.5],[1,0,0]),/实体之外/);
});
