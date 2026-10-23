import test from 'node:test';
import assert from 'node:assert/strict';
import {harmonicCoefficient,analyzeSolidHarmonics,projectHarmonicProbe} from '../public/solid-harmonics.js';
import {SolidService} from '../server/solid.mjs';
import {validateControlSet} from '../server/solid-studies.mjs';
import {mkdtemp,mkdir,writeFile,rm,readFile} from 'node:fs/promises';
import {buildSolidHarmonicReport} from '../public/solid-harmonic-report.js';
import {tmpdir} from 'node:os';import {join} from 'node:path';

function sample(){
  const times=Array.from({length:1001},(_,i)=>i*.05),f=.2;
  const linear=times.map(t=>1e-7*Math.cos(2*Math.PI*f*t)+.2e-9*Math.cos(4*Math.PI*f*t));
  const geometric=times.map(t=>.3e-9*Math.cos(4*Math.PI*f*t+.4)),material=times.map(t=>2e-9*Math.cos(4*Math.PI*f*t+1.2));
  const signals=[linear,linear.map((v,i)=>v+geometric[i]),linear.map((v,i)=>v+geometric[i]+material[i]),linear.map((v,i)=>2*v+4*geometric[i]),linear.map((v,i)=>2*v+4*(geometric[i]+material[i]))];
  const names=['linear_A','geometric_A','material_A','geometric_2A','material_2A'];
  return {schema:'workai.solid.harmonic-probe.v1',study_id:'study-fixture',point_mm:[1,2,3],direction:[1,0,0],same_mesh_and_time_step:true,times_us:times,scene:{source:{frequency_mhz:f},acquisition:{sample_step_us:.05}},controls:Object.fromEntries(names.map((role,i)=>[role,{xyz_m:signals[i].map(v=>[v,2*v,0]),max_export_roundoff_m:1e-15}]))};
}
test('complex harmonic extraction separates a phased material increment and checks quadratic scaling',()=>{
  const p=sample(),a=analyzeSolidHarmonics(p,0,50);
  assert.ok(Math.abs(a.material_second.amplitude_m/2e-9-1)<1e-10);
  assert.ok(Math.abs(a.material_second.phase_deg-1.2*180/Math.PI)<1e-8);
  assert.ok(Math.abs(a.amplitude_ratio_2A_A-4)<1e-10);assert.ok(a.relative_complex_quadratic_error<1e-10);
  assert.equal(a.quadratic_scaling_within_5_percent,true);assert.equal(a.harmonic_convergence_verified,false);
  assert.ok(Math.abs(a.linear_second.amplitude_m/.2e-9-1)<1e-8);
  const negative=analyzeSolidHarmonics(p,0,50,[-1,0,0]);assert.ok(Math.abs(negative.material_second.re_m+a.material_second.re_m)<1e-22);
  const y=projectHarmonicProbe(p,[0,1,0]);assert.equal(y.controls.material_A.projected_m[13],2*p.controls.material_A.xyz_m[13][0]);
});
test('equal magnitudes with opposite phases do not disappear in material subtraction',()=>{
  const p=sample();p.controls.geometric_A.xyz_m=p.times_us.map(t=>[1e-9*Math.cos(4*Math.PI*.2*t),0,0]);p.controls.material_A.xyz_m=p.times_us.map(t=>[-1e-9*Math.cos(4*Math.PI*.2*t),0,0]);
  const a=analyzeSolidHarmonics(p,0,50);assert.ok(Math.abs(a.material_second.amplitude_m/2e-9-1)<1e-12);
  assert.equal(a.quadratic_scaling_within_5_percent,false);
});
test('roundoff-dominated or incomplete harmonic records never pass acceptance',()=>{
  const p=sample();for(const c of Object.values(p.controls))c.max_export_roundoff_m=1e-9;
  const a=analyzeSolidHarmonics(p,0,50);assert.equal(a.amplitude_ratio_2A_A,null);assert.equal(a.quadratic_scaling_within_5_percent,false);
  assert.throws(()=>analyzeSolidHarmonics(p,0,1));assert.throws(()=>analyzeSolidHarmonics(p,-1,50));assert.throws(()=>analyzeSolidHarmonics(p,0,51));
  assert.throws(()=>projectHarmonicProbe(p,[0,0,0]));assert.throws(()=>harmonicCoefficient([0,1,2,4],[0,1,0,0],.1,0,4));
});
test('a running study blocks new standalone solves even between child jobs',async t=>{
  const root=await mkdtemp(join(tmpdir(),'workai-study-lock-'));t.after(()=>rm(root,{recursive:true,force:true}));const service=new SolidService(root);await service.init();
  const id='study-'+'a'.repeat(32),path=service.studies.path(id);await mkdir(path);await writeFile(join(path,'status.json'),JSON.stringify({study_id:id,phase:'running'}));
  await assert.rejects(()=>service.submit({}),e=>e.status===409);
  assert.throws(()=>service.studies.path('../outside'));
});
test('study validation refuses missing controls before trusting mesh identity',()=>{
  assert.throws(()=>validateControlSet({schema:'workai.solid.nonlinear-study.v1',controls:[]},[]));
});

test('a changed control mesh or internal time step cannot masquerade as a material harmonic',()=>{
  const names=['linear_A','geometric_A','material_A','geometric_2A','material_2A'],scene={name:'fixture',constitutive:'murnaghan',geometry:{kind:'plate'},material:{rho:2700},source:{amplitude_mpa:1},acquisition:{sample_step_us:.1}};
  const controls=names.map((role,i)=>({role,job_id:'solid-'+String(i).repeat(32)}));const study={schema:'workai.solid.nonlinear-study.v1',scene,controls};
  const records=controls.map((c,i)=>({schema:'workai.hyperelastic.result.v1',engine:'workai_explicit_hyperelastic_3d',job_id:c.job_id,scene:{...structuredClone(scene),constitutive:['linear','stvk','murnaghan','stvk','murnaghan'][i],source:{amplitude_mpa:i>=3?2:1}},nodes_mm:[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],tetra:[[0,1,2,3]],surface:[[0,1,2]],times_us:Array.from({length:21},(_,i)=>i*.1),source_sha256:{physics:'fixture'},diagnostics:{internal_step_us:.01,max_export_roundoff_m:0},field:{dtype:'little_endian_float64',unit:'m',layout:'time,node,xyz',shape:[21,4,3],uncompressed_bytes:2016}}));
  validateControlSet(study,records);
  let changed=structuredClone(records);changed[2].nodes_mm[1][0]=1.01;assert.throws(()=>validateControlSet(study,changed));
  changed=structuredClone(records);changed[4].diagnostics.internal_step_us=.005;assert.throws(()=>validateControlSet(study,changed));
  changed=structuredClone(records);changed[3].scene.source.amplitude_mpa=1;assert.throws(()=>validateControlSet(study,changed));
});

test('offline harmonic export keeps the selected window and excludes unrelated session data',async()=>{
  const p=sample();p.analysis_window_us=[5,45];p.scene.name='</script><script>malicious()</script>';p.csrf_token='secret-not-for-export';
  const resources={};for(const [key,file] of Object.entries({css:'styles.css',data:'solid-data.js',dispersion:'plate-dispersion.js',propagation:'solid-propagation.js',harmonics:'solid-harmonics.js',viewer:'solid-harmonic-viewer.js'}))resources[key]=await readFile(new URL('../public/'+file,import.meta.url),'utf8');
  const html=buildSolidHarmonicReport(p,resources);assert.ok(!html.includes(p.csrf_token));assert.ok(!html.includes(p.scene.name));
  const match=html.match(/<script id="harmonic-record" type="application\/json">([\s\S]*?)<\/script>/);const record=JSON.parse(match[1]);assert.deepEqual(record.analysis_window_us,[5,45]);assert.equal(record.scene.name,p.scene.name);
  assert.throws(()=>buildSolidHarmonicReport({...p,analysis_window_us:[0,1]},resources));
});

test('numerical reference applies only to its exact receiver, direction, window and harmonic value',()=>{
  const p=sample(),a=analyzeSolidHarmonics(p,0,50);p.numerical_reference={schema:'workai.solid.harmonic-validation.v1',threshold:.05,references:[{level:'timecheck',study_id:p.study_id}],rows:[{point_mm:p.point_mm,direction:p.direction,window_us:[0,50],material_second_nm:a.material_second.amplitude_m*1e9,spatial_with_changed_stable_step:.01,temporal_same_mesh:.001}]};
  assert.equal(analyzeSolidHarmonics(p,0,50).selected_reference.all_selected_checks_within_5_percent,true);
  assert.equal(analyzeSolidHarmonics(p,0,50,[-1,0,0]).selected_reference,null);
  assert.equal(analyzeSolidHarmonics(p,5,45).selected_reference,null);
  const changed=structuredClone(p);changed.point_mm=[...p.point_mm];changed.point_mm[0]+=1;assert.equal(analyzeSolidHarmonics(changed,0,50).selected_reference,null);
  p.numerical_reference.rows[0].spatial_with_changed_stable_step=.0502589057593498;
  assert.equal(analyzeSolidHarmonics(p,0,50).selected_reference.all_selected_checks_within_5_percent,false,'a value slightly over 5% must not pass after display rounding');
});
