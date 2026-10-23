import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {s0Dispersion} from '../public/plate-dispersion.js';
import {planPlatePropagation,comparePropagationProbes} from '../public/solid-propagation.js';
import {buildPropagationReport} from '../public/solid-propagation-report.js';

function scene(){return {name:'fixture',geometry:{kind:'plate',length_mm:60,width_mm:4,thickness_mm:1},source:{face:'xmin',radius_mm:0,direction:[1,0,0],frequency_mhz:.3,cycles:2,delay_us:0},boundary:{fixed_face:'none'},acquisition:{duration_us:15,sample_step_us:.05},material:{rho_kg_m3:2700,lambda_pa:58.1e9,mu_pa:26.1e9}};}
function probes(speedFactor=1){const s=scene(),times=Array.from({length:301},(_,i)=>i*.05),k=planPlatePropagation(s,[12,2,.5]).fundamental.wavenumber_rad_m/speedFactor;
  return [12,18,24].map(x=>{const linear=times.map(t=>1e-7*Math.cos(2*Math.PI*.3*t-k*x*.001)),geom=times.map(t=>2e-10*Math.sin(4*Math.PI*.3*t-2*k*x*.001)),material=times.map(t=>x*1e-11*Math.sin(4*Math.PI*.3*t-2*k*x*.001));
    const values=[linear,linear.map((v,i)=>v+geom[i]),linear.map((v,i)=>v+geom[i]+material[i]),linear.map((v,i)=>2*v+4*geom[i]),linear.map((v,i)=>2*v+4*(geom[i]+material[i]))];
    return {schema:'workai.solid.harmonic-probe.v1',study_id:'study-fixture',scene:s,point_mm:[x,2,.5],direction:[1,0,0],same_mesh_and_time_step:true,times_us:times,controls:Object.fromEntries(['linear_A','geometric_A','material_A','geometric_2A','material_2A'].map((role,i)=>[role,{xyz_m:values[i].map(v=>[v,0,0]),max_export_roundoff_m:0}]))};
  });}
test('S0 candidate tends to the independently derived long-wave plate speed',()=>{
  const m=scene().material,lam=m.lambda_pa,mu=m.mu_pa,nu=lam/(2*(lam+mu)),young=mu*(3*lam+2*mu)/(lam+mu),cp=Math.sqrt(young/(m.rho_kg_m3*(1-nu**2)));
  const a=s0Dispersion(1e4,.001,Math.sqrt((lam+2*mu)/m.rho_kg_m3),Math.sqrt(mu/m.rho_kg_m3));
  assert.ok(Math.abs(a.phase_velocity_m_s/cp-1)<1e-5);assert.ok(Math.abs(a.group_velocity_m_s/cp-1)<1e-5);
  assert.equal(s0Dispersion(2e6,.001,6300,3100),null);
});
test('moving windows are equal-duration, mirror with the source end, and screen short-record/echo cases',()=>{
  const s=scene(),plans=[12,18,24].map(x=>planPlatePropagation(s,[x,2,.5]));
  for(const p of plans){assert.equal(p.available,true);assert.equal(p.passes_end_echo_screen,true);assert.ok(Math.abs(p.window_us[1]-p.window_us[0]-10)<1e-12);assert.ok(p.coherence_length_mm>1000);}
  s.source.face='xmax';s.source.direction=[-1,0,0];assert.deepEqual(planPlatePropagation(s,[36,2,.5]),plans[2]);
  s.source.face='xmin';s.geometry.length_mm=30;assert.equal(planPlatePropagation(s,[24,2,.5]).passes_end_echo_screen,false);
  s.geometry.length_mm=60;s.acquisition.duration_us=10;assert.equal(planPlatePropagation(s,[24,2,.5]).record_covers_window,false);
  assert.equal(planPlatePropagation(s,[1,2,.5]).record_covers_window,false);
});
test('the plate candidate is never applied to holes, pipes, patches or transverse excitation',()=>{
  for(const kind of ['hole','pipe','notch','cad']){const s=scene();s.geometry.kind=kind;assert.equal(planPlatePropagation(s,[12,2,.5]).available,false);}
  const s=scene();s.source.radius_mm=1;assert.equal(planPlatePropagation(s,[12,2,.5]).available,false);
  s.source.radius_mm=0;s.source.direction=[0,0,1];assert.equal(planPlatePropagation(s,[12,2,.5]).available,false);
  assert.equal(planPlatePropagation(scene(),[12,2,2]).available,false);
  assert.equal(planPlatePropagation({},[12,2,.5]).available,false);
});
test('phase-consistent traveling signals pass only the candidate screen; a different speed fails it',()=>{
  const a=comparePropagationProbes(probes());assert.equal(a.candidate_phase_agreement_within_5_percent,true);assert.equal(a.pure_mode_accumulation_verified,false);assert.equal(a.selected_numerical_checks_pass,false);
  assert.ok(a.rows.every(r=>r.analysis.quadratic_scaling_within_5_percent));assert.ok(a.phase_checks.every(p=>Math.abs(p.relative_wavenumber_difference)<1e-4));
  assert.equal(comparePropagationProbes(probes(.9)).candidate_phase_agreement_within_5_percent,false);
  const mixed=probes();mixed[1].study_id='different';assert.throws(()=>comparePropagationProbes(mixed));
  const duplicated=probes();duplicated[1]=duplicated[0];assert.throws(()=>comparePropagationProbes(duplicated));
  const direction=probes();direction[1].direction=[0,1,0];assert.throws(()=>comparePropagationProbes(direction));
});
test('finite-width comparisons stay bound to the original points, windows and completed controls',()=>{
  const p=probes(),controls=['linear_A','geometric_A','material_A'].map((role,i)=>({role,job_id:'solid-'+i,result_sha256:'sha-'+i}));
  const reference={schema:'workai.rectangular-waveguide.validation.v1',study_id:p[0].study_id,frequency_mhz:.3,cross_section_mm:{width:4,thickness:1},material:p[0].scene.material,modal_purity_verified:false,transient_references:controls,rows:p.map(r=>({point_mm:r.point_mm,window_us:planPlatePropagation(r.scene,r.point_mm).window_us}))};
  for(const item of p){item.waveguide_reference=structuredClone(reference);for(const r of controls)Object.assign(item.controls[r.role],r);}
  assert.deepEqual(comparePropagationProbes(p).waveguide_reference,reference);
  for(const mutate of [q=>q[1].point_mm[0]+=.1,q=>q[1].controls.material_A.result_sha256='changed',q=>q.forEach(r=>r.waveguide_reference.rows[0].window_us[0]+=.1),q=>q.forEach(r=>r.waveguide_reference.cross_section_mm.width=5),q=>q.forEach(r=>r.waveguide_reference.modal_purity_verified=true)]){
    const changed=structuredClone(p);mutate(changed);assert.equal(comparePropagationProbes(changed).waveguide_reference,null);
  }
});
test('multi-point offline export keeps actual controls but excludes unrelated session fields and script injection',async()=>{
  const p=probes();p[0].session_secret='not-for-export';p.forEach(r=>r.scene.name='</script><script>bad()</script>');const resources={};
  for(const [key,file] of Object.entries({css:'styles.css',data:'solid-data.js',dispersion:'plate-dispersion.js',harmonics:'solid-harmonics.js',propagation:'solid-propagation.js',viewer:'solid-harmonic-viewer.js',propagationViewer:'solid-propagation-viewer.js'}))resources[key]=await readFile(new URL('../public/'+file,import.meta.url),'utf8');
  const html=buildPropagationReport(p,resources);assert.ok(!html.includes('not-for-export'));assert.ok(!html.includes(p[0].scene.name));assert.ok(!/^import /m.test(html));
  const saved=JSON.parse(html.match(/<script id="propagation-record" type="application\/json">([\s\S]*?)<\/script>/)[1]);assert.equal(saved.length,3);assert.equal(saved[0].scene.name,p[0].scene.name);assert.equal(saved[0].controls.material_A.xyz_m.length,301);
});
