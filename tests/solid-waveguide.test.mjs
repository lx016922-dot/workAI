import test from 'node:test';import assert from 'node:assert/strict';import {createHash} from 'node:crypto';
import {validateWaveguideReference,validateWaveguideDispersion} from '../server/solid-waveguide.mjs';
import {planPlatePropagation} from '../public/solid-propagation.js';
function fixture(){const scene={geometry:{kind:'plate',length_mm:60,width_mm:4,thickness_mm:1},source:{face:'xmin',radius_mm:0,direction:[1,0,0],frequency_mhz:.3,cycles:2,delay_us:0},boundary:{fixed_face:'none'},acquisition:{duration_us:15},material:{rho_kg_m3:2700,lambda_pa:58.1e9,mu_pa:26.1e9},receivers:[12,18,24].map(x=>({point_mm:[x,2,.5]}))};
  const controls=['linear_A','geometric_A','material_A'].map((role,i)=>({role,job_id:'solid-'+String(i).repeat(32),result_sha256:String(i).repeat(64)})),study={study_id:'study-'+'1'.repeat(32),scene,controls};
  const branches=[375,960].map((k,i)=>({frequency_hz:3e5*(i+1),wavenumber_rad_m:k,phase_velocity_m_s:2*Math.PI*3e5*(i+1)/k,group_velocity_m_s:4800/(i+1),relative_eigen_residual:1e-12,minimum_tracking_MAC:.99}));
  const r={schema:'workai.rectangular-waveguide.validation.v1',study_id:study.study_id,transient_references:controls,frequency_mhz:.3,cross_section_mm:{width:4,thickness:1},material:scene.material,branches,mismatch_rad_m:210,coherence_length_mm:Math.PI/210*1000,successive_relative_k_differences:[[.01,.01],[.001,.001]],dispersion_finest_pair_within_one_percent:true,modal_purity_verified:false,nonlinear_coupling_integral_computed:false,other_second_harmonic_modes_excluded:false,window_changed_for_this_fit:false,
    rows:scene.receivers.map(rx=>({point_mm:rx.point_mm,window_us:planPlatePropagation(scene,rx.point_mm).window_us,same_branch_group_walkoff_us:1,linear_fundamental_shape:{relative_mass_L2_shape_mismatch:.02,complex_coefficient:[1,2],not_a_modal_energy_fraction:true},material_second_shape:{relative_mass_L2_shape_mismatch:.4,complex_coefficient:[3,4],not_a_modal_energy_fraction:true}})),phase_checks:[{from_mm:12,to_mm:18,relative_wavenumber_difference:.001,spacing_below_half_wavelength:true},{from_mm:18,to_mm:24,relative_wavenumber_difference:.001,spacing_below_half_wavelength:true}],fundamental_candidate_checks_within_5_percent:true};return {r,study};}
test('finite-width evidence cannot be borrowed from another scene, child control or window',()=>{
  const {r,study}=fixture();validateWaveguideReference(r,study);
  for(const mutate of [r=>r.cross_section_mm.width=5,r=>r.transient_references[0].result_sha256='different',r=>r.rows[0].window_us[0]+=.1,r=>r.modal_purity_verified=true,r=>r.mismatch_rad_m=1]){const bad=structuredClone(r);mutate(bad);assert.throws(()=>validateWaveguideReference(bad,study));}
});
test('shape failures, aliasing and unconverged dispersion cannot keep a passed candidate label',()=>{
  for(const mutate of [r=>r.rows[0].linear_fundamental_shape.relative_mass_L2_shape_mismatch=.06,r=>{r.successive_relative_k_differences[1][1]=.02;r.dispersion_finest_pair_within_one_percent=false;},r=>r.phase_checks[0].relative_wavenumber_difference=.06]){
    const {r,study}=fixture();mutate(r);assert.throws(()=>validateWaveguideReference(r,study));r.fundamental_candidate_checks_within_5_percent=false;validateWaveguideReference(r,study);
  }
});
test('displayed dispersion must match its hashed independent eigensolver record',()=>{
  const {r}=fixture(),d={schema:'workai.rectangular-waveguide.dispersion.v1',levels:[{branches:r.branches}],successive_relative_k_differences:r.successive_relative_k_differences,mismatch_rad_m:r.mismatch_rad_m,coherence_length_mm:r.coherence_length_mm},bytes=Buffer.from(JSON.stringify(d));r.dispersion_sha256=createHash('sha256').update(bytes).digest('hex');validateWaveguideDispersion(r,d,bytes);
  r.branches=structuredClone(r.branches);r.branches[0].wavenumber_rad_m+=1;assert.throws(()=>validateWaveguideDispersion(r,d,bytes));
});
