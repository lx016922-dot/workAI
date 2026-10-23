import {s0Dispersion} from './plate-dispersion.js';
import {analyzeSolidHarmonics,harmonicCoefficient} from './solid-harmonics.js';

// The free infinite-width plate branch is a candidate for a finite-width solid,
// never a modal decomposition. The end-echo screen excludes only the far end.
export function planPlatePropagation(scene,point){
  const g=scene?.geometry,tx=scene?.source,m=scene?.material;
  const unavailable=reason=>({available:false,reason,modal_separation:false});
  if(g?.kind!=='plate'||!['xmin','xmax'].includes(tx?.face)||tx.radius_mm!==0||scene?.boundary?.fixed_face!=='none')return unavailable('直达波建议窗仅用于无孔直板、自由边界、完整 X 端面加载。');
  if(!Array.isArray(tx.direction)||Math.abs(Math.abs(tx.direction[0])-1)>1e-10||Math.abs(tx.direction[1])+Math.abs(tx.direction[2])>1e-10)return unavailable('当前 S₀ 候选要求沿 X 方向的端面牵引。');
  if(!Array.isArray(point)||point.length!==3||!point.every(Number.isFinite)||!['length_mm','width_mm','thickness_mm'].every((k,i)=>Number.isFinite(g[k])&&g[k]>0&&point[i]>0&&point[i]<g[k]))return unavailable('接收点必须位于直板内部。');
  if(!m||![m.rho_kg_m3,m.mu_pa,m.lambda_pa+2*m.mu_pa].every(v=>Number.isFinite(v)&&v>0)||![tx.frequency_mhz,tx.cycles].every(v=>Number.isFinite(v)&&v>0)||!Number.isFinite(tx.delay_us)||!Number.isFinite(scene.acquisition?.duration_us))return unavailable('材料、激励或记录参数不完整。');
  const cl=Math.sqrt((m.lambda_pa+2*m.mu_pa)/m.rho_kg_m3),ct=Math.sqrt(m.mu_pa/m.rho_kg_m3),f=tx.frequency_mhz;
  const fundamental=s0Dispersion(f*1e6,g.thickness_mm*.001,cl,ct),second=s0Dispersion(2*f*1e6,g.thickness_mm*.001,cl,ct);
  if(!fundamental||!second)return unavailable('基频或二次谐波超出当前 S₀ 分支求解范围（f·板厚 ≤ 1.2 MHz·mm）。');
  const distance=tx.face==='xmin'?point[0]:g.length_mm-point[0],arrival=tx.delay_us+distance*1000/fundamental.group_velocity_m_s;
  const window=[arrival-.5/f,arrival+(tx.cycles+.5)/f],frontSpeed=1.05*cl,echo=tx.delay_us+(2*g.length_mm-distance)*1000/frontSpeed;
  const mismatch=second.wavenumber_rad_m-2*fundamental.wavenumber_rad_m,halfPhase=mismatch*distance*.001/2;
  const covered=window[0]>=0&&window[1]<=scene.acquisition.duration_us,margin=echo-window[1];
  return {available:true,model:'isotropic_free_infinite_width_plate_S0_candidate',distance_mm:distance,window_us:window,arrival_us:arrival,
    record_covers_window:covered,end_echo_estimate_us:echo,end_echo_margin_us:margin,passes_end_echo_screen:covered&&margin>0,
    echo_speed_m_s:frontSpeed,echo_speed_assumption:'1.05 times small-strain longitudinal bulk speed; estimate, not a rigorous nonlinear wavefront bound',
    fundamental,second,mismatch_rad_m:mismatch,coherence_length_mm:Math.abs(mismatch)<1e-12?null:Math.PI/Math.abs(mismatch)*1000,
    group_walkoff_us:distance*1000*Math.abs(1/second.group_velocity_m_s-1/fundamental.group_velocity_m_s),
    phase_matching_factor_candidate:Math.abs(halfPhase)<1e-10?1:Math.abs(Math.sin(halfPhase)/halfPhase),
    modal_separation:false,scope:'Suggested equal-duration moving window. Finite-width modes, near fields and lateral reflections are not separated; the distal-end echo is screened only by a speed estimate.'};
}

export function comparePropagationProbes(probes){
  if(!Array.isArray(probes)||probes.length<3||probes.length>8)throw new Error('沿程比较需要 3–8 个同一研究的接收点。');
  const first=probes[0];
  const rows=probes.map(probe=>{
    if(probe.study_id!==first.study_id||JSON.stringify(probe.scene)!==JSON.stringify(first.scene)||JSON.stringify(probe.times_us)!==JSON.stringify(first.times_us)||probe.point_mm[1]!==first.point_mm[1]||probe.point_mm[2]!==first.point_mm[2]||probe.direction.some((v,i)=>Math.abs(v-[1,0,0][i])>1e-10))throw new Error('沿程比较须使用同一研究、同一 Y/Z 位置、X 接收方向和相同采样。');
    const plan=planPlatePropagation(probe.scene,probe.point_mm);
    if(!plan.available||!plan.passes_end_echo_screen)throw new Error(plan.reason??'建议窗未完整记录或可能包含远端回波；不能用于本次直达波比较。');
    const a=analyzeSolidHarmonics(probe,...plan.window_us),linear=harmonicCoefficient(probe.times_us,a.signals_m.linear,probe.scene.source.frequency_mhz,...plan.window_us);
    return {point_mm:probe.point_mm,distance_mm:plan.distance_mm,plan,linear_fundamental:linear,analysis:a};
  }).sort((a,b)=>a.distance_mm-b.distance_mm);
  if(rows.some((r,i)=>i>0&&r.distance_mm===rows[i-1].distance_mm))throw new Error('接收点传播距离不能重复。');
  const phase_checks=rows.slice(1).map((b,i)=>{
    const a=rows[i],dx=(b.distance_mm-a.distance_mm)*.001,k=b.plan.fundamental.wavenumber_rad_m,za=a.linear_fundamental,zb=b.linear_fundamental;
    if(za.amplitude_m<=0||zb.amplitude_m<=0)return {from_mm:a.distance_mm,to_mm:b.distance_mm,available:false};
    const expected=-k*dx,measured=Math.atan2(zb.im_m,zb.re_m)-Math.atan2(za.im_m,za.re_m),residual=Math.atan2(Math.sin(measured-expected),Math.cos(measured-expected));
    return {from_mm:a.distance_mm,to_mm:b.distance_mm,available:true,phase_residual_rad:residual,inferred_wavenumber_rad_m:k-residual/dx,relative_wavenumber_difference:-residual/(dx*k),
      phase_unwrap_reference:'nearest predicted S0 branch, not independent modal identification'};
  });
  // Only display a saved cross-section comparison for the same completed
  // controls and the exact positions/windows used in that comparison.
  let waveguide_reference=null;const reference=first.waveguide_reference;
  const equal=(a,b)=>Array.isArray(a)&&a.length===b.length&&a.every((v,i)=>Number.isFinite(v)&&Math.abs(v-b[i])<1e-10);
  if(reference?.schema==='workai.rectangular-waveguide.validation.v1'&&reference.study_id===first.study_id&&reference.frequency_mhz===first.scene.source.frequency_mhz&&reference.cross_section_mm?.width===first.scene.geometry.width_mm&&reference.cross_section_mm?.thickness===first.scene.geometry.thickness_mm&&['rho_kg_m3','lambda_pa','mu_pa'].every(k=>reference.material?.[k]===first.scene.material[k])&&reference.modal_purity_verified===false&&Array.isArray(reference.rows)&&Array.isArray(reference.transient_references)){
    const matches=probes.every(p=>JSON.stringify(p.waveguide_reference)===JSON.stringify(reference)&&reference.transient_references.length===3&&reference.transient_references.every(r=>p.controls[r.role]?.job_id===r.job_id&&p.controls[r.role]?.result_sha256===r.result_sha256))&&rows.every(row=>reference.rows.some(r=>equal(r.point_mm,row.point_mm)&&equal(r.window_us,row.plan.window_us)));
    if(matches)waveguide_reference=reference;
  }
  return {schema:'workai.solid.propagation-analysis.v1',study_id:first.study_id,rows,phase_checks,
    waveguide_reference,
    end_echo_screens_pass:rows.every(r=>r.plan.passes_end_echo_screen),
    candidate_phase_agreement_within_5_percent:phase_checks.every(p=>p.available&&Math.abs(p.relative_wavenumber_difference)<=.05),
    selected_numerical_checks_pass:rows.every(r=>r.analysis.selected_reference?.all_selected_checks_within_5_percent===true),
    pure_mode_accumulation_verified:false,surrogate_used:false,
    scope:'Pointwise material differences of the finite-width three-dimensional solid. Neither candidate phase agreement nor increasing amplitudes establishes a pure-mode accumulation law.'};
}
