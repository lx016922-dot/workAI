// Plane-strain result viewer: section animation, receiver waveforms/spectra, receiver-line harmonics and f–k.
import { harmonicCoefficients, lineTrace, lineHarmonics, accumulationFit, lineVelocities, fkSpectrum, amplitudeSpectrum } from './plane-data.js';

const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=v=>!Number.isFinite(v)?'—':Math.abs(v)>=.01&&Math.abs(v)<10000?v.toFixed(3):v.toExponential(3);

export function chart(x,y,labelX,labelY,color='#148778',width=900,overlays=[]) {
  const height=250,left=68,right=20,top=15,bottom=40;
  const xmin=Math.min(...x),xmax=Math.max(...x);let ymin=Math.min(0,...y),ymax=Math.max(0,...y);for(const o of overlays){ymin=Math.min(ymin,...o.y);ymax=Math.max(ymax,...o.y);}if(ymax-ymin<1e-30){ymin=-1;ymax=1;}
  const sx=v=>left+(v-xmin)/(xmax-xmin||1)*(width-left-right),sy=v=>height-bottom-(v-ymin)/(ymax-ymin)*(height-top-bottom);
  const line=(xs,ys,c,dash='')=>`<polyline points="${xs.map((v,i)=>`${sx(v).toFixed(2)},${sy(ys[i]).toFixed(2)}`).join(' ')}" fill="none" stroke="${c}" stroke-width="1.7" ${dash?`stroke-dasharray="${dash}"`:''}/>`;
  const f=v=>Math.abs(v)>=.01&&Math.abs(v)<10000?v.toFixed(2):v.toExponential(1);
  const ticks=Array.from({length:5},(_,i)=>{const xv=xmin+(xmax-xmin)*i/4,yv=ymin+(ymax-ymin)*i/4;return `<path d="M${left} ${sy(yv)}H${width-right}" stroke="#e6edef"/><text x="${left-8}" y="${sy(yv)+4}" text-anchor="end">${f(yv)}</text><text x="${sx(xv)}" y="${height-bottom+20}" text-anchor="middle">${f(xv)}</text>`;}).join('');
  return `<svg class="solid-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(labelY)}对${esc(labelX)}曲线"><g font-size="11" fill="#748995">${ticks}<text x="${width/2}" y="${height-3}" text-anchor="middle">${esc(labelX)}</text><text x="7" y="11">${esc(labelY)}</text></g>${line(x,y,color)}${overlays.map(o=>line(o.x,o.y,o.color||'#b67e4e',o.dash||'5 4')).join('')}</svg>`;
}

export function planeResultMarkup(r) {
  const d=r.diagnostics,hasField=!!r.field,lines=r.receiver_lines||[],points=r.receivers||[];
  return `<div class="notice">二维平面应变谱元解 · ${esc(r.scene.name)} · ${r.degree} 阶 · 每米宽度计。${esc(d.excitation_model==='multi_source_v2'?'多源/边界 v2 激励':'')}；结果精度仍需网格与阶次对照。</div>
  <div class="solid-layout"><section class="panel"><div class="panel-heading"><h2>截面位移场动画</h2><span class="pill">x–z 截面</span></div><div class="panel-body">
  ${hasField?`<canvas id="plane-canvas" class="plane-canvas" aria-label="二维截面位移场动画"></canvas>
  <div class="solid-animation"><button class="button secondary small" id="plane-play">播放</button><input id="plane-frame" aria-label="动画帧" type="range" min="0" max="${r.field.frame_indices.length-1}" value="0"><span id="plane-time"></span></div>
  <div class="form-grid"><label class="field"><span>着色分量</span><select id="plane-component"><option value="magnitude">位移模长</option><option value="0">u_x</option><option value="1">u_z</option></select></label><label class="field"><span>形变显示倍率</span><input id="plane-scale" type="number" min="0" max="1e8" step="100"></label><label class="field"><span>厚度方向放大</span><input id="plane-aspect" type="number" min="1" max="200" step="1" value="1"></label></div>
  <p id="plane-color-note" class="section-note"></p><p class="section-note">动画只保存了每 ${r.field.every_n_frames} 帧之一（${r.field.frame_indices.length} 帧），接收点与接收线为全采样。拖动可平移，滚轮缩放。</p>`:'<p class="section-note">该任务未保存位移场动画。</p>'}
  <dl class="definition"><dt>网格</dt><dd>${r.mesh.nodes_mm.length} 节点 / ${r.mesh.elements.length} 个 ${r.mesh.degree} 阶四边形</dd><dt>内部步长</dt><dd>${d.internal_step_us.toPrecision(4)} μs × ${d.internal_steps}</dd><dt>能量收支误差</dt><dd>${(100*d.external_work_balance_relative).toPrecision(4)}%</dd><dt>耗散能量</dt><dd>${fmt(d.dissipated_energy_j_per_m)} J/m</dd><dt>最大应变 / 最小 J</dt><dd>${d.max_strain.toExponential(3)} / ${d.min_J.toFixed(8)}</dd><dt>求解耗时</dt><dd>${Math.round(d.elapsed_s)} s</dd></dl>
  </div></section>
  <section class="panel"><div class="panel-heading"><h2>接收点波形与频谱</h2></div><div class="panel-body">
  <div class="form-grid"><label class="field"><span>接收点</span><select id="plane-receiver">${points.map((p,i)=>`<option value="p${i}">${esc(p.name)} (${p.point_mm[0]}, ${p.point_mm[2]}) mm</option>`).join('')}${lines.map((l,i)=>`<option value="l${i}">${esc(l.name)} 线 · 可选点</option>`).join('')}</select></label><label class="field"><span>线上点序号</span><input id="plane-line-point" type="number" min="0" step="1" value="0"></label>
  <label class="field"><span>方向 x</span><input id="plane-dir-0" type="number" step="any" value="1"></label><label class="field"><span>方向 z</span><input id="plane-dir-1" type="number" step="any" value="0"></label>
  <label class="field"><span>分析窗起点 / μs</span><input id="plane-win-start" type="number" step="any" value="0"></label><label class="field"><span>分析窗终点 / μs</span><input id="plane-win-end" type="number" step="any" value="${r.times_us.at(-1)}"></label></div>
  <p id="plane-probe-message" class="section-note" role="status"></p><div id="plane-waveform"></div><div id="plane-spectrum"></div><div id="plane-harmonics"></div>
  <p class="section-note">f 与 2f 为 Hann 加权、去均值的准确频率复系数，相位以记录起点为参考；与三维页面及 Python 分析模块算法相同。</p></div></section></div>
  ${lines.length?`<section class="panel" id="plane-line-panel"><div class="panel-heading"><h2>接收线：沿程谐波、速度与 f–k</h2><span class="muted small-text">复用当前任务的接收线数据</span></div><div class="panel-body">
  <div class="form-grid"><label class="field"><span>接收线</span><select id="plane-line">${lines.map((l,i)=>`<option value="${i}">${esc(l.name)} · ${l.count} 点 · ${l.distance_mm.at(-1).toFixed(1)} mm</option>`).join('')}</select></label><label class="field"><span>分析载频 / MHz</span><input id="plane-line-frequency" type="number" step="any" value="${r.scene.source.frequency_mhz}"></label>
  <label class="field"><span>时间窗</span><select id="plane-line-mode"><option value="moving">随 x / c_g 移动</option><option value="fixed">固定窗</option></select></label><label class="field"><span>群速度 / m·s⁻¹（移动窗）</span><input id="plane-line-cg" type="number" step="any" value=""></label>
  <label class="field"><span>脉冲时长 / μs（移动窗）</span><input id="plane-line-burst" type="number" step="any" value="${(r.scene.source.cycles/r.scene.source.frequency_mhz).toFixed(3)}"></label><label class="field"><span>激励时延 / μs</span><input id="plane-line-delay" type="number" step="any" value="${r.scene.source.delay_us}"></label>
  <label class="field"><span>固定窗起点 / μs</span><input id="plane-line-start" type="number" step="any" value="0"></label><label class="field"><span>固定窗终点 / μs</span><input id="plane-line-end" type="number" step="any" value="${r.times_us.at(-1)}"></label></div>
  <div class="solid-path-actions"><button class="button primary small" id="plane-line-analyze">提取沿程 f / 2f 与速度</button><button class="button secondary small" id="plane-line-fk">计算 f–k 谱</button><button class="button secondary small" id="plane-line-csv" disabled>下载沿程 CSV</button></div>
  <p id="plane-line-message" class="section-note" role="status">群速度留空时使用频散曲线中与载频对应的 S0/A0 群速度（若已加载）。</p>
  <div id="plane-line-charts" class="solid-path-charts"></div><p id="plane-line-fit" class="section-note"></p><div id="plane-line-table" class="table-scroll"></div>
  <canvas id="plane-fk" class="fk-canvas" hidden aria-label="f–k 幅值谱"></canvas><p id="plane-fk-note" class="section-note"></p>
  <p class="section-note">A₂/A₁² 的斜率与窗口加权有关，仅用于同一设置下的相对比较；相速度由展开相位斜率得到。f–k 谱叠加的虚线为该板厚的 Rayleigh–Lamb 理论曲线（向 +x 传播在正波数）。</p></div></section>`:''}`;
}

export function mountPlaneResult(container,result,field,lines,dispersion) {
  const $=s=>container.querySelector(s),r=result,nodes=r.mesh.nodes_mm,n=nodes.length,times=r.times_us;
  let disposed=false,timer,frame=0,playing=false,last=0,drag=null,pan=[0,0],zoom=1,cleanupFns=[];
  const listen=(sel,ev,fn)=>{const el=$(sel);if(!el)return;el.addEventListener(ev,fn);cleanupFns.push(()=>el.removeEventListener(ev,fn));};
  // ---------- section animation ----------
  if(field&&$('#plane-canvas')){
    const canvas=$('#plane-canvas'),ctx=canvas.getContext('2d'),indices=r.field.frame_indices,degree=r.mesh.degree,size=degree+1;
    const xs=nodes.map(p=>p[0]),zs=nodes.map(p=>p[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),zmin=Math.min(...zs),zmax=Math.max(...zs);
    let peak=0;for(let i=0;i<field.length;i+=2)peak=Math.max(peak,Math.hypot(field[i],field[i+1]));
    $('#plane-scale').value=Math.min(1e8,Math.round((zmax-zmin)*.3/(peak*1000||1)));
    $('#plane-aspect').value=Math.max(1,Math.min(200,Math.round((xmax-xmin)/(zmax-zmin)/4)));
    frame=Math.floor(indices.length*.5);$('#plane-frame').value=frame;
    const subquads=[];for(const ids of r.mesh.elements)for(let i=0;i<degree;i++)for(let j=0;j<degree;j++)subquads.push([ids[i*size+j],ids[(i+1)*size+j],ids[(i+1)*size+j+1],ids[i*size+j+1]]);
    function draw(){
      if(disposed)return;const box=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2),w=Math.max(240,box.width),h=Math.max(200,box.height);canvas.width=w*ratio;canvas.height=h*ratio;ctx.scale(ratio,ratio);
      ctx.fillStyle='#f3f7f8';ctx.fillRect(0,0,w,h);
      const aspect=Math.max(1,Number($('#plane-aspect').value)||1),multiplier=Math.max(0,Number($('#plane-scale').value)||0),component=$('#plane-component').value,offset=frame*n*2;
      const spanX=xmax-xmin||1,spanZ=(zmax-zmin||1)*aspect,scale=Math.min((w-40)/spanX,(h-50)/spanZ)*zoom;
      const px=x=>w/2+pan[0]+(x-(xmin+xmax)/2)*scale,pz=z=>h/2-10+pan[1]-(z-(zmin+zmax)/2)*aspect*scale;
      const pos=new Float64Array(n*2),val=new Float64Array(n);
      for(let i=0;i<n;i++){const ux=field[offset+i*2],uz=field[offset+i*2+1];pos[i*2]=px(nodes[i][0]+ux*1000*multiplier);pos[i*2+1]=pz(nodes[i][1]+uz*1000*multiplier);val[i]=component==='magnitude'?Math.hypot(ux,uz):component==='0'?ux:uz;}
      for(const q of subquads){const v=(val[q[0]]+val[q[1]]+val[q[2]]+val[q[3]])/4,s=Math.max(-1,Math.min(1,v/(peak||1))),m=component==='magnitude'?s*2-1:s;
        const rgb=m>=0?[245,Math.round(238-155*m),Math.round(225-170*m)]:[Math.round(235+160*m),Math.round(240+70*m),246];
        ctx.beginPath();q.forEach((id,k)=>{if(k===0)ctx.moveTo(pos[id*2],pos[id*2+1]);else ctx.lineTo(pos[id*2],pos[id*2+1]);});ctx.closePath();ctx.fillStyle=`rgb(${rgb.join(',')})`;ctx.fill();}
      ctx.strokeStyle='#41647e';ctx.lineWidth=.9;ctx.beginPath();for(const [a,b] of r.mesh.boundary_lines){ctx.moveTo(pos[a*2],pos[a*2+1]);ctx.lineTo(pos[b*2],pos[b*2+1]);}ctx.stroke();
      ctx.fillStyle='#537482';ctx.font='12px sans-serif';ctx.fillText(`x / z · mm   t = ${times[indices[frame]].toFixed(2)} μs   厚度放大 ${aspect}×`,15,20);
      // scale bar
      ctx.fillStyle='#537482';const bar=Math.pow(10,Math.floor(Math.log10(spanX/4)));ctx.fillRect(15,h-14,bar*scale,2);ctx.fillText(`${bar} mm`,20+bar*scale,h-10);
      $('#plane-time').textContent=times[indices[frame]].toFixed(2)+' μs';$('#plane-color-note').textContent=`固定色标：${component==='magnitude'?'0':'−'+(peak*1e9).toFixed(3)} 至 ${(peak*1e9).toFixed(3)} nm；形变显示 ${multiplier} 倍，厚度方向放大 ${aspect} 倍。`;
    }
    function animate(time){if(disposed)return;if(playing&&time-last>66){frame=(frame+1)%indices.length;$('#plane-frame').value=frame;draw();last=time;}timer=requestAnimationFrame(animate);}
    listen('#plane-play','click',()=>{playing=!playing;$('#plane-play').textContent=playing?'暂停':'播放';});
    listen('#plane-frame','input',()=>{frame=Number($('#plane-frame').value);draw();});
    for(const id of ['#plane-component','#plane-scale','#plane-aspect'])listen(id,'change',draw);
    listen('#plane-canvas','pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});listen('#plane-canvas','pointerup',()=>drag=null);
    listen('#plane-canvas','pointermove',e=>{if(drag){pan[0]+=e.clientX-drag[0];pan[1]+=e.clientY-drag[1];drag=[e.clientX,e.clientY];draw();}});
    listen('#plane-canvas','wheel',e=>{e.preventDefault();zoom=Math.max(.3,Math.min(20,zoom*Math.exp(-e.deltaY*.001)));draw();});
    const resize=new ResizeObserver(draw);resize.observe(canvas);cleanupFns.push(()=>resize.disconnect());
    timer=requestAnimationFrame(animate);cleanupFns.push(()=>cancelAnimationFrame(timer));
  }
  // ---------- receiver waveform & spectrum ----------
  function probeSeries(){
    const choice=$('#plane-receiver').value,d=[Number($('#plane-dir-0').value),Number($('#plane-dir-1').value)],norm=Math.hypot(...d);if(!(norm>0))throw new Error('方向不能为零向量。');
    const unit=d.map(v=>v/norm);
    if(choice.startsWith('p')){const p=r.receivers[Number(choice.slice(1))];return {label:p.name,values:p.u_x_m.map((v,i)=>v*unit[0]+p.u_z_m[i]*unit[1])};}
    const line=r.receiver_lines[Number(choice.slice(1))],index=Math.max(0,Math.min(line.count-1,Number($('#plane-line-point').value)||0));
    const total=r.lines.shape[1],column=line.columns[index],values=new Array(times.length);
    for(let t=0;t<times.length;t++){const base=(t*total+column)*2;values[t]=lines[base]*unit[0]+lines[base+1]*unit[1];}
    return {label:`${line.name}[${index}] @ ${line.distance_mm[index].toFixed(2)} mm`,values};
  }
  function extract(){
    try{
      const {label,values}=probeSeries(),f=r.scene.source.frequency_mhz,analysis=harmonicCoefficients(times,values,f,Number($('#plane-win-start').value),Number($('#plane-win-end').value));
      const step=r.scene.acquisition.sample_step_us,spectrum=amplitudeSpectrum(analysis.sample_count>=4?values.slice():values,step);
      $('#plane-waveform').innerHTML=chart(times,values.map(v=>v*1e9),'时间 / μs','位移 / nm');
      const stop=Math.min(spectrum.nyquist_mhz,Math.max(f*5,1)),idx=spectrum.frequency_mhz.map((v,i)=>v<=stop?i:-1).filter(i=>i>=0);
      $('#plane-spectrum').innerHTML=chart(idx.map(i=>spectrum.frequency_mhz[i]),idx.map(i=>spectrum.amplitude_m[i]*1e9),'频率 / MHz','幅值 / nm（整段记录）','#5b80b6');
      $('#plane-harmonics').innerHTML=`<div class="table-scroll"><table><thead><tr><th>分量</th><th>频率 / MHz</th><th>幅值 / nm</th><th>相位 / °</th></tr></thead><tbody>${analysis.harmonics.map(h=>`<tr><td>${h.order===1?'f':'2f'}</td><td>${h.frequency_mhz.toPrecision(5)}</td><td>${h.available?(h.amplitude_m*1e9).toExponential(5):'超过 Nyquist'}</td><td>${h.available&&h.phase_deg!==null?h.phase_deg.toFixed(3):'—'}</td></tr>`).join('')}</tbody></table></div>`;
      $('#plane-probe-message').textContent=`${label} · 分析窗 ${analysis.start_us.toFixed(3)}–${analysis.end_us.toFixed(3)} μs（${analysis.sample_count} 点，${analysis.carrier_cycles.toFixed(1)} 个载频周期）。`;$('#plane-probe-message').classList.remove('model-error');
    }catch(error){$('#plane-probe-message').textContent=error.message;$('#plane-probe-message').classList.add('model-error');}
  }
  for(const id of ['#plane-receiver','#plane-line-point','#plane-dir-0','#plane-dir-1','#plane-win-start','#plane-win-end'])listen(id,'change',extract);
  if(r.receivers.length||r.receiver_lines.length)extract();
  // ---------- receiver line analysis ----------
  let rows=null;
  const theory=()=>{if(!dispersion)return null;const thickness=r.scene.geometry.thickness_mm;return {thickness,modes:dispersion.modes};};
  function modeAt(name,frequency){const t=theory();if(!t)return null;const m=t.modes.find(x=>x.name===name);if(!m)return null;const fd=frequency*t.thickness,grid=m.fd_mhz_mm;if(fd<grid[0]||fd>grid.at(-1))return null;let i=0;while(i<grid.length-2&&grid[i+1]<fd)i++;const w=(fd-grid[i])/(grid[i+1]-grid[i]);return {cp:m.phase_velocity_m_s[i]*(1-w)+m.phase_velocity_m_s[i+1]*w,cg:m.group_velocity_m_s[i]*(1-w)+m.group_velocity_m_s[i+1]*w};}
  if($('#plane-line')){
    const hint=modeAt('S0',r.scene.source.frequency_mhz);if(hint&&!$('#plane-line-cg').value)$('#plane-line-cg').value=hint.cg.toFixed(1);
    listen('#plane-line-analyze','click',()=>{
      try{
        const line=r.receiver_lines[Number($('#plane-line').value)],f=Number($('#plane-line-frequency').value),mode=$('#plane-line-mode').value;
        let cg=Number($('#plane-line-cg').value);if(!(cg>0)){const guess=modeAt('S0',f);cg=guess?guess.cg:NaN;}
        rows=lineHarmonics(r,lines,line,f,{mode,group_velocity:cg,burst_us:Number($('#plane-line-burst').value),delay_us:Number($('#plane-line-delay').value),start_us:Number($('#plane-line-start').value),end_us:Number($('#plane-line-end').value)});
        const fit=accumulationFit(rows),v=lineVelocities(rows,f),x=rows.map(q=>q.distance_mm);
        $('#plane-line-charts').innerHTML=`<div><h3>基频幅值 A₁</h3>${chart(x,rows.map(q=>q.harmonics[0].amplitude_m*1e9),'沿程 / mm','nm')}</div><div><h3>二次谐波幅值 A₂</h3>${fit.available?chart(x,fit.a2.map(v=>v*1e9),'沿程 / mm','nm','#ad6c36'):'<p>2f 超过 Nyquist。</p>'}</div><div><h3>A₂ / A₁²</h3>${fit.available?chart(x,fit.ratio,'沿程 / mm','1/m','#5b80b6',900,[{x,y:x.map(q=>fit.fit_ratio.slope*q*1e-3+fit.fit_ratio.intercept)}]):''}</div>`;
        const s0=modeAt('S0',f),a0=modeAt('A0',f);
        $('#plane-line-fit').textContent=`相速度 ${fmt(v.phase_velocity_m_s)} m/s（相位拟合 R² ${fmt(v.phase_fit_r_squared)}）${s0?`；理论 S0 ${s0.cp.toFixed(1)} m/s，A0 ${a0?a0.cp.toFixed(1):'—'} m/s`:''}。${fit.available?`A₂/A₁² 线性拟合斜率 ${fit.fit_ratio.slope.toExponential(3)} m⁻¹·m⁻¹，截距 ${fit.fit_ratio.intercept.toExponential(3)}，R² ${fmt(fit.fit_ratio.r_squared)}；A₂ 对 x 的 R² ${fmt(fit.fit_a2.r_squared)}。`:''}${mode==='moving'?`移动窗群速度 ${cg.toFixed(1)} m/s。`:''}`;
        $('#plane-line-table').innerHTML=`<table><thead><tr><th>点</th><th>x / mm</th><th>窗 / μs</th><th>A₁ / nm</th><th>φ₁ / °</th><th>A₂ / nm</th><th>φ₂ / °</th></tr></thead><tbody>${rows.filter((_,i)=>i%Math.max(1,Math.ceil(rows.length/24))===0||i===rows.length-1).map(q=>`<tr><td>${q.index}</td><td>${q.distance_mm.toFixed(3)}</td><td>${q.window_us.map(w=>w.toFixed(2)).join('–')}</td><td>${(q.harmonics[0].amplitude_m*1e9).toExponential(4)}</td><td>${q.harmonics[0].phase_deg?.toFixed(2)??'—'}</td><td>${q.harmonics[1].available?(q.harmonics[1].amplitude_m*1e9).toExponential(4):'—'}</td><td>${q.harmonics[1].available&&q.harmonics[1].phase_deg!==null?q.harmonics[1].phase_deg.toFixed(2):'—'}</td></tr>`).join('')}</tbody></table>`;
        $('#plane-line-csv').disabled=false;$('#plane-line-message').textContent=`已提取 ${rows.length} 点，未重新求解。`;$('#plane-line-message').classList.remove('model-error');
      }catch(error){rows=null;$('#plane-line-csv').disabled=true;$('#plane-line-message').textContent=error.message;$('#plane-line-message').classList.add('model-error');}
    });
    listen('#plane-line-csv','click',()=>{if(!rows)return;const header=['point_index','distance_mm','window_start_us','window_end_us','a1_m','phase1_deg','a2_m','phase2_deg'];
      const text=[`# job_id,${r.job_id}`,`# frequency_mhz,${$('#plane-line-frequency').value}`,`# window_mode,${$('#plane-line-mode').value}`,header.join(',')].concat(rows.map(q=>[q.index,q.distance_mm,...q.window_us,q.harmonics[0].amplitude_m,q.harmonics[0].phase_deg??'',q.harmonics[1].available?q.harmonics[1].amplitude_m:'',q.harmonics[1].available&&q.harmonics[1].phase_deg!==null?q.harmonics[1].phase_deg:''].join(','))).join('\n');
      const a=document.createElement('a'),url=URL.createObjectURL(new Blob([text],{type:'text/csv;charset=utf-8'}));a.href=url;a.download=`${r.job_id}-line-harmonics.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
    listen('#plane-line-fk','click',()=>{
      try{
        const line=r.receiver_lines[Number($('#plane-line').value)],fk=fkSpectrum(r,lines,line,2),canvas=$('#plane-fk');canvas.hidden=false;
        const ctx=canvas.getContext('2d'),ratio=Math.min(devicePixelRatio||1,2),w=Math.max(320,canvas.getBoundingClientRect().width),h=300;canvas.width=w*ratio;canvas.height=h*ratio;ctx.scale(ratio,ratio);
        const fmax=Math.min(Math.max(...fk.frequency_mhz),r.scene.source.frequency_mhz*3),kmax=Math.max(...fk.wavenumber_rad_per_mm.map(Math.abs));
        const left=55,bottom=36,pw=w-left-15,ph=h-bottom-15;let peak=0;for(let j=0;j<fk.rows;j++)for(let f=0;f<fk.columns;f++)if(fk.frequency_mhz[f]<=fmax)peak=Math.max(peak,fk.magnitude(f,j));
        ctx.fillStyle='#f3f7f8';ctx.fillRect(0,0,w,h);const fcount=fk.frequency_mhz.filter(v=>v<=fmax).length;
        for(let j=0;j<fk.rows;j++){const k=fk.wavenumber_rad_per_mm[j];const x=left+(k+kmax)/(2*kmax)*pw,cw=Math.max(1,pw/fk.rows+.5);for(let f=0;f<fcount;f++){const m=fk.magnitude(f,j)/(peak||1),y=15+ph-(fk.frequency_mhz[f]/fmax)*ph,ch=Math.max(1,ph/fcount+.5);const g=Math.sqrt(m);ctx.fillStyle=`rgb(${Math.round(245-200*g)},${Math.round(238-140*g)},${Math.round(225-60*g)})`;ctx.fillRect(x,y-ch,cw,ch);}}
        const t=theory();if(t)for(const m of t.modes){ctx.strokeStyle=m.family==='S'?'#148778':'#b67e4e';ctx.setLineDash([5,4]);ctx.lineWidth=1.2;ctx.beginPath();let started=false;m.frequency_mhz.forEach((f,i)=>{const k=m.wavenumber_rad_per_mm[i];if(f>fmax||k>kmax)return;const x=left+(k+kmax)/(2*kmax)*pw,y=15+ph-(f/fmax)*ph;if(!started){ctx.moveTo(x,y);started=true;}else ctx.lineTo(x,y);});ctx.stroke();ctx.setLineDash([]);}
        ctx.fillStyle='#537482';ctx.font='11px sans-serif';ctx.fillText('k / rad·mm⁻¹（+ 向 +x 传播）',left+pw/2-70,h-8);ctx.save();ctx.translate(12,15+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText('f / MHz',-20,0);ctx.restore();
        [-kmax,0,kmax].forEach(k=>ctx.fillText(k.toFixed(2),left+(k+kmax)/(2*kmax)*pw-10,h-22));[0,fmax/2,fmax].forEach(f=>ctx.fillText(f.toFixed(2),left-40,15+ph-(f/fmax)*ph+4));
        $('#plane-fk-note').textContent=`f–k 幅值谱（Hann 二维窗，2 倍补零），显示 0–${fmax.toFixed(2)} MHz，|k| ≤ ${kmax.toFixed(2)} rad/mm。${t?'虚线：Rayleigh–Lamb 理论（绿 S 族，棕 A 族）。':'未加载理论频散曲线。'}`;
      }catch(error){$('#plane-fk-note').textContent=error.message;}
    });
  }
  const cleanup=()=>{disposed=true;cleanupFns.forEach(fn=>fn());};return cleanup;
}
