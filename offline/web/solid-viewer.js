import { pointSeries, analyzeTimeWindow } from './solid-data.js';

const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const responseLabels={full:'完整位移',linear:'线性位移',stvk:'StVK 位移',geometric:'几何非线性增量',material:'材料非线性增量',nonlinear:'非线性增量（完整 − 线性）'};
/** What a result's computed controls give (offline/domain.py parts_for): how the shown response relates to them. */
export function controlsNote(r){
  const laws=(r.controls||[]).map(c=>c.law);
  if(laws.length===1)return '这次只计算了完整（Murnaghan）控制，没有线性和 StVK 对照；要分离非线性增量，请在“求解器配置”中选择“完整 + 线性”或“全部”。';
  if(laws.length===2)return '完整与线性两个冻结网络分支使用相同网格与激励；非线性增量 = 完整 − 线性（几何与材料增量之和）。要把它分成几何增量和材料增量，请在“求解器配置”中选择“全部”。';
  return '三个冻结网络分支使用相同网格与激励；材料增量 = 完整材料 − StVK，几何增量 = StVK − 线性。';
}
/** Defects that act as one-sided contact: closed cracks and delaminations, contact interfaces (physics/defects.py is_contact). */
export const contactDefects=scene=>(scene?.defects||[]).filter(d=>d.type==='contact'||(['crack','delamination'].includes(d.type)&&d.closed===true));
/** Where the harmonics of contact are: the contact acts in every control, so they are in each control's response and cancel in the differences. */
export function contactNote(r){
  if(!contactDefects(r.scene).length)return '';
  return ['geometric','material','nonlinear'].includes(r.response_part)
    ?'这个模型有接触（闭合裂纹、闭合分层或接触界面）。接触在每一组控制里都起作用，它产生的谐波在“完整位移”和“线性位移”里；现在显示的增量是两组控制之差，只含材料和几何非线性，<b>不含接触谐波</b>。要看接触非线性，请把“响应”改为完整位移。'
    :'这个模型有接触（闭合裂纹、闭合分层或接触界面）。接触在每一组控制里都起作用，它产生的谐波就在现在显示的响应里；“非线性增量”等几种增量是两组控制之差，不含接触谐波。';
}
/** How the displacement field is first shown: a strip (hardly any width: a plane-strain model) from its side with the thickness drawn
 *  larger, a pipe and a plate at an angle. `stretch` multiplies the drawn z extent, not the displacement. */
export function initialView(kind,min,max){
  if(kind==='pipe')return {yaw:1.1,pitch:.45,stretch:1,strip:false};
  const size=max.map((v,j)=>v-min[j]);
  if(size[1]<=.05*size[0]&&size[2]>0)return {yaw:0,pitch:0,stretch:Math.max(1,Math.round(size[0]/(10*size[2]))),strip:true};
  return {yaw:.55,pitch:1.05,stretch:1,strip:false};
}
/** The displacement (m) at the end of the colour scale: the largest of the record, of the frame shown, or the value given in nm. */
export function colorPeak(mode,customNm,recordPeak,framePeak){
  if(mode==='custom'&&Number.isFinite(customNm)&&customNm>0)return customNm*1e-9;
  if(mode==='frame'&&framePeak>0)return framePeak;
  return recordPeak;
}
export const COLOR_MODES={record:'整段记录的最大值（固定）',frame:'当前时刻的最大值',custom:'自定义上限'};
export function solidResultMarkup(r) {
  const d=r.diagnostics;
  const spectral=true, difference=['geometric','material','nonlinear'].includes(r.response_part), compact=r.recording?.record_mode==='compact', shown=r.display;
  return `<div class="notice">图网络离线预测 · ${esc(r.scene.name)} · ${esc(responseLabels[r.response_part]||'完整位移')}。${controlsNote(r)}新增工况与任意点位的精度需独立核验。</div>${contactNote(r)?`<div class="notice warn" id="solid-contact-note">${contactNote(r)}</div>`:''}
  <div class="solid-layout"><section class="panel"><div class="panel-heading"><h2>三维位移场动画</h2><span class="pill">本机计算</span></div><div class="panel-body">
  <canvas id="solid-canvas" class="solid-canvas" aria-label="三维铝结构位移场动画"></canvas>
  <div class="solid-views" role="group" aria-label="视角"><button type="button" class="button secondary small" data-solid-view="iso">等轴</button><button type="button" class="button secondary small" data-solid-view="side">侧视 XZ</button><button type="button" class="button secondary small" data-solid-view="top">俯视 XY</button><label class="solid-check" title="把板厚方向画得更大，位移不变；长而薄的模型默认打开"><input type="checkbox" id="solid-thin"> 放大薄方向</label></div>
  <div class="solid-animation"><button class="button secondary small" id="solid-play">播放</button><input id="solid-frame" aria-label="动画时间帧" type="range" min="0" max="${r.times_us.length-1}" value="0"><span id="solid-time"></span></div>
  <div class="form-grid"><label class="field"><span>着色分量</span><select id="solid-component"><option value="magnitude">位移模长</option><option value="0">X 位移</option><option value="1">Y 位移</option><option value="2">Z 位移</option><option value="direction">接收方向投影</option></select></label><label class="field"><span>形变显示倍率（可设 0）</span><input id="solid-scale" type="number" min="0" max="10000000" step="100"></label><label class="field"><span>色标范围</span><select id="solid-color-mode">${Object.entries(COLOR_MODES).map(([k,v])=>`<option value="${k}">${v}</option>`).join('')}</select></label><label class="field"><span>色标上限 / nm（选“自定义上限”时）</span><input id="solid-color-max" type="number" min="0" step="any" disabled></label></div>
  <p id="solid-color-note" class="section-note"></p><p class="section-note">${r.recording?`${compact?'精简记录':'完整记录'} ${(r.recording.duration_us/1000).toFixed(3)} ms / ${r.recording.frame_count} 个采样点；动画抽帧显示，分析和 CSV 使用${compact?'接收点的完整采样波形':'完整记录'}。`:"历史短记录；要分析 5 ms，请使用新配置重新计算。"}拖动旋转，Shift+拖动平移，滚轮缩放。颜色为真实位移；形变按显示倍率放大。橙色边线表示加载区域。${shown?`动画用显示网格：${shown.triangles.toLocaleString()} 个表面三角形 / ${shown.nodes.toLocaleString()} 个节点（计算网格 ${shown.computation_nodes.toLocaleString()} 个节点${shown.cell_mm?`，表面节点按 ${shown.cell_mm.toFixed(2)} mm 合并`:shown.stride>1?`，每个单元面每 ${shown.stride} 个节点取一个`:'，表面节点全部显示'}），float32；显示网格不影响计算和接收点波形。`:'显示全部表面三角形，包括孔和管的内表面。'}</p>
  </div></section><section class="panel"><div class="panel-heading"><h2>任意点位与方向</h2></div><div class="panel-body">
  <label class="field"><span>预设接收点</span><select id="solid-receiver">${r.scene.receivers.map((p,i)=>`<option value="${i}">${esc(p.name)}</option>`).join('')}</select></label>
  <div class="form-grid solid-vectors">${['x','y','z'].map((a,i)=>`<label class="field"><span>${a.toUpperCase()} / mm</span><input id="solid-point-${i}" type="number" step="any"></label>`).join('')}${['x','y','z'].map((a,i)=>`<label class="field"><span>方向 ${a}</span><input id="solid-direction-${i}" type="number" step="any"></label>`).join('')}</div>
  <button class="button primary" id="solid-extract">提取波形与频谱</button><p id="solid-probe-message" class="section-note" role="status"></p>
  <div class="form-grid"><label class="field"><span>分析窗起点 / μs</span><input id="solid-window-start" type="number" step="any" min="0" max="${r.times_us.at(-1)}" value="0"></label><label class="field"><span>分析窗终点 / μs</span><input id="solid-window-end" type="number" step="any" min="0" max="${r.times_us.at(-1)}" value="${r.times_us.at(-1)}"></label></div>
  <label class="field"><span>频谱显示上限 / MHz</span><input id="solid-frequency-limit" type="number" min="0.001" max="${.5/r.scene.acquisition.sample_step_us}" step="any" value="${Math.min(.5/r.scene.acquisition.sample_step_us,Math.max(r.scene.source.frequency_mhz*5,1))}"></label>
  <p class="section-note">${compact?`精简记录只在接收点保存了全部采样点（float64）：请从“预设接收点”中选择。其他位置需要先设为接收点再计算，或在“求解器配置”中选择完整记录。另有 ${r.recording.snapshot_frames.length} 个时刻的完整 float64 位移场快照随 ZIP 导出。`:`使用完整时域节点场进行${spectral?'高阶谱单元':'四面体'}插值，可在求解后增加实体内的点位。`}方向自动归一化；输出为有正负号的位移投影，不是位移模长。</p>
  <button class="button secondary small" id="solid-csv">下载当前点 CSV</button>
  ${difference?'<p class="section-note">以下应变、J 和能量检查来自完整 Murnaghan 控制。位移差不是单独的物理状态，不为增量场定义独立能量。</p>':''}
  <dl class="definition"><dt>网格</dt><dd>${(r.spectral?.node_count??r.nodes_mm.length).toLocaleString()} 节点 / ${spectral?(r.spectral.element_count??r.spectral.elements.length).toLocaleString()+' 高阶六面体':r.tetra.length+' 四面体'}</dd><dt>${shown?'动画存储精度':'场存储精度'}</dt><dd>${esc(d.output_dtype)}${shown?'（接收点 float64）':''}</dd><dt>内部步长</dt><dd>${d.internal_step_us.toPrecision(4)} μs</dd><dt>能量收支误差</dt><dd>${Number.isFinite(d.external_work_balance_relative)?(100*d.external_work_balance_relative).toPrecision(4)+'%':'尚未独立核验'}</dd><dt>控制场最大应变</dt><dd>${d.max_strain.toExponential(3)}</dd><dt>控制场最小 J</dt><dd>${d.min_J.toFixed(8)}</dd></dl>
  </div></section></div><section class="panel"><div class="panel-heading"><h2>方向位移波形</h2><span class="muted small-text">nm · 采样间隔 ${r.scene.acquisition.sample_step_us} μs</span></div><div class="panel-body"><div id="solid-waveform"></div></div></section>
  <section class="panel"><div class="panel-heading"><h2>单边幅值频谱</h2><span class="muted small-text">Hann 窗 · 去加权均值</span></div><div class="panel-body"><div id="solid-spectrum"></div><p id="solid-spectrum-note" class="section-note"></p><div id="solid-harmonics"></div><p class="section-note">${r.scene.signal?.kind==='chirp'?'扫频按激励频带及二倍频带积分功率，报告 Hann 加权 RMS；不定义固定 f / 2f 相位。':'f 与 2f 在准确频率处计算复数系数，使用上方同一时间窗；相位以记录起点为参考。'}短脉冲泄漏、反射和时间窗均会影响幅值，这些数值不能直接视为材料非线性系数。</p></div></section>`;
}

export function chart(x,y,labelX,labelY,color='#148778',width=900) {
  const height=250,left=65,right=20,top=15,bottom=40;
  let xmin=Infinity,xmax=-Infinity,ymin=0,ymax=0;for(let i=0;i<x.length;i++){xmin=Math.min(xmin,x[i]);xmax=Math.max(xmax,x[i]);ymin=Math.min(ymin,y[i]);ymax=Math.max(ymax,y[i]);}if(ymax-ymin<1e-20){ymin=-1;ymax=1;}
  const sx=v=>left+(v-xmin)/(xmax-xmin||1)*(width-left-right),sy=v=>height-bottom-(v-ymin)/(ymax-ymin)*(height-top-bottom);
  const indices=[];const stride=Math.max(1,Math.ceil(x.length/1200));for(let a=0;a<x.length;a+=stride){let lo=a,hi=a;for(let i=a+1;i<Math.min(x.length,a+stride);i++){if(y[i]<y[lo])lo=i;if(y[i]>y[hi])hi=i;}indices.push(...[a,lo,hi,Math.min(x.length-1,a+stride-1)].sort((a,b)=>a-b).filter((v,i,a)=>!i||v!==a[i-1]));}const path=indices.map(i=>`${sx(x[i]).toFixed(2)},${sy(y[i]).toFixed(2)}`).join(' ');
  const fmt=v=>Math.abs(v)>=.01&&Math.abs(v)<10000?v.toFixed(2):v.toExponential(1);
  const ticks=Array.from({length:5},(_,i)=>{const xv=xmin+(xmax-xmin)*i/4,yv=ymin+(ymax-ymin)*i/4;return `<path d="M${left} ${sy(yv)}H${width-right}" stroke="#e6edef"/><text x="${left-8}" y="${sy(yv)+4}" text-anchor="end">${fmt(yv)}</text><text x="${sx(xv)}" y="${height-bottom+20}" text-anchor="middle">${fmt(xv)}</text>`;}).join('');
  return `<svg class="solid-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(labelY)}对${esc(labelX)}曲线"><g font-size="11" fill="#748995">${ticks}<text x="${width/2}" y="${height-3}" text-anchor="middle">${esc(labelX)}</text><text x="7" y="11">${esc(labelY)}</text></g><polyline points="${path}" fill="none" stroke="${color}" stroke-width="1.7"/></svg>`;
}

export function mountSolidResult(container,result,field,extractSeries=pointSeries,viewState=null) {
  const $=s=>container.querySelector(s),canvas=$('#solid-canvas'),ctx=canvas.getContext('2d'),nodes=result.nodes_mm,n=nodes.length;
  const loadedFaces=new Set(result.loaded_faces||[]);
  let frame=viewState?.frame??Math.floor(result.times_us.length*.45),playing=false,timer,last=0,yaw=0,pitch=0,zoom=1,panX=0,panY=0,stretch=1,drag=null,probe=null,analysis=null,direction=[1,0,0],disposed=false,extraction=0;
  const edges=new Map(),edgeKey=(a,b)=>a<b?`${a},${b}`:`${b},${a}`;
  for(const face of result.surface){const a=nodes[face[0]],b=nodes[face[1]].map((v,j)=>v-a[j]),c=nodes[face[2]].map((v,j)=>v-a[j]),normal=[b[1]*c[2]-b[2]*c[1],b[2]*c[0]-b[0]*c[2],b[0]*c[1]-b[1]*c[0]],length=Math.hypot(...normal);
    for(let j=0;j<3;j++){const key=edgeKey(face[j],face[(j+1)%3]);if(!edges.has(key))edges.set(key,[]);edges.get(key).push(normal.map(v=>v/length));}}
  const features=new Set([...edges].filter(([,normals])=>normals.length!==2||Math.abs(normals[0].reduce((s,v,j)=>s+v*normals[1][j],0))<Math.cos(Math.PI/6)).map(([key])=>key));
  const min=[0,1,2].map(j=>Math.min(...nodes.map(p=>p[j]))),max=[0,1,2].map(j=>Math.max(...nodes.map(p=>p[j]))),center=min.map((v,j)=>(v+max[j])/2),span=Math.hypot(...min.map((v,j)=>max[j]-v));
  let peak=0;for(let i=0;i<field.length;i+=3)peak=Math.max(peak,Math.hypot(field[i],field[i+1],field[i+2]));
  const first=initialView(result.scene.geometry.kind,min,max),kept=viewState?.view;
  ({yaw,pitch}=kept??first);zoom=kept?.zoom??1;panX=kept?.panX??0;panY=kept?.panY??0;$('#solid-thin').checked=kept?kept.thin:first.stretch>1;
  const drawn=()=>$('#solid-thin').checked?Math.max(first.stretch,Math.round((max[0]-min[0])/(10*(max[2]-min[2]||1))),1):1;
  $('#solid-scale').value=Math.min(1e7,Math.round(span*.05/(peak*1000||1)));
  $('#solid-frame').value=frame;
  // the drawn position of a point moved by `move` (mm): the thickness is stretched, the displacement is not
  const project=(p,move=[0,0,0])=>{const x=p[0]-center[0]+move[0],y=p[1]-center[1]+move[1],z=(p[2]-center[2])*stretch+move[2],a=Math.cos(yaw)*x-Math.sin(yaw)*y,b=Math.sin(yaw)*x+Math.cos(yaw)*y;return [a,Math.cos(pitch)*z-Math.sin(pitch)*b,Math.sin(pitch)*z+Math.cos(pitch)*b];};
  function draw() {
    if(disposed)return;
    const box=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2),w=Math.max(240,box.width),h=Math.max(300,box.height);canvas.width=w*ratio;canvas.height=h*ratio;ctx.scale(ratio,ratio);
    ctx.fillStyle='#f3f7f8';ctx.fillRect(0,0,w,h);
    stretch=drawn();const cx=w/2+panX,cy=h/2+panY;
    // fit the body as it is seen from here: the box of its eight corners on the screen
    let wide=0,high=0;for(let k=0;k<8;k++){const q=project([k&1?max[0]:min[0],k&2?max[1]:min[1],k&4?max[2]:min[2]]);wide=Math.max(wide,Math.abs(q[0]));high=Math.max(high,Math.abs(q[1]));}
    const scale=.86*Math.min(w/(2*wide||span),h/(2*high||span))*zoom,multiplier=Math.max(0,Math.min(1e7,Number($('#solid-scale').value)||0)),component=$('#solid-component').value,offset=frame*n*3;
    const projected=[],values=[];
    let framePeak=0;
    for(let i=0;i<n;i++){const u=[field[offset+i*3],field[offset+i*3+1],field[offset+i*3+2]],p=project(nodes[i],u.map(v=>v*1000*multiplier));projected.push([cx+p[0]*scale,cy-p[1]*scale,p[2]]);
      const value=component==='magnitude'?Math.hypot(...u):component==='direction'?u.reduce((a,v,j)=>a+v*direction[j],0):u[Number(component)];values.push(value);framePeak=Math.max(framePeak,Math.abs(value));}
    const mode=$('#solid-color-mode').value,top=colorPeak(mode,Number($('#solid-color-max').value),peak,framePeak);$('#solid-color-max').disabled=mode!=='custom';
    const triangles=result.surface.map((face,index)=>({face,index,depth:face.reduce((s,i)=>s+projected[i][2],0)})).sort((a,b)=>a.depth-b.depth);
    for(const {face,index} of triangles){const value=face.reduce((s,i)=>s+values[i],0)/3,q=Math.max(-1,Math.min(1,value/(top||1))),m=component==='magnitude'?(q*2-1):q;
      const rgb=m>=0?[245,Math.round(238-155*m),Math.round(225-170*m)]:[Math.round(235+160*m),Math.round(240+70*m),246];
      ctx.beginPath();face.forEach((id,j)=>{const p=projected[id];if(j===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});ctx.closePath();ctx.fillStyle=`rgb(${rgb.join(',')})`;ctx.fill();ctx.strokeStyle=loadedFaces.has(index)?'#cc874a':'#314d6026';ctx.lineWidth=loadedFaces.has(index)?.8:.35;ctx.stroke();
      for(let j=0;j<3;j++)if(features.has(edgeKey(face[j],face[(j+1)%3]))){const a=projected[face[j]],b=projected[face[(j+1)%3]];ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.strokeStyle=loadedFaces.has(index)?'#b47734':'#41647e';ctx.lineWidth=.8;ctx.stroke();}}
    if(probe){const location=probe.location,moved=!result.display&&location.ids?.length,point=moved?location.ids.reduce((sum,id,k)=>sum.map((v,j)=>v+location.weights[k]*nodes[id][j]),[0,0,0]):location.point_mm,
      move=moved?location.ids.reduce((sum,id,k)=>sum.map((v,j)=>v+location.weights[k]*field[offset+id*3+j]*1000*multiplier),[0,0,0]):[0,0,0],p=project(point,move);
      ctx.beginPath();ctx.arc(cx+p[0]*scale,cy-p[1]*scale,5,0,2*Math.PI);ctx.fillStyle='#103d52';ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=2;ctx.stroke();}
    ctx.fillStyle='#537482';ctx.font='12px sans-serif';ctx.fillText(`X / Y / Z · mm     t = ${result.times_us[frame].toFixed(2)} μs`,15,24);
    // the axes as they are drawn: which way x, y and z point in this view
    const origin=[34,h-34];ctx.font='11px sans-serif';
    [['X','#c0392b',[1,0,0]],['Y','#2f855a',[0,1,0]],['Z','#2b6cb0',[0,0,1]]].forEach(([name,colour,axis])=>{const a=Math.cos(yaw)*axis[0]-Math.sin(yaw)*axis[1],b=Math.sin(yaw)*axis[0]+Math.cos(yaw)*axis[1],y=Math.cos(pitch)*axis[2]-Math.sin(pitch)*b;
      if(Math.hypot(a,y)<.15)return;ctx.beginPath();ctx.moveTo(origin[0],origin[1]);ctx.lineTo(origin[0]+a*22,origin[1]-y*22);ctx.strokeStyle=colour;ctx.lineWidth=1.6;ctx.stroke();ctx.fillStyle=colour;ctx.fillText(name,origin[0]+a*30-3,origin[1]-y*30+4);});
    $('#solid-time').textContent=result.times_us[frame].toFixed(2)+' μs';$('#solid-color-note').textContent=`色标（${COLOR_MODES[mode]}）：${component==='magnitude'?'0':'−'+(top*1e9).toPrecision(4)} 至 ${(top*1e9).toPrecision(4)} nm；形变显示 ${multiplier} 倍${stretch>1?`；板厚方向放大 ${stretch} 倍显示（位移不放大）`:''}。`;
  }
  async function extract() {
    const token=++extraction;probe=null;analysis=null;$('#solid-csv').disabled=true;$('#solid-probe-message').textContent='正在读取完整采样记录…';
    try {
      if([0,1,2].some(j=>$('#solid-point-'+j).value===''||$('#solid-direction-'+j).value===''))throw new Error('请填写完整坐标与方向。');
      const point=[0,1,2].map(j=>Number($('#solid-point-'+j).value)),dir=[0,1,2].map(j=>Number($('#solid-direction-'+j).value));
      const next=await extractSeries(result,field,point,dir);if(disposed||token!==extraction)return;probe=next;direction=probe.direction;
      if($('#solid-window-start').value===''||$('#solid-window-end').value==='')throw new Error('请填写分析时间窗。');
      analysis=analyzeTimeWindow(probe.projected_m,probe.times_us??result.times_us,result.scene.source.frequency_mhz,Number($('#solid-window-start').value),Number($('#solid-window-end').value),result.scene.signal);
      $('#solid-csv').disabled=false;
      const wave=probe.projected_m.map(v=>v*1e9),spectrum=analysis.spectrum;
      $('#solid-waveform').innerHTML=chart(probe.times_us??result.times_us,wave,'时间 / μs','位移 / nm');
      const stop=Math.min(spectrum.nyquist_mhz,Math.max(spectrum.bin_spacing_mhz,Number($('#solid-frequency-limit').value)||1)),indices=spectrum.frequency_mhz.map((f,i)=>f<=stop?i:-1).filter(i=>i>=0);
      $('#solid-spectrum').innerHTML=chart(indices.map(i=>spectrum.frequency_mhz[i]),indices.map(i=>spectrum.amplitude_m[i]*1e9),'频率 / MHz','幅值 / nm','#5b80b6');
      $('#solid-spectrum-note').textContent=`分析窗 ${analysis.start_us.toFixed(3)}–${analysis.end_us.toFixed(3)} μs（${analysis.sample_count} 点）；分辨率约 ${spectrum.record_resolution_mhz.toFixed(4)} MHz，FFT 频点间隔 ${spectrum.bin_spacing_mhz.toFixed(4)} MHz，Nyquist ${spectrum.nyquist_mhz.toFixed(2)} MHz。零填充仅增加显示频点；频谱峰不能单独证明材料非线性。当前显示 0–${stop.toFixed(2)} MHz。${analysis.carrier_cycles<3?'该窗口不足三个基频周期，谐波幅值容易受到窗函数和泄漏影响。':''}`;
      $('#solid-harmonics').innerHTML=`<div class="table-scroll"><table><thead><tr><th>当前响应</th><th>频率 / MHz</th><th>幅值 / nm</th><th>相位 / °</th></tr></thead><tbody>${analysis.harmonics.map(h=>`<tr><td>${h.order===1?'f':'2f'}</td><td>${h.frequency_mhz.toPrecision(5)}</td><td>${h.available?(h.amplitude_m*1e9).toExponential(5):'超过 Nyquist'}</td><td>${h.available&&h.phase_deg!==null?h.phase_deg.toFixed(3):'—'}</td></tr>`).join('')}</tbody></table></div>`;
      if(analysis.bands.length)$('#solid-harmonics').innerHTML='<p>线性扫频使用频带功率，不计算固定 f / 2f 系数。Hann 加权频带 RMS：</p><ul>'+analysis.bands.map(b=>`<li>${b.start_mhz.toFixed(3)}–${b.end_mhz.toFixed(3)} MHz：${(b.windowed_rms_m*1e9).toExponential(5)} nm</li>`).join('')+'</ul>';
      $('#solid-probe-message').textContent=`已提取 [${point.join(', ')}] mm，单位方向 [${direction.map(v=>v.toFixed(4)).join(', ')}]；未重新求解。`;$('#solid-probe-message').classList.remove('model-error');draw();
    }catch(error){if(disposed||token!==extraction)return;probe=null;analysis=null;$('#solid-csv').disabled=true;$('#solid-waveform').innerHTML='';$('#solid-spectrum').innerHTML='';$('#solid-harmonics').innerHTML='';$('#solid-spectrum-note').textContent='';$('#solid-probe-message').textContent=error.message;$('#solid-probe-message').classList.add('model-error');draw();}
  }
  function preset(){const p=result.scene.receivers[Number($('#solid-receiver').value)];[0,1,2].forEach(j=>{$('#solid-point-'+j).value=p.point_mm[j];$('#solid-direction-'+j).value=p.direction[j];});extract();}
  function animate(time){if(disposed)return;if(playing&&time-last>66){frame=(frame+1)%result.times_us.length;$('#solid-frame').value=frame;draw();last=time;}timer=requestAnimationFrame(animate);}
  $('#solid-play').onclick=()=>{playing=!playing;$('#solid-play').textContent=playing?'暂停':'播放';};
  $('#solid-frame').oninput=()=>{frame=Number($('#solid-frame').value);draw();};
  $('#solid-component').onchange=draw;$('#solid-scale').onchange=draw;$('#solid-color-mode').onchange=draw;$('#solid-color-max').onchange=draw;$('#solid-color-max').oninput=draw;$('#solid-thin').onchange=draw;
  const VIEWS={iso:()=>first.strip ? [.55,1.05] : [first.yaw,first.pitch],side:()=>[0,0],top:()=>[0,Math.PI/2]};
  container.querySelectorAll('[data-solid-view]').forEach(b=>b.onclick=()=>{[yaw,pitch]=VIEWS[b.dataset.solidView]();zoom=1;panX=0;panY=0;draw();});$('#solid-receiver').onchange=preset;$('#solid-extract').onclick=extract;$('#solid-frequency-limit').onchange=extract;
  $('#solid-window-start').onchange=extract;$('#solid-window-end').onchange=extract;
  $('#solid-csv').onclick=()=>{if(!probe||!analysis)return;const spectrum=analysis.spectrum,lines=['# job_id,'+result.job_id,'# response_part,'+(result.response_part??'full'),'# point_mm,'+probe.location.point_mm.join(','),'# normalized_direction,'+probe.direction.join(','),'# analysis_window_us,'+analysis.start_us+','+analysis.end_us,'# spectrum,Hann weighted mean removed single-sided amplitude',...analysis.harmonics.map(h=>'# harmonic,'+[h.order,h.frequency_mhz,h.available?h.real_m:'',h.available?h.imaginary_m:'',h.available?h.amplitude_m:'',h.phase_deg??''].join(',')),'# harmonic_columns,order,frequency_mhz,real_m,imaginary_m,amplitude_m,phase_deg','time_us,u_x_m,u_y_m,u_z_m,projected_displacement_m'];
    lines.splice(2,0,'# signal_kind,'+(result.scene.signal?.kind??'hann'),...analysis.bands.map(b=>'# band_windowed_rms,'+[b.order,b.start_mhz,b.end_mhz,b.windowed_rms_m].join(',')));
    (probe.times_us??result.times_us).forEach((t,i)=>lines.push([t,...probe.xyz_m[i],probe.projected_m[i]].join(',')));
    lines.push('','frequency_mhz,single_sided_amplitude_m');spectrum.frequency_mhz.forEach((f,i)=>lines.push([f,spectrum.amplitude_m[i]].join(',')));
    const a=document.createElement('a'),url=URL.createObjectURL(new Blob([lines.join('\n')],{type:'text/csv;charset=utf-8'}));a.href=url;a.download=result.job_id+'-'+(result.response_part??'full')+'-point-spectrum.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};canvas.onpointerup=()=>drag=null;
  canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag[0],dy=e.clientY-drag[1];if(e.shiftKey){panX+=dx;panY+=dy;}else{yaw+=dx*.01;pitch+=dy*.01;}drag=[e.clientX,e.clientY];draw();};
  canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.3,Math.min(40,zoom*Math.exp(-e.deltaY*.001)));draw();};
  const resize=new ResizeObserver(draw);resize.observe(canvas);
  if(viewState){
    for(const [key,value] of Object.entries(viewState.inputs??{})){const element=$('#'+key);if(element)element.value=value;}
    extract();
  }else preset();
  timer=requestAnimationFrame(animate);
  const cleanup=()=>{disposed=true;cancelAnimationFrame(timer);resize.disconnect();};
  cleanup.snapshot=()=>({frame,view:{yaw,pitch,zoom,panX,panY,thin:$('#solid-thin').checked},inputs:Object.fromEntries(['solid-color-mode','solid-color-max','solid-receiver','solid-point-0','solid-point-1','solid-point-2','solid-direction-0','solid-direction-1','solid-direction-2','solid-window-start','solid-window-end','solid-frequency-limit','solid-component'].map(id=>[id,$('#'+id).value]))});
  return cleanup;
}
