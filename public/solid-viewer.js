import { pointSeries, amplitudeSpectrum } from './solid-data.js';

const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function solidResultMarkup(r) {
  const d=r.diagnostics;
  const hybrid=r.engine==='workai_modal_energy_3d',spectral=r.engine==='workai_spectral_reference_3d';
  const reference=r.spectral_reference_checks,referencePassed=spectral&&reference?.threshold===.05&&['h_passed','p_passed','time_passed','field_passed'].every(k=>reference[k]===true);
  const referenceNotice=referencePassed?'<div class="notice">本工况的网格、阶次、时间步及质量加权全场检查通过 5% 门槛，涵盖完整位移和材料增量。它们不是连续域误差上界；新增点位与接收方向尚未逐点复核。</div>':'';
  const comparison=r.validation,passed=comparison?.spatial_with_changed_stable_step.selected_receivers_within_5_percent&&comparison?.temporal_same_mesh.selected_receivers_within_5_percent;
  const quality=comparison?`<section class="panel"><div class="panel-heading"><h2>所选接收点的数值对照</h2><span class="pill ${passed?'good':'warn'}">${passed?'所选点对照通过':'部分对照未通过'}</span></div><div class="table-scroll"><table><thead><tr><th>接收点</th><th>空间/稳定步长联合差异</th><th>同网格时间步差异</th></tr></thead><tbody>${comparison.spatial_with_changed_stable_step.rows.map((p,i)=>`<tr><td>${esc(p.name)}</td><td>${(100*p.vector_relative_l2).toFixed(4)}%</td><td>${(100*comparison.temporal_same_mesh.rows[i].vector_relative_l2).toFixed(4)}%</td></tr>`).join('')}</tbody></table></div><div class="panel-footer">阈值 5%。空间比较同时使用各网格对应的稳定时间步；时间比较复用完全相同网格。该检查只覆盖列出的接收点，不是全域误差上界，也不是 COMSOL 对标。新增点位尚未逐点复核。</div></section>`:'';
  return `<div class="notice ${hybrid?'error-notice':''}">${hybrid?`实验性降阶预测 · ${esc(r.scene.name)} · ${esc({full:'完整位移',linear:'线性位移',geometric:'几何非线性增量',material:'材料非线性增量'}[r.response_part])}。使用经典模态传播与三个学习能量耦合系数，尚未通过完整替代验收。动画不是有限元参考场；新增点位只做插值，未获得独立精度保证。`:spectral?`高阶超弹性有限元参考场 · ${esc(r.scene.name)} · ${r.response_part==='material'?'材料非线性增量（Murnaghan − StVK）':'完整位移'}。采用 ${r.spectral.degree} 阶谱单元，点位按相同高阶形函数提取。表面三角形用于显示。新增点位尚未逐点复核；此结果未用于启用 AI 替代求解。`:`三维超弹性数值结果 · ${esc(r.scene.name)}。采用实际四面体网格；当前代理模型未覆盖该结构。能量检查不能代替网格、时间步及 COMSOL 对照。`}</div>
  ${quality}${referenceNotice}<div class="solid-layout"><section class="panel"><div class="panel-heading"><h2>三维位移场动画</h2><span class="pill ${passed||referencePassed?'good':'warn'}">${referencePassed?'本工况离散检查通过':passed?'已有所选点对照':'数值精度待独立验证'}</span></div><div class="panel-body">
  <canvas id="solid-canvas" class="solid-canvas" aria-label="三维铝结构位移场动画"></canvas>
  <div class="solid-animation"><button class="button secondary small" id="solid-play">播放</button><input id="solid-frame" aria-label="动画时间帧" type="range" min="0" max="${r.times_us.length-1}" value="0"><span id="solid-time"></span></div>
  <div class="form-grid"><label class="field"><span>着色分量</span><select id="solid-component"><option value="magnitude">位移模长</option><option value="0">X 位移</option><option value="1">Y 位移</option><option value="2">Z 位移</option><option value="direction">接收方向投影</option></select></label><label class="field"><span>形变显示倍率（可设 0）</span><input id="solid-scale" type="number" min="0" max="10000000" step="100"></label></div>
  <p id="solid-color-note" class="section-note"></p><p class="section-note">拖动旋转，滚轮缩放。颜色为真实位移；形变按显示倍率放大。橙色边线表示加载区域。显示全部表面三角形，包括孔和管的内表面。</p>
  </div></section><section class="panel"><div class="panel-heading"><h2>任意点位与方向</h2></div><div class="panel-body">
  <label class="field"><span>预设接收点</span><select id="solid-receiver">${r.scene.receivers.map((p,i)=>`<option value="${i}">${esc(p.name)}</option>`).join('')}</select></label>
  <div class="form-grid solid-vectors">${['x','y','z'].map((a,i)=>`<label class="field"><span>${a.toUpperCase()} / mm</span><input id="solid-point-${i}" type="number" step="any"></label>`).join('')}${['x','y','z'].map((a,i)=>`<label class="field"><span>方向 ${a}</span><input id="solid-direction-${i}" type="number" step="any"></label>`).join('')}</div>
  <button class="button primary" id="solid-extract">提取波形与频谱</button><p id="solid-probe-message" class="section-note" role="status"></p>
  <label class="field"><span>频谱显示上限 / MHz</span><input id="solid-frequency-limit" type="number" min="0.001" max="${.5/r.scene.acquisition.sample_step_us}" step="any" value="${Math.min(.5/r.scene.acquisition.sample_step_us,Math.max(r.scene.source.frequency_mhz*5,1))}"></label>
  <p class="section-note">使用完整时域节点场进行${spectral?'高阶谱单元':'四面体'}插值，可在求解后增加实体内的点位。方向自动归一化；输出为有正负号的位移投影，不是位移模长。</p>
  <button class="button secondary small" id="solid-csv">下载当前点 CSV</button>
  <dl class="definition"><dt>网格</dt><dd>${r.nodes_mm.length} 节点 / ${spectral?r.spectral.elements.length+' 高阶六面体':r.tetra.length+' 四面体'}</dd><dt>场存储精度</dt><dd>${esc(d.output_dtype)}</dd><dt>内部步长</dt><dd>${d.internal_step_us.toPrecision(4)} μs</dd><dt>能量收支误差</dt><dd>${Number.isFinite(d.external_work_balance_relative)?(100*d.external_work_balance_relative).toPrecision(4)+'%':'尚未独立核验'}</dd><dt>完整场最大应变</dt><dd>${d.max_strain.toExponential(3)}</dd><dt>完整场最小 J</dt><dd>${d.min_J.toFixed(8)}</dd></dl>
  </div></section></div><section class="panel"><div class="panel-heading"><h2>方向位移波形</h2><span class="muted small-text">nm · 采样间隔 ${r.scene.acquisition.sample_step_us} μs</span></div><div class="panel-body"><div id="solid-waveform"></div></div></section>
  <section class="panel"><div class="panel-heading"><h2>单边幅值频谱</h2><span class="muted small-text">Hann 窗 · 去加权均值</span></div><div class="panel-body"><div id="solid-spectrum"></div><p id="solid-spectrum-note" class="section-note"></p></div></section>`;
}

function chart(x,y,labelX,labelY,color='#148778') {
  const width=900,height=250,left=65,right=20,top=15,bottom=40;
  const xmin=Math.min(...x),xmax=Math.max(...x);let ymin=Math.min(0,...y),ymax=Math.max(0,...y);if(ymax-ymin<1e-20){ymin=-1;ymax=1;}
  const sx=v=>left+(v-xmin)/(xmax-xmin||1)*(width-left-right),sy=v=>height-bottom-(v-ymin)/(ymax-ymin)*(height-top-bottom);
  const path=x.map((v,i)=>`${sx(v).toFixed(2)},${sy(y[i]).toFixed(2)}`).join(' ');
  const fmt=v=>Math.abs(v)>=.01&&Math.abs(v)<10000?v.toFixed(2):v.toExponential(1);
  const ticks=Array.from({length:5},(_,i)=>{const xv=xmin+(xmax-xmin)*i/4,yv=ymin+(ymax-ymin)*i/4;return `<path d="M${left} ${sy(yv)}H${width-right}" stroke="#e6edef"/><text x="${left-8}" y="${sy(yv)+4}" text-anchor="end">${fmt(yv)}</text><text x="${sx(xv)}" y="${height-bottom+20}" text-anchor="middle">${fmt(xv)}</text>`;}).join('');
  return `<svg class="solid-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(labelY)}对${esc(labelX)}曲线"><g font-size="11" fill="#748995">${ticks}<text x="${width/2}" y="${height-3}" text-anchor="middle">${esc(labelX)}</text><text x="7" y="11">${esc(labelY)}</text></g><polyline points="${path}" fill="none" stroke="${color}" stroke-width="1.7"/></svg>`;
}

export function mountSolidResult(container,result,field,extractSeries=pointSeries) {
  const $=s=>container.querySelector(s),canvas=$('#solid-canvas'),ctx=canvas.getContext('2d'),nodes=result.nodes_mm,n=nodes.length;
  const loadedFaces=new Set(result.loaded_faces||[]);
  let frame=Math.floor(result.times_us.length*.45),playing=false,timer,last=0,yaw=result.scene.geometry.kind==='pipe'?1.1:.55,pitch=result.scene.geometry.kind==='pipe'?.45:1.05,zoom=1,drag=null,probe=null,direction=[1,0,0],disposed=false;
  const edges=new Map(),edgeKey=(a,b)=>a<b?`${a},${b}`:`${b},${a}`;
  for(const face of result.surface){const a=nodes[face[0]],b=nodes[face[1]].map((v,j)=>v-a[j]),c=nodes[face[2]].map((v,j)=>v-a[j]),normal=[b[1]*c[2]-b[2]*c[1],b[2]*c[0]-b[0]*c[2],b[0]*c[1]-b[1]*c[0]],length=Math.hypot(...normal);
    for(let j=0;j<3;j++){const key=edgeKey(face[j],face[(j+1)%3]);if(!edges.has(key))edges.set(key,[]);edges.get(key).push(normal.map(v=>v/length));}}
  const features=new Set([...edges].filter(([,normals])=>normals.length!==2||Math.abs(normals[0].reduce((s,v,j)=>s+v*normals[1][j],0))<Math.cos(Math.PI/6)).map(([key])=>key));
  const min=[0,1,2].map(j=>Math.min(...nodes.map(p=>p[j]))),max=[0,1,2].map(j=>Math.max(...nodes.map(p=>p[j]))),center=min.map((v,j)=>(v+max[j])/2),span=Math.hypot(...min.map((v,j)=>max[j]-v));
  let peak=0;for(let i=0;i<field.length;i+=3)peak=Math.max(peak,Math.hypot(field[i],field[i+1],field[i+2]));
  $('#solid-scale').value=Math.min(1e7,Math.round(span*.05/(peak*1000||1)));
  $('#solid-frame').value=frame;
  const project=p=>{const [x,y,z]=p.map((v,j)=>v-center[j]),a=Math.cos(yaw)*x-Math.sin(yaw)*y,b=Math.sin(yaw)*x+Math.cos(yaw)*y;return [a,Math.cos(pitch)*z-Math.sin(pitch)*b,Math.sin(pitch)*z+Math.cos(pitch)*b];};
  function draw() {
    if(disposed)return;
    const box=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2),w=Math.max(240,box.width),h=Math.max(300,box.height);canvas.width=w*ratio;canvas.height=h*ratio;ctx.scale(ratio,ratio);
    ctx.fillStyle='#f3f7f8';ctx.fillRect(0,0,w,h);
    const scale=.8*Math.min(w,h)/span*zoom,multiplier=Math.max(0,Math.min(1e7,Number($('#solid-scale').value)||0)),component=$('#solid-component').value,offset=frame*n*3;
    const projected=[],values=[];
    for(let i=0;i<n;i++){const u=[field[offset+i*3],field[offset+i*3+1],field[offset+i*3+2]],p=project(nodes[i].map((v,j)=>v+u[j]*1000*multiplier));projected.push([w/2+p[0]*scale,h/2-p[1]*scale,p[2]]);
      values.push(component==='magnitude'?Math.hypot(...u):component==='direction'?u.reduce((a,v,j)=>a+v*direction[j],0):u[Number(component)]);}
    const triangles=result.surface.map((face,index)=>({face,index,depth:face.reduce((s,i)=>s+projected[i][2],0)})).sort((a,b)=>a.depth-b.depth);
    for(const {face,index} of triangles){const value=face.reduce((s,i)=>s+values[i],0)/3,q=Math.max(-1,Math.min(1,value/(peak||1))),m=component==='magnitude'?(q*2-1):q;
      const rgb=m>=0?[245,Math.round(238-155*m),Math.round(225-170*m)]:[Math.round(235+160*m),Math.round(240+70*m),246];
      ctx.beginPath();face.forEach((id,j)=>{const p=projected[id];if(j===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});ctx.closePath();ctx.fillStyle=`rgb(${rgb.join(',')})`;ctx.fill();ctx.strokeStyle=loadedFaces.has(index)?'#cc874a':'#314d6026';ctx.lineWidth=loadedFaces.has(index)?.8:.35;ctx.stroke();
      for(let j=0;j<3;j++)if(features.has(edgeKey(face[j],face[(j+1)%3]))){const a=projected[face[j]],b=projected[face[(j+1)%3]];ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.strokeStyle=loadedFaces.has(index)?'#b47734':'#41647e';ctx.lineWidth=.8;ctx.stroke();}}
    if(probe){const location=probe.location,point=location.ids.reduce((sum,id,k)=>sum.map((v,j)=>v+location.weights[k]*(nodes[id][j]+field[offset+id*3+j]*1000*multiplier)),[0,0,0]),p=project(point);
      ctx.beginPath();ctx.arc(w/2+p[0]*scale,h/2-p[1]*scale,5,0,2*Math.PI);ctx.fillStyle='#103d52';ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=2;ctx.stroke();}
    ctx.fillStyle='#537482';ctx.font='12px sans-serif';ctx.fillText(`X / Y / Z · mm     t = ${result.times_us[frame].toFixed(2)} μs`,15,24);
    $('#solid-time').textContent=result.times_us[frame].toFixed(2)+' μs';$('#solid-color-note').textContent=`固定色标：${component==='magnitude'?'0':'−'+(peak*1e9).toFixed(3)} 至 ${(peak*1e9).toFixed(3)} nm；形变显示 ${multiplier} 倍。`;
  }
  function extract() {
    try {
      if([0,1,2].some(j=>$('#solid-point-'+j).value===''||$('#solid-direction-'+j).value===''))throw new Error('请填写完整坐标与方向。');
      const point=[0,1,2].map(j=>Number($('#solid-point-'+j).value)),dir=[0,1,2].map(j=>Number($('#solid-direction-'+j).value));
      probe=extractSeries(result,field,point,dir);direction=probe.direction;
      $('#solid-csv').disabled=false;
      const wave=probe.projected_m.map(v=>v*1e9),spectrum=amplitudeSpectrum(probe.projected_m,result.scene.acquisition.sample_step_us);
      $('#solid-waveform').innerHTML=chart(result.times_us,wave,'时间 / μs','位移 / nm');
      const stop=Math.min(spectrum.nyquist_mhz,Math.max(spectrum.bin_spacing_mhz,Number($('#solid-frequency-limit').value)||1)),indices=spectrum.frequency_mhz.map((f,i)=>f<=stop?i:-1).filter(i=>i>=0);
      $('#solid-spectrum').innerHTML=chart(indices.map(i=>spectrum.frequency_mhz[i]),indices.map(i=>spectrum.amplitude_m[i]*1e9),'频率 / MHz','幅值 / nm','#5b80b6');
      $('#solid-spectrum-note').textContent=`记录分辨率约 ${spectrum.record_resolution_mhz.toFixed(4)} MHz，FFT 频点间隔 ${spectrum.bin_spacing_mhz.toFixed(4)} MHz，Nyquist ${spectrum.nyquist_mhz.toFixed(2)} MHz。零填充仅增加显示频点；频谱峰不能单独证明材料非线性。当前显示 0–${stop.toFixed(2)} MHz。`;
      $('#solid-probe-message').textContent=`已提取 [${point.join(', ')}] mm，单位方向 [${direction.map(v=>v.toFixed(4)).join(', ')}]；未重新求解。`;$('#solid-probe-message').classList.remove('model-error');draw();
    }catch(error){probe=null;$('#solid-csv').disabled=true;$('#solid-waveform').innerHTML='';$('#solid-spectrum').innerHTML='';$('#solid-probe-message').textContent=error.message;$('#solid-probe-message').classList.add('model-error');draw();}
  }
  function preset(){const p=result.scene.receivers[Number($('#solid-receiver').value)];[0,1,2].forEach(j=>{$('#solid-point-'+j).value=p.point_mm[j];$('#solid-direction-'+j).value=p.direction[j];});extract();}
  function animate(time){if(disposed)return;if(playing&&time-last>66){frame=(frame+1)%result.times_us.length;$('#solid-frame').value=frame;draw();last=time;}timer=requestAnimationFrame(animate);}
  $('#solid-play').onclick=()=>{playing=!playing;$('#solid-play').textContent=playing?'暂停':'播放';};
  $('#solid-frame').oninput=()=>{frame=Number($('#solid-frame').value);draw();};
  $('#solid-component').onchange=draw;$('#solid-scale').onchange=draw;$('#solid-receiver').onchange=preset;$('#solid-extract').onclick=extract;$('#solid-frequency-limit').onchange=extract;
  $('#solid-csv').onclick=()=>{if(!probe)return;const spectrum=amplitudeSpectrum(probe.projected_m,result.scene.acquisition.sample_step_us),lines=['# job_id,'+result.job_id,'# point_mm,'+probe.location.point_mm.join(','),'# normalized_direction,'+probe.direction.join(','),'# spectrum,Hann weighted mean removed single-sided amplitude','time_us,u_x_m,u_y_m,u_z_m,projected_displacement_m'];
    result.times_us.forEach((t,i)=>lines.push([t,...probe.xyz_m[i],probe.projected_m[i]].join(',')));
    lines.push('','frequency_mhz,single_sided_amplitude_m');spectrum.frequency_mhz.forEach((f,i)=>lines.push([f,spectrum.amplitude_m[i]].join(',')));
    const a=document.createElement('a'),url=URL.createObjectURL(new Blob([lines.join('\n')],{type:'text/csv;charset=utf-8'}));a.href=url;a.download=result.job_id+'-point-spectrum.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};canvas.onpointerup=()=>drag=null;
  canvas.onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*.01;pitch+=(e.clientY-drag[1])*.01;drag=[e.clientX,e.clientY];draw();}};
  canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.3,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));draw();};
  const resize=new ResizeObserver(draw);resize.observe(canvas);preset();timer=requestAnimationFrame(animate);
  return ()=>{disposed=true;cancelAnimationFrame(timer);resize.disconnect();};
}
