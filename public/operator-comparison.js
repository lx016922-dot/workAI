import {unitVector,amplitudeSpectrum} from './solid-data.js';

const ocEscape=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ocNumber=v=>Number.isFinite(v)?v.toPrecision(5):'信号不足';
const ocPercent=v=>Number.isFinite(v)?(100*v).toFixed(2)+'%':'信号不足';
const ocTick=v=>v===0?'0':Math.abs(v)<.001?v.toExponential(2):v.toPrecision(4);

export function comparisonSeries(record,receiver,direction,part){
  const d=unitVector(direction),t=record.times_us,dt=t?.[1]-t?.[0],ids={full:[0,1,2],linear:[0],geometric:[1],material:[2]}[part];
  if(!ids||!Array.isArray(t)||t.length<4||!(dt>0)||!t.every((v,i)=>Number.isFinite(v)&&Math.abs(v-t[0]-i*dt)<1e-8)||!(record.frequency_mhz>0))throw Error('波形采样或响应选择无效。');
  const project=field=>{
    if(!Array.isArray(field)||field.length!==t.length||!field.every(row=>row.length===3&&row.every(v=>v.length===3&&v.every(Number.isFinite))))throw Error('三分量训练结果不完整。');
    return field.map(row=>ids.reduce((sum,k)=>sum+row[k].reduce((s,v,j)=>s+v*d[j],0),0));
  };
  const prediction=project(receiver.predicted_parts_m),fem=project(receiver.fem_parts_m),norm=fem.reduce((s,v)=>s+v*v,0),relative=norm>1e-30?Math.sqrt(prediction.reduce((s,v,i)=>s+(v-fem[i])**2,0)/norm):null;
  const coefficient=values=>{const w=t.map((_,i)=>.5*(1-Math.cos(2*Math.PI*i/(t.length-1)))),sum=w.reduce((a,b)=>a+b,0),mean=values.reduce((s,v,i)=>s+w[i]*v,0)/sum;let re=0,im=0;t.forEach((time,i)=>{const phase=4*Math.PI*record.frequency_mhz*time,value=2*w[i]*(values[i]-mean)/sum;re+=value*Math.cos(phase);im-=value*Math.sin(phase);});return {re,im,amplitude_m:Math.hypot(re,im)};};
  const predicted2f=coefficient(prediction),fem2f=coefficient(fem),secondError=fem2f.amplitude_m>1e-13?Math.hypot(predicted2f.re-fem2f.re,predicted2f.im-fem2f.im)/fem2f.amplitude_m:null;
  return {times_us:t,direction:d,part,predicted_m:prediction,fem_m:fem,relative_l2:relative,predicted_2f:predicted2f,fem_2f:fem2f,second_complex_relative_error:secondError,predicted_spectrum:amplitudeSpectrum(prediction,dt),fem_spectrum:amplitudeSpectrum(fem,dt)};
}

function comparisonChart(x,truth,prediction,xlabel,frequency){
  const all=[...truth,...prediction],low=Math.min(0,...all),high=Math.max(0,...all),pad=Math.max((high-low)*.06,1e-12),lo=low-pad,hi=high+pad,x0=x[0],x1=x.at(-1),px=v=>72+760*(v-x0)/(x1-x0),py=v=>230-200*(v-lo)/(hi-lo);
  let marks='';for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4;marks+=`<path d="M72 ${py(v)}H832" stroke="#e2eaef"/><text x="64" y="${py(v)+4}" text-anchor="end">${ocTick(v)}</text>`;const xx=x0+(x1-x0)*i/4;marks+=`<text x="${px(xx)}" y="249" text-anchor="middle">${xx.toFixed(2)}</text>`;}
  if(frequency)for(const [n,label] of [[1,'f'],[2,'2f']]){const v=n*frequency;if(v<=x1)marks+=`<path d="M${px(v)} 25V230" stroke="#acb9c4" stroke-dasharray="3 4"/><text x="${px(v)+3}" y="22">${label}</text>`;}
  const line=(values,color,dash)=>`<polyline fill="none" stroke="${color}" stroke-width="2" ${dash?'stroke-dasharray="6 4"':''} points="${values.map((v,i)=>`${px(x[i])},${py(v)}`).join(' ')}"/>`;
  return `<svg viewBox="0 0 880 285" role="img" aria-label="${ocEscape(xlabel)}对照" style="width:100%;min-width:0;font-size:11px;fill:#526773">${marks}${line(truth,'#245c80',false)}${line(prediction,'#cb6a45',true)}<text x="450" y="278" text-anchor="middle">${ocEscape(xlabel)}</text><text x="4" y="15">位移 / nm</text></svg>`;
}

export function mountOperatorComparison(root,data){
  const $=id=>root.querySelector('#'+id);let selected=null;
  const predictionLabel=ocEscape(data.prediction_label??'AI（实验）');
  const referenceComparison=data.comparison_kind==='reference_convergence',referenceLabel=referenceComparison?'较细高阶网格（1.0 mm）':'有限元';
  root.innerHTML=`<section class="panel"><div class="panel-body"><div class="form-grid"><label class="field"><span>未见测试结构</span><select id="oc-case">${data.cases.map((c,i)=>`<option value="${i}">${ocEscape(c.case_id)}</option>`).join('')}</select></label><label class="field"><span>保留接收点</span><select id="oc-probe"></select></label><label class="field"><span>响应</span><select id="oc-part"><option value="full">完整位移</option><option value="linear">线性位移</option><option value="geometric">几何非线性增量</option><option value="material">材料非线性增量</option></select></label>${['X','Y','Z'].map((a,i)=>`<label class="field"><span>接收方向 ${a}</span><input id="oc-direction-${i}" type="number" step="any" value="${i===0?1:0}"></label>`).join('')}</div><div class="actions"><button id="oc-apply" class="button">更新方向</button><button id="oc-csv" class="button secondary">导出当前波形 CSV</button></div><p id="oc-location" class="section-note"></p><p id="oc-message" role="status"></p></div></section><section class="panel"><div class="panel-heading"><h2>预测与有限元的方向波形</h2></div><div class="panel-body"><p>蓝色实线：有限元；橙色虚线：${predictionLabel}</p><div id="oc-wave"></div><div id="oc-metrics"></div></div></section><section class="panel"><div class="panel-heading"><h2>方向位移频谱</h2></div><div class="panel-body"><div id="oc-spectrum"></div><p class="section-note">全记录 Hann 窗、加权去均值、单边幅值；频谱含反射。补零不会提高物理频率分辨率。2f 差异使用载频两倍处的复数系数，不用最近的 FFT 格点代替。</p></div></section>`;
  const record=()=>data.cases[Number($('oc-case').value)],probe=()=>record().receivers[Number($('oc-probe').value)];
  if(referenceComparison){
    $('oc-case').parentElement.querySelector('span').textContent='参考计算工况';
    const panels=root.querySelectorAll('section.panel');panels[1].querySelector('h2').textContent='高阶网格加密前后的方向波形';
    panels[1].querySelector('.panel-body > p').textContent=`蓝色实线：${referenceLabel}；橙色虚线：${data.prediction_label}`;
  }
  function update(){try{
    const inputs=[0,1,2].map(i=>$('oc-direction-'+i));if(inputs.some(x=>x.value===''))throw Error('请填写完整的三分量方向。');
    const r=record(),p=probe(),s=comparisonSeries(r,p,inputs.map(x=>Number(x.value)),$('oc-part').value);selected={record:r,probe:p,series:s};
    $('oc-location').textContent=`点位 [${p.point_mm.map(ocNumber).join(', ')}] mm；单位方向 [${s.direction.map(v=>v.toFixed(4)).join(', ')}]；载频 ${r.frequency_mhz.toFixed(4)} MHz。`;
    $('oc-wave').innerHTML=comparisonChart(s.times_us,s.fem_m.map(v=>v*1e9),s.predicted_m.map(v=>v*1e9),'时间 / μs');
    const ps=s.predicted_spectrum,fs=s.fem_spectrum,limit=Math.min(fs.nyquist_mhz,Math.max(1,5*r.frequency_mhz)),keep=fs.frequency_mhz.map((f,i)=>f<=limit?i:-1).filter(i=>i>=0);
    $('oc-spectrum').innerHTML=comparisonChart(keep.map(i=>fs.frequency_mhz[i]),keep.map(i=>fs.amplitude_m[i]*1e9),keep.map(i=>ps.amplitude_m[i]*1e9),'频率 / MHz',r.frequency_mhz);
    $('oc-metrics').innerHTML=`<div class="table-scroll"><table><thead><tr><th>波形相对 L2</th><th>${predictionLabel} 的 2f / nm</th><th>${referenceLabel} 2f / nm</th><th>2f 复数差异</th></tr></thead><tbody><tr><td>${ocPercent(s.relative_l2)}</td><td>${ocNumber(s.predicted_2f.amplitude_m*1e9)}</td><td>${ocNumber(s.fem_2f.amplitude_m*1e9)}</td><td>${ocPercent(s.second_complex_relative_error)}</td></tr></tbody></table></div>`;
    $('oc-message').textContent=referenceComparison?'仅投影已保存的三分量参考波形。接近零的方向投影可能有很大相对差异，所选方向不自动获得精度保证。':'仅投影已保存的三分量波形，不启动新训练或求解。模型尚未通过代理验收；改变查看方向不能使模型自动通过。';$('oc-message').className='section-note';$('oc-csv').disabled=false;
  }catch(e){selected=null;$('oc-wave').innerHTML='';$('oc-spectrum').innerHTML='';$('oc-metrics').innerHTML='';$('oc-message').textContent=e.message;$('oc-message').className='model-error';$('oc-csv').disabled=true;}}
  function receiverChanged(){probe().direction.forEach((v,i)=>$('oc-direction-'+i).value=v);update();}
  function caseChanged(){$('oc-probe').innerHTML=record().receivers.map((p,i)=>`<option value="${i}">${ocEscape(p.name)}</option>`).join('');receiverChanged();}
  $('oc-case').onchange=caseChanged;$('oc-probe').onchange=receiverChanged;$('oc-part').onchange=update;$('oc-apply').onclick=update;
  $('oc-csv').onclick=()=>{if(!selected)return;const {record:r,probe:p,series:s}=selected,lines=[referenceComparison?'# comparison,high_order_reference_refinement':'# experimental_ai_not_accepted,true','# run_id,'+data.run_id,'# case_id,'+r.case_id,'# point_mm,'+p.point_mm.join(','),'# normalized_direction,'+s.direction.join(','),'# response,'+s.part,referenceComparison?'time_us,coarse_reference_m,fine_reference_m':'time_us,ai_displacement_m,fem_displacement_m'];s.times_us.forEach((t,i)=>lines.push([t,s.predicted_m[i],s.fem_m[i]].join(',')));const url=URL.createObjectURL(new Blob([lines.join('\n')],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=r.case_id+'-'+p.name+'-'+s.part+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);};
  caseChanged();return {getSelection:()=>selected};
}
