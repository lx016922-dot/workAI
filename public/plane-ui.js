// Plane-strain page: structured form + JSON editor for version-2 scenes, job list, viewer.
import { decodePlaneResult } from './plane-data.js';
import { planeResultMarkup, mountPlaneResult } from './plane-viewer.js';

const SIGNALS={tone_burst:'窗脉冲（载频 × 周期数）',gaussian_pulse:'高斯脉冲',ricker:'Ricker 子波',chirp:'线性调频',multi_tone:'多频（在 JSON 中编辑分量）',samples:'采样序列（在 JSON 中粘贴）'};
const WINDOWS=[['hann','Hann'],['tukey','Tukey'],['gaussian','高斯'],['rectangular','矩形']];
const FACES=[['xmin','X 最小端'],['xmax','X 最大端'],['zmin','Z 底面'],['zmax','Z 顶面']];

export function createPlaneUI({state,api,render,toast,e,heading}) {
  let templates=null,scene=null,degree=4,jobs=[],selected='',status=null,result=null,field=null,lines=null,dispersion=null,error='',loading=false,submitting=false,poll=null,dispose=null,loaded=false,jsonText='',jsonError='';
  const phases={starting:'准备中',meshing:'生成网格',running:'求解中',exporting:'保存结果',completed:'已完成',failed:'失败'};
  const value=path=>path.split('.').reduce((v,k)=>v?.[k],scene);
  const input=(name,path,extra='')=>`<label class="field"><span>${name}</span><input data-plane-path="${path}" type="number" step="any" ${extra} value="${e(value(path)??'')}"></label>`;
  const select=(name,path,options)=>`<label class="field"><span>${name}</span><select data-plane-path="${path}">${options.map(([k,v])=>`<option value="${k}" ${String(value(path))===String(k)?'selected':''}>${v}</option>`).join('')}</select></label>`;
  const boundary=type=>scene.boundaries.find(b=>b.type===type&&b.face!=='zmax');
  const absorbing=face=>scene.boundaries.find(b=>b.type==='absorbing'&&b.face===face);
  const midplane=()=>scene.boundaries.find(b=>b.face==='zmax'&&(b.type==='symmetry'||b.type==='antisymmetry'));
  function estimate(){
    const g=scene.geometry,h=scene.numerics.h_mm,p=degree,nx=Math.ceil(g.length_mm/h),nz=Math.ceil(g.thickness_mm/h),nodes=(nx*p+1)*(nz*p+1);
    const spacing=h*(p===2?.5:p===3?.2764:p===4?.1727:p===5?.1175:.085)/1,dt=.3*spacing*1e-3/(6391*1.05)*1e6,steps=Math.ceil(scene.acquisition.sample_step_us/dt)*Math.round(scene.acquisition.duration_us/scene.acquisition.sample_step_us);
    return {nodes,elements:nx*nz,steps,seconds:nodes*steps/2.2e6};
  }
  function markup() {
    if(!loaded&&!loading)void load();
    const title=heading('二维平面应变','x–z 截面非线性谱元 · 任意信号、多源、对称面与吸收层 · 接收线沿程分析与 f–k');
    if(!scene)return title+`<div class="notice ${error?'error-notice':''}">${e(error||'正在读取二维模板…')}</div>`;
    const g=scene.geometry,src=scene.source,sig=src.signal||{type:'tone_burst',window:'hann'},prof=src.profile||{type:'uniform'},est=estimate(),line=scene.receiver_lines?.[0];
    return title+`<div class="notice">平面应变：无限宽板、u_y = 0、沿 y 不变，力与能量按每米宽度计。本构、信号、边界与三维求解器同源；结果精度仍需本工况的网格/阶次对照。</div>
      ${error?`<div class="notice error-notice" role="alert">${e(error)}</div>`:''}
      <div class="solid-layout"><section class="panel"><div class="panel-heading"><h2>截面几何与数值设置</h2></div><div class="panel-body"><div class="form-grid">
      <label class="field full"><span>场景名称</span><input id="plane-name" maxlength="100" value="${e(scene.name)}"></label>
      ${select('截面类型','geometry.kind',[['plate','板'],['notch','顶面贯穿缺口板']])}${input('长度 x / mm','geometry.length_mm')}${input('厚度 z / mm','geometry.thickness_mm')}
      ${g.kind==='notch'?input('缺口中心 x / mm','geometry.notch_x_mm')+input('缺口宽度 / mm','geometry.notch_width_mm')+input('缺口深度 / mm','geometry.notch_depth_mm'):''}
      ${select('材料本构','constitutive',[['murnaghan','Murnaghan 材料非线性'],['stvk','几何非线性 StVK'],['linear','线性对照']])}
      <label class="field"><span>单元阶次</span><select id="plane-degree">${[2,3,4,5,6].map(p=>`<option value="${p}" ${p===degree?'selected':''}>${p} 阶</option>`).join('')}</select></label>${input('单元尺寸 h / mm','numerics.h_mm')}${input('CFL','numerics.cfl')}
      ${input('记录时长 / μs','acquisition.duration_us')}${input('输出间隔 / μs','acquisition.sample_step_us')}
      </div><p class="section-note">预估：${est.elements} 个单元、约 ${est.nodes} 个节点、${est.steps} 个内部时间步，单核约 ${est.seconds<60?Math.max(1,Math.round(est.seconds))+' 秒':(est.seconds/60).toFixed(1)+' 分钟'}（经验估计）。网页预算：≤ 60000 节点、场存储 ≤ 768 MiB。</p>
      <h3>边界</h3><div class="form-grid">
      <label class="field"><span>z 中面（zmax）</span><select id="plane-midplane"><option value="none" ${!midplane()?'selected':''}>整板</option><option value="symmetry" ${midplane()?.type==='symmetry'?'selected':''}>对称 · 半厚度模型，只保留 S 族</option><option value="antisymmetry" ${midplane()?.type==='antisymmetry'?'selected':''}>反对称 · 只保留 A 族</option></select></label>
      ${select('固定面','boundary.fixed_face',[['none','无'],['xmin','X 最小端'],['xmax','X 最大端'],['zmin','Z 底面']])}
      <label class="field"><span>xmax 吸收层长度 / mm（0 为无）</span><input id="plane-absorb-xmax" type="number" step="any" min="0" value="${e(absorbing('xmax')?.length_mm??0)}"></label>
      <label class="field"><span>xmin 吸收层长度 / mm（0 为无）</span><input id="plane-absorb-xmin" type="number" step="any" min="0" value="${e(absorbing('xmin')?.length_mm??0)}"></label>
      <label class="field"><span>吸收层设计衰减 / dB</span><input id="plane-absorb-db" type="number" step="any" min="10" max="120" value="${e(scene.boundaries.find(b=>b.type==='absorbing')?.attenuation_db??60)}"></label>
      </div><p class="section-note">半厚度模型时请把厚度设为原板的一半，激励与接收的 z 坐标相应调整。吸收层按纵波往返衰减设计，实际残余回波随频率变化，结果页会给出耗散能量。</p></div></section>
      <section class="panel"><div class="panel-heading"><h2>激励与接收</h2></div><div class="panel-body"><div class="form-grid">
      ${select('加载边','source.face',FACES)}${input('中心 x / mm','source.center_mm.0')}${input('中心 z / mm','source.center_mm.2')}${input('区域半长 / mm（0 为整边）','source.radius_mm')}
      ${input('方向 x','source.direction.0')}${input('方向 z','source.direction.2')}${input('幅值 / MPa','source.amplitude_mpa')}${input('载频 / MHz','source.frequency_mhz')}
      <label class="field"><span>信号类型</span><select id="plane-signal-type">${Object.entries(SIGNALS).map(([k,v])=>`<option value="${k}" ${sig.type===k?'selected':''}>${v}</option>`).join('')}</select></label>
      ${sig.type==='tone_burst'?select('窗函数','source.signal.window',WINDOWS)+input('周期数','source.cycles')+(sig.window==='tukey'?input('Tukey α','source.signal.tukey_alpha'):'')+(sig.window==='gaussian'?input('高斯宽度','source.signal.gaussian_width'):''):''}
      ${sig.type==='gaussian_pulse'?input('−6 dB 相对带宽','source.signal.bandwidth'):''}${sig.type==='chirp'?input('终止频率 / MHz','source.signal.end_frequency_mhz')+input('时长 / μs','source.signal.duration_us')+select('窗函数','source.signal.window',[['hann','Hann'],['rectangular','矩形']]):''}
      ${input('时延 / μs','source.delay_us')}${input('相位 / °','source.phase_deg')}
      <label class="field"><span>空间分布</span><select id="plane-profile-type"><option value="uniform" ${prof.type==='uniform'?'selected':''}>均匀</option><option value="gaussian" ${prof.type==='gaussian'?'selected':''}>高斯</option><option value="comb" ${prof.type==='comb'?'selected':''}>梳状</option></select></label>
      ${prof.type==='gaussian'?input('σ / mm','source.profile.sigma_mm'):''}${prof.type==='comb'?input('齿距 / mm','source.profile.pitch_mm')+input('齿数','source.profile.teeth','step="1" min="1" max="64"')+input('齿宽 / mm','source.profile.tooth_width_mm')+select('反号交替','source.profile.alternating_sign',[['false','否'],['true','是']]):''}
      </div><p class="section-note">额外激励（阵列、双频）、多频分量与采样序列请在下方 JSON 中编辑，字段说明见 docs/solver-excitation-v2.md。</p>
      <h3>接收点</h3>${scene.receivers.map((p,i)=>`<div class="form-grid solid-vectors"><label class="field"><span>名称</span><input data-plane-text="receivers.${i}.name" value="${e(p.name)}"></label>${input('x / mm',`receivers.${i}.point_mm.0`)}${input('z / mm',`receivers.${i}.point_mm.2`)}${input('方向 x',`receivers.${i}.direction.0`)}${input('方向 z',`receivers.${i}.direction.2`)}<label class="field"><span>&nbsp;</span><button class="button secondary small" data-plane-remove="${i}" ${scene.receivers.length<=1?'disabled':''}>移除</button></label></div>`).join('')}
      <button class="button secondary small" id="plane-add-receiver">增加接收点</button>
      <h3>接收线（沿程与 f–k 分析）</h3><div class="form-grid"><label class="field"><span>启用</span><select id="plane-line-enabled"><option value="yes" ${line?'selected':''}>是</option><option value="no" ${!line?'selected':''}>否</option></select></label>
      ${line?input('起点 x / mm','receiver_lines.0.from_mm.0')+input('终点 x / mm','receiver_lines.0.to_mm.0')+input('z / mm','receiver_lines.0.from_mm.2')+input('点数（2–1024）','receiver_lines.0.count','step="1" min="2" max="1024"')+input('方向 x','receiver_lines.0.direction.0')+input('方向 z','receiver_lines.0.direction.2'):''}</div>
      <div class="action-row plane-actions"><button class="button primary" id="plane-submit" ${submitting?'disabled':''}>${submitting?'正在提交…':'开始二维计算'}</button><button class="button secondary" id="plane-export">导出场景 JSON</button><label class="button secondary file-button">导入 JSON<input id="plane-import" type="file" accept=".json,application/json" hidden></label></div>
      </div></section></div>
      <section class="panel"><div class="panel-heading"><h2>场景 JSON</h2><button class="button secondary small" id="plane-json-apply">应用 JSON 到表单</button></div><div class="panel-body"><textarea id="plane-json" class="plane-json" rows="12" spellcheck="false">${e(jsonText)}</textarea>${jsonError?`<p class="section-note model-error">${e(jsonError)}</p>`:'<p class="section-note">表单改动会同步到这里；也可以直接编辑后点“应用”。提交时以当前 JSON 为准，服务器会再次完整校验。</p>'}</div></section>
      <section class="panel"><div class="panel-heading"><h2>二维计算记录</h2><button class="button secondary small" id="plane-refresh">刷新</button></div><div class="panel-body"><select id="plane-job-select"><option value="">选择任务</option>${jobs.map(j=>`<option value="${j.job_id}" ${j.job_id===selected?'selected':''}>${e(j.name||j.job_id.slice(0,14))} · ${e(phases[j.phase]||j.phase)} · ${e((j.started_at_utc||'').slice(5,16).replace('T',' '))}</option>`).join('')}</select>
      ${status?`<p class="section-note">${e(phases[status.phase]||status.phase)} · ${status.step??0}/${status.steps_total??'—'} 帧 · ${Math.round(status.elapsed_s||0)} 秒${status.nodes?` · ${status.nodes} 节点`:''}</p>${status.error?`<div class="notice error-notice">${e(status.error)}</div>`:''}`:''}
      ${result?'<button class="button secondary small" id="plane-reuse">以此任务参数编辑</button>':''}</div></section>
      ${result?`<div id="plane-result-container">${planeResultMarkup(result)}</div>`:loading?'<div class="loading">正在读取结果…</div>':''}`;
  }
  function syncJson(){jsonText=JSON.stringify(scene,null,2);jsonError='';}
  async function load(){loading=true;try{const response=await api('/api/plane/templates');templates=response.templates;scene=structuredClone(templates.plate);syncJson();jobs=(await api('/api/plane/jobs')).jobs;loaded=true;}catch(err){error=err.message;loaded=true;}finally{loading=false;if(state.view==='plane')render();}}
  async function refresh(){try{jobs=(await api('/api/plane/jobs')).jobs;if(selected)await read(selected);else if(state.view==='plane')render();}catch(err){error=err.message;if(state.view==='plane')render();}}
  async function loadDispersion(thickness){try{dispersion=await api('/api/plane/dispersion?thickness_mm='+encodeURIComponent(thickness));}catch{dispersion=null;}}
  async function read(id){clearTimeout(poll);selected=id;loading=true;error='';try{const base='/api/plane/jobs/'+id;status=await api(base);
    if(status.phase==='completed'&&result?.job_id!==id){const next=await api(base+'/result');const [fieldResponse,lineResponse]=await Promise.all([next.field?fetch(base+'/field'):null,fetch(base+'/lines')]);
      if((fieldResponse&&!fieldResponse.ok)||!lineResponse.ok)throw new Error('二维数据读取失败。');
      const decoded=await decodePlaneResult(next,fieldResponse?await fieldResponse.arrayBuffer():null,await lineResponse.arrayBuffer());
      if(!dispersion||dispersion.thickness_mm!==next.scene.geometry.thickness_mm)await loadDispersion(next.scene.geometry.thickness_mm);
      if(selected===id){result=next;field=decoded.field;lines=decoded.lines;}}
    if(!['completed','failed'].includes(status.phase)&&state.view==='plane')poll=setTimeout(()=>void read(id),2500);
  }catch(err){error=err.message;}finally{loading=false;if(state.view==='plane')render();}}
  function leave(clearPoll=true){dispose?.();dispose=null;if(clearPoll)clearTimeout(poll);}
  function setPath(path,raw){const parts=path.split('.'),key=parts.pop(),target=parts.reduce((v,k)=>v[k],scene);target[key]=raw;}
  function mount(){dispose?.();dispose=null;const main=document.querySelector('#main');if(!scene)return;
    main.querySelectorAll('[data-plane-path]').forEach(el=>el.onchange=()=>{let v=el.value;if(el.tagName==='INPUT')v=v===''?null:Number(v);else if(v==='true'||v==='false')v=v==='true';else if(!isNaN(Number(v))&&el.dataset.planePath.endsWith('teeth'))v=Number(v);setPath(el.dataset.planePath,v);syncJson();render();});
    main.querySelectorAll('[data-plane-text]').forEach(el=>el.onchange=()=>{setPath(el.dataset.planeText,el.value);syncJson();});
    main.querySelector('#plane-name').onchange=ev=>{scene.name=ev.target.value;syncJson();};
    main.querySelector('#plane-degree').onchange=ev=>{degree=Number(ev.target.value);render();};
    main.querySelector('#plane-signal-type').onchange=ev=>{const type=ev.target.value,f=scene.source.frequency_mhz;scene.source.signal={tone_burst:{type,window:'hann'},gaussian_pulse:{type,bandwidth:.6},ricker:{type},chirp:{type,end_frequency_mhz:f*1.5,duration_us:Math.max(4/f,1),window:'hann'},multi_tone:{type,components:[{frequency_mhz:f,weight:1,phase_deg:0},{frequency_mhz:f*1.5,weight:.5,phase_deg:0}],duration_us:Math.max(6/f,1),window:'hann'},samples:{type,sample_step_us:.1,values:Array.from({length:32},(_,i)=>Math.sin(2*Math.PI*i/16)*Math.sin(Math.PI*i/31)),normalize:true}}[type];if(type!=='tone_burst')scene.source.cycles=1;syncJson();render();};
    main.querySelector('#plane-profile-type').onchange=ev=>{const type=ev.target.value;if(type==='uniform')delete scene.source.profile;else scene.source.profile=type==='gaussian'?{type,sigma_mm:1}:{type,pitch_mm:10,teeth:3,tooth_width_mm:5,along:'x',alternating_sign:false};syncJson();render();};
    main.querySelector('#plane-midplane').onchange=ev=>{scene.boundaries=scene.boundaries.filter(b=>b.face!=='zmax');if(ev.target.value!=='none')scene.boundaries.push({type:ev.target.value,face:'zmax'});syncJson();render();};
    for(const face of ['xmin','xmax'])main.querySelector('#plane-absorb-'+face).onchange=ev=>{const length=Number(ev.target.value)||0,db=Number(main.querySelector('#plane-absorb-db').value)||60;scene.boundaries=scene.boundaries.filter(b=>!(b.type==='absorbing'&&b.face===face));if(length>0)scene.boundaries.push({type:'absorbing',face,length_mm:length,ramp:'cubic',attenuation_db:db});syncJson();render();};
    main.querySelector('#plane-absorb-db').onchange=ev=>{const db=Number(ev.target.value)||60;scene.boundaries.forEach(b=>{if(b.type==='absorbing')b.attenuation_db=db;});syncJson();render();};
    main.querySelector('#plane-line-enabled').onchange=ev=>{if(ev.target.value==='yes'&&!scene.receiver_lines?.length){const L=scene.geometry.length_mm,H=scene.geometry.thickness_mm;scene.receiver_lines=[{name:'top',from_mm:[L*.2,0,H],to_mm:[L*.75,0,H],count:111,direction:[1,0,0]}];}else if(ev.target.value==='no')scene.receiver_lines=[];syncJson();render();};
    main.querySelector('#plane-add-receiver').onclick=()=>{if(scene.receivers.length>=32)return toast('最多 32 个接收点。');const last=scene.receivers.at(-1);scene.receivers.push({...structuredClone(last),name:'P'+(scene.receivers.length+1)});syncJson();render();};
    main.querySelectorAll('[data-plane-remove]').forEach(el=>el.onclick=()=>{scene.receivers.splice(Number(el.dataset.planeRemove),1);syncJson();render();});
    main.querySelector('#plane-json-apply').onclick=()=>{try{const parsed=JSON.parse(main.querySelector('#plane-json').value);if(!parsed||parsed.schema!=='workai.hyperelastic.solid.v2'||parsed.dimension!=='2d_plane_strain'||!parsed.geometry||!parsed.source)throw new Error('需要 schema 为 workai.hyperelastic.solid.v2 且 dimension 为 2d_plane_strain 的场景。');parsed.boundaries??=[];parsed.receiver_lines??=[];parsed.receivers??=[];scene=parsed;syncJson();render();toast('已应用 JSON；提交时服务器会完整校验。');}catch(err){jsonError=err.message;jsonText=main.querySelector('#plane-json').value;render();}};
    main.querySelector('#plane-json').onchange=ev=>{jsonText=ev.target.value;};
    main.querySelector('#plane-export').onclick=()=>{const a=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify({scene,degree},null,2)],{type:'application/json'}));a.href=url;a.download='workai-plane-strain-scene.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
    main.querySelector('#plane-import').onchange=async ev=>{try{const file=ev.target.files[0];if(!file)return;if(file.size>512*1024)throw new Error('配置不能超过 512 KiB。');const d=JSON.parse(await file.text());const candidate=d.scene||d;if(candidate.schema!=='workai.hyperelastic.solid.v2'||candidate.dimension!=='2d_plane_strain')throw new Error('不是二维平面应变 v2 场景。');scene=candidate;scene.boundaries??=[];scene.receiver_lines??=[];if([2,3,4,5,6].includes(d.degree))degree=d.degree;syncJson();render();}catch(err){error=err.message;render();}ev.target.value='';};
    main.querySelector('#plane-submit').onclick=async()=>{if(submitting)return;submitting=true;error='';render();try{const body=JSON.parse(main.querySelector('#plane-json')?.value||jsonText);const r=await api('/api/plane/jobs',{scene:body,degree});result=null;field=null;lines=null;jobs=(await api('/api/plane/jobs')).jobs;await read(r.job_id);toast('二维计算已开始。');}catch(err){error=err.message;}finally{submitting=false;render();}};
    main.querySelector('#plane-refresh').onclick=()=>void refresh();main.querySelector('#plane-job-select').onchange=ev=>{result=null;field=null;lines=null;if(ev.target.value)void read(ev.target.value);else{selected='';status=null;render();}};
    const reuse=main.querySelector('#plane-reuse');if(reuse)reuse.onclick=()=>{scene=structuredClone(result.scene);degree=result.degree;syncJson();render();window.scrollTo({top:0,behavior:'smooth'});toast('已载入所选任务的参数。');};
    if(result&&lines)dispose=mountPlaneResult(main.querySelector('#plane-result-container'),result,field,lines,dispersion);
  }
  return {markup,mount,leave,refresh};
}
