import { decodeSolidField } from './solid-data.js';
import { decodeSpectralField, spectralPointSeries } from './spectral-data.js';
import { solidResultMarkup, mountSolidResult } from './solid-viewer.js';
import { createSolidStudyUI } from './solid-study-ui.js';

export function createSolidUI({state,api,render,toast,e,heading}) {
  let templates=null,scene=null,jobs=[],selected='',status=null,result=null,field=null,error='',loading=false,submitting=false,poll=null,dispose=null,loaded=false,asset=null,uploading=false;
  let method='tetra',degree=3;
  const endpoint=id=>id.startsWith('spectral-')?'/api/spectral/jobs':'/api/solid/jobs';
  async function listJobs(){const values=await Promise.all([api('/api/solid/jobs'),api('/api/spectral/jobs')]);return values.flatMap(v=>v.jobs).sort((a,b)=>String(b.started_at_utc).localeCompare(String(a.started_at_utc)));}
  const studyUI=createSolidStudyUI({api,render,toast,e,getScene:()=>structuredClone(scene),isVisible:()=>state.view==='solid'});
  const label={plate:'平板',pipe:'圆管',hole:'带通孔板',notch:'带缺口板'};
  const phases={starting:'准备中',meshing:'生成网格',running:'瞬态求解中',exporting:'保存完整位移场',completed:'已完成',failed:'失败'};
  const value=path=>path.split('.').reduce((v,k)=>v[k],scene);
  const input=(name,path,step='any')=>`<label class="field"><span>${name}</span><input data-solid-path="${path}" type="number" step="${step}" value="${e(value(path))}"></label>`;
  const select=(name,path,options)=>`<label class="field"><span>${name}</span><select data-solid-path="${path}">${options.map(([k,v])=>`<option value="${k}" ${value(path)===k?'selected':''}>${v}</option>`).join('')}</select></label>`;
  function markup() {
    if(!loaded&&!loading)void load();
    const title=heading('三维超弹性研究','板、管、通孔与缺口 · 真实位移场、任意点位波形和频谱');
    if(!scene)return title+`<div class="notice ${error?'error-notice':''}">${e(error||'正在读取三维模型…')}</div>`;
    const g=scene.geometry;
    return title+`<div class="notice">材料采用已记录的多晶铝参考值，默认使用 Murnaghan 能量形式的超弹性本构。当前只计算小弹性应变范围，超过 0.001 的应变保护值会停止；不包含塑性或断裂。</div>
      ${error?`<div class="notice error-notice" role="alert">${e(error)}</div>`:''}
      <div class="solid-layout"><section class="panel"><div class="panel-heading"><h2>结构与材料</h2></div><div class="panel-body"><div class="form-grid">
      <label class="field"><span>参数化结构</span><select id="solid-template">${g.kind==='cad'?'<option value="cad" selected>已导入 CAD 实体</option>':''}${Object.entries(label).map(([k,v])=>`<option value="${k}" ${g.kind===k?'selected':''}>${v}</option>`).join('')}</select></label>
      ${g.kind==='cad'?`<div class="field"><span>导入文件</span><strong>${e(asset?.original_name||g.asset_id)}</strong></div>`:input('长度 / mm','geometry.length_mm')}
      ${g.kind==='cad'?'':g.kind==='pipe'?input('外半径 / mm','geometry.outer_radius_mm')+input('内半径 / mm','geometry.inner_radius_mm'):input('宽度 / mm','geometry.width_mm')+input('厚度 / mm','geometry.thickness_mm')}
      ${g.kind==='hole'?input('孔半径 / mm','geometry.hole_radius_mm')+input('孔中心 X / mm','geometry.hole_x_mm')+input('孔中心 Y / mm','geometry.hole_y_mm'):''}
      ${g.kind==='notch'?input('缺口中心 X / mm','geometry.notch_x_mm')+input('缺口宽度 / mm','geometry.notch_width_mm')+input('缺口深度 / mm','geometry.notch_depth_mm'):''}
      ${select('材料本构','constitutive',[['murnaghan','Murnaghan 材料非线性'],['stvk','几何非线性 StVK'],['linear','线性对照']])}
      ${select('固定端面（其余自由）','boundary.fixed_face',[['none','全部自由'],['xmin','X 最小端'],['xmax','X 最大端'],['ymin','Y 最小面'],['ymax','Y 最大面'],['zmin','Z 最小面'],['zmax','Z 最大面']])}
      </div><p class="section-note">板范围 X=[0,L]、Y=[0,W]、Z=[0,H]；管轴为 X，截面圆心 Y=Z=0。通孔沿 Z 贯穿；缺口从 Z 最大表面切入并贯穿板宽。</p>
      <div class="form-grid"><label class="field"><span>导入其他形状（≤8 MiB）</span><input id="solid-cad-file" type="file" accept=".step,.stp,.stl"></label><label class="field"><span>STL 坐标单位</span><select id="solid-cad-unit"><option value="1">mm</option><option value="0.001">μm</option><option value="1000">m</option></select></label></div><button class="button secondary small" id="solid-upload" ${uploading?'disabled':''}>${uploading?'检查几何中…':'导入 STEP / STL'}</button>
      <p class="section-note">STEP 按声明单位转换为 mm；STL 需要明确单位，且必须闭合、朝向一致。支持可网格化的单一实体，装配体与非流形网格会被拒绝。${asset?.bounds_mm?'当前包围盒（mm）：'+asset.bounds_mm.map(v=>Number(v).toFixed(3)).join(', ')+'；请据此填写实体内接收点。':''}</p>
      <p class="section-note">ρ=2700 kg/m³；λ=58.1、μ=26.1、l=−143、m=−297、n=−345 GPa。三阶常数采用参考体积 Green 应变定义。</p></div></section>
      <section class="panel"><div class="panel-heading"><h2>激励与求解</h2></div><div class="panel-body"><div class="form-grid">
      <label class="field"><span>数值求解方式</span><select id="solid-method"><option value="tetra" ${method==='tetra'?'selected':''}>四面体计算（支持 CAD）</option><option value="spectral" ${method==='spectral'?'selected':''}>高阶计算（参数化结构）</option></select></label>
      ${method==='spectral'?`<label class="field"><span>高阶单元阶次</span><select id="solid-degree"><option value="3" ${degree===3?'selected':''}>3 阶</option><option value="4" ${degree===4?'selected':''}>4 阶</option></select></label>`:''}
      ${select('名义牵引加载面','source.face',[['xmin','X 最小端'],['xmax','X 最大端'],['ymin','Y 最小面'],['ymax','Y 最大面'],['zmin','Z 最小面'],['zmax','Z 最大面'],['surface_patch','任意曲面局部区域']])}
      ${input('应力幅值 / MPa','source.amplitude_mpa')}${input('载频 / MHz','source.frequency_mhz')}${input('Hann 脉冲周期数','source.cycles')}${input('延迟 / μs','source.delay_us')}${input('相位 / °','source.phase_deg')}
      ${[0,1,2].map((i)=>input('方向 '+['X','Y','Z'][i],'source.direction.'+i)).join('')}
      ${input('孔径半径 / mm（0 为整面）','source.radius_mm')}${[0,1,2].map(i=>input('孔径中心 '+['X','Y','Z'][i]+' / mm','source.center_mm.'+i)).join('')}
      ${scene.source.face==='surface_patch'?[0,1,2].map(i=>input('区域外法线 '+['X','Y','Z'][i],'source.patch_normal.'+i)).join(''):''}
      ${input('时长 / μs','acquisition.duration_us')}${input('输出间隔 / μs','acquisition.sample_step_us')}${input('网格目标尺寸 / mm','numerics.h_mm')}
      </div><p class="section-note">当前为等效表面牵引激励，不是压电换能器电压模型。${method==='spectral'?'高阶计算支持板、管、通孔和缺口，使用完整 X 端面加载（孔径半径为 0），可固定另一 X 端面。最多 30000 节点、192 MiB 位移场；新增工况仍需网格和时间步对照。':'曲面区域按中心球形半径与外法线夹角小于 60° 选面，力的方向另行指定。最大 40000 四面体、12000 节点。'}具体网格和时间步会在求解前检查。</p>
      <button class="button primary" id="solid-submit" ${submitting?'disabled':''}>${submitting?'正在提交…':'开始三维计算'}</button>
      </div></section></div>
      <section class="panel"><div class="panel-heading"><h2>初始接收点</h2><button class="button secondary small" id="solid-add-receiver">增加点位</button></div><div class="panel-body">${scene.receivers.map((p,i)=>`<div class="solid-receiver-row"><strong>${e(p.name)}</strong><div class="form-grid solid-vectors">${[0,1,2].map(j=>input(['X','Y','Z'][j]+' / mm',`receivers.${i}.point_mm.${j}`)).join('')}${[0,1,2].map(j=>input('方向 '+['X','Y','Z'][j],`receivers.${i}.direction.${j}`)).join('')}</div></div>`).join('')}<p class="section-note">求解完成后仍可指定新点位和新方向，直接从完整位移场插值。</p></div></section>
      <section class="panel"><div class="panel-heading"><h2>三维计算记录</h2><button class="button secondary small" id="solid-refresh">刷新</button></div><div class="panel-body"><select id="solid-job-select"><option value="">选择任务</option>${jobs.map(j=>`<option value="${j.job_id}" ${j.job_id===selected?'selected':''}>${j.job_id.startsWith('spectral-')?'高阶 · ':''}${e(j.job_id.slice(0,18))}… · ${e(phases[j.phase]||j.phase)}</option>`).join('')}</select>
      ${status?`<p class="section-note">${e(phases[status.phase]||status.phase)} · ${status.step??0}/${status.steps_total??'—'} 步 · ${Math.round(status.elapsed_s||0)} 秒</p>${status.error?`<div class="notice error-notice">${e(status.error)}</div>`:''}`:''}${result?'<button class="button secondary small" id="solid-reuse">以此任务参数编辑</button>':''}<button class="button secondary small" id="solid-export-scene">导出当前配置 JSON</button></div></section>
      ${result?`<div id="solid-result-container">${solidResultMarkup(result)}</div>`:loading?'<div class="loading">正在读取结果…</div>':''}${method==='spectral'?'<p class="section-note">当前高阶模式提供单次三维位移场。五组非线性对照研究可切换至四面体计算模式，按该模式重新设置网格尺寸。</p>':studyUI.markup()}`;
  }
  async function load(){loading=true;try{const response=await api('/api/solid/templates');templates=response.templates;scene=structuredClone(templates.plate);jobs=await listJobs();loaded=true;}catch(err){error=err.message;loaded=true;}finally{loading=false;if(state.view==='solid')render();}}
  async function refresh(){try{jobs=await listJobs();if(selected)await read(selected);else if(state.view==='solid')render();}catch(err){error=err.message;if(state.view==='solid')render();}}
  async function read(id){clearTimeout(poll);selected=id;loading=true;error='';try{const base=endpoint(id)+'/'+id;status=await api(base);
    if(status.phase==='completed'&&result?.job_id!==id){const next=await api(base+'/result'),response=await fetch(base+'/field');if(!response.ok)throw new Error('位移场读取失败。');const decoded=await (id.startsWith('spectral-')?decodeSpectralField:decodeSolidField)(next,await response.arrayBuffer());if(selected===id){result=next;field=decoded;}}
    if(!['completed','failed'].includes(status.phase)&&state.view==='solid')poll=setTimeout(()=>void read(id),2500);
  }catch(err){error=err.message;}finally{loading=false;if(state.view==='solid')render();}}
  function leave(clearPoll=true){dispose?.();dispose=null;if(clearPoll)clearTimeout(poll);studyUI.leave(clearPoll);}
  function mount(){dispose?.();dispose=null;const main=document.querySelector('#main');
    if(!scene)return;
    main.querySelectorAll('[data-solid-path]').forEach(el=>el.onchange=()=>{const parts=el.dataset.solidPath.split('.'),key=parts.pop(),target=parts.reduce((v,k)=>v[k],scene);target[key]=el.tagName==='SELECT'?el.value:el.value===''?null:Number(el.value);if(el.dataset.solidPath==='source.face'){if(el.value==='surface_patch'){scene.source.patch_normal??=[-1,0,0];if(!scene.source.radius_mm)scene.source.radius_mm=1;}render();}});
    main.querySelector('#solid-template').onchange=ev=>{if(templates[ev.target.value]){scene=structuredClone(templates[ev.target.value]);render();}};
    main.querySelector('#solid-method').onchange=ev=>{method=ev.target.value;if(method==='spectral')studyUI.leave();render();};
    const order=main.querySelector('#solid-degree');if(order)order.onchange=ev=>{degree=Number(ev.target.value);};
    main.querySelector('#solid-upload').onclick=async()=>{const file=main.querySelector('#solid-cad-file').files[0],unit=Number(main.querySelector('#solid-cad-unit').value);if(!file)return toast('请先选择 STEP 或 STL 文件。');if(file.size>8*1024*1024)return toast('几何文件不能超过 8 MiB。');uploading=true;error='';render();try{const bytes=new Uint8Array(await file.arrayBuffer());let binary='';for(let i=0;i<bytes.length;i+=8192)binary+=String.fromCharCode(...bytes.subarray(i,i+8192));const uploaded=await api('/api/solid/assets',{filename:file.name,data_base64:btoa(binary)});asset=await api('/api/solid/assets/inspect',{asset_id:uploaded.asset_id,stl_unit_mm:uploaded.format==='step'?1:unit});scene=structuredClone(templates.plate);scene.name=uploaded.original_name;scene.geometry.kind='cad';scene.geometry.asset_id=uploaded.asset_id;scene.geometry.stl_unit_mm=asset.stl_unit_mm;toast('几何已导入，请设置加载区域和实体内接收点。');}catch(err){error=err.message;}finally{uploading=false;render();}};
    main.querySelector('#solid-add-receiver').onclick=()=>{if(scene.receivers.length>=32)return toast('最多 32 个初始接收点。');scene.receivers.push({...structuredClone(scene.receivers[0]),name:'P'+(scene.receivers.length+1)});render();};
    main.querySelector('#solid-submit').onclick=async()=>{if(submitting)return;submitting=true;error='';render();try{const r=await api(method==='spectral'?'/api/spectral/jobs':'/api/solid/jobs',method==='spectral'?{scene,degree}:scene);result=null;field=null;jobs=await listJobs();await read(r.job_id);toast('三维计算已开始，使用真实超弹性数值求解。');}catch(err){error=err.message;}finally{submitting=false;render();}};
    main.querySelector('#solid-refresh').onclick=()=>void refresh();main.querySelector('#solid-job-select').onchange=ev=>{result=null;field=null;if(ev.target.value)void read(ev.target.value);};
    main.querySelector('#solid-export-scene').onclick=()=>{const a=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(scene,null,2)],{type:'application/json'}));a.href=url;a.download='workai-solid-scene.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
    const reuse=main.querySelector('#solid-reuse');if(reuse)reuse.onclick=()=>{scene=structuredClone(result.scene);method=result.engine==='workai_spectral_reference_3d'?'spectral':'tetra';if(result.spectral)degree=result.spectral.degree;render();toast('已载入所选任务的参数，可以修改后重新计算。');};
    if(result&&field)dispose=mountSolidResult(main.querySelector('#solid-result-container'),result,field,result.engine==='workai_spectral_reference_3d'?spectralPointSeries:undefined);
    if(method!=='spectral')studyUI.mount();
  }
  return {markup,mount,leave,refresh};
}
