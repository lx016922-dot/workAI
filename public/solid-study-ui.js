import {harmonicViewerMarkup,mountHarmonicViewer} from './solid-harmonic-viewer.js';
import {buildSolidHarmonicReport} from './solid-harmonic-report.js';
import {planPlatePropagation,comparePropagationProbes} from './solid-propagation.js';
import {propagationSummaryMarkup} from './solid-propagation-viewer.js';
import {buildPropagationReport} from './solid-propagation-report.js';

export function createSolidStudyUI({api,render,toast,e,getScene,isVisible}){
  let loaded=false,loading=false,submitting=false,reading=false,studies=[],selected='',status=null,result=null,probe=null,error='',timer=null,receiverIndex=0,propagationProbes=null;
  const phase={starting:'准备中',running:'五组计算中',completed:'已完成',failed:'失败'};
  function redraw(){if(isVisible())render();}
  async function load(){loading=true;try{studies=(await api('/api/solid/studies')).studies;loaded=true;}catch(err){error=err.message;loaded=true;}finally{loading=false;redraw();}}
  function markup(){if(!loaded&&!loading)void load();return `<section class="panel"><div class="panel-heading"><div><h2>材料非线性五组对照</h2><p>同网格 · 同内部时间步 · 单一载频 · A 与 2A</p></div></div><div class="panel-body"><p class="section-note">使用上方当前几何、激励、采样和接收点，依次计算线性 A、几何 A、材料＋几何 A、几何 2A、材料＋几何 2A。将启动五次真实本地计算，使用 float64 完整位移场，基准幅值最多 5 MPa。</p>
    <button id="solid-study-launch" class="button primary" ${submitting?'disabled':''}>${submitting?'正在提交…':'运行五组非线性对照'}</button> <button id="solid-study-refresh" class="button secondary small">刷新对照记录</button>
    <label class="field"><span>对照研究</span><select id="solid-study-select"><option value="">选择已保存的对照</option>${studies.map(s=>`<option value="${s.study_id}" ${s.study_id===selected?'selected':''}>${e(s.study_id.slice(0,18))}… · ${phase[s.phase]||e(s.phase)} · ${s.completed_controls??0}/5</option>`).join('')}</select></label>
    ${status?`<p id="solid-study-status" class="section-note">${phase[status.phase]||e(status.phase)} · ${status.completed_controls??0}/5 · ${Math.round(status.elapsed_s||0)} 秒${status.active_role?' · '+e(status.active_role):''}</p>`:''}
    ${error||status?.error?`<div class="notice error-notice" role="alert">${e(error||status.error)}</div>`:''}
    ${result?`<p class="section-note">五组已完成。可指定新的实体内点位，读取五组完整场进行对照；该操作不会重新计算。</p><label class="field"><span>预设接收点</span><select id="solid-study-receiver">${result.scene.receivers.map((r,i)=>`<option value="${i}" ${i===receiverIndex?'selected':''}>${e(r.name)}</option>`).join('')}</select></label><div class="form-grid">${[0,1,2].map(j=>`<label class="field"><span>${['X','Y','Z'][j]} / mm</span><input id="solid-study-point-${j}" type="number" step="any" value="${probe?.point_mm[j]??result.scene.receivers[receiverIndex].point_mm[j]}"></label>`).join('')}</div><button id="solid-study-probe" class="button secondary" ${reading?'disabled':''}>${reading?'正在读取五组场…':'提取此点的五组波形'}</button> <button id="solid-propagation-read" class="button secondary" ${reading?'disabled':''}>比较预设点的直达波（X 方向）</button>`:''}
    </div></section>${propagationProbes?propagationSummaryMarkup(propagationProbes)+'<button id="solid-propagation-export" class="button secondary">导出沿程比较 HTML</button>':''}${probe?`<div class="row-actions"><button id="solid-harmonic-export" class="button secondary">导出此点的离线对照 HTML</button></div><div id="solid-harmonic-result">${harmonicViewerMarkup(probe)}</div>`:''}`;}
  async function read(id){clearTimeout(timer);selected=id;error='';try{const next=await api('/api/solid/studies/'+id);if(selected!==id)return;status=next;
    if(next.phase==='completed'&&result?.study_id!==id){const record=await api('/api/solid/studies/'+id+'/result');if(selected!==id)return;result=record;probe=null;receiverIndex=0;}
    if(!['completed','failed'].includes(next.phase)&&isVisible())timer=setTimeout(()=>void read(id),2500);
  }catch(err){if(selected===id)error=err.message;}finally{redraw();}}
  async function refresh(){await load();if(selected)await read(selected);}
  function mount(){const $=s=>document.querySelector(s),button=$('#solid-study-launch');if(!button)return;
    button.onclick=async()=>{if(submitting)return;submitting=true;error='';redraw();try{const reply=await api('/api/solid/studies',{scene:getScene()});result=null;probe=null;propagationProbes=null;await load();await read(reply.study_id);toast('五组对照已启动；每组使用相同网格和内部时间步。');}catch(err){error=err.message;}finally{submitting=false;redraw();}};
    $('#solid-study-refresh').onclick=()=>void refresh();$('#solid-study-select').onchange=event=>{clearTimeout(timer);selected=event.target.value;status=null;result=null;probe=null;propagationProbes=null;if(selected)void read(selected);else redraw();};
    if(result){$('#solid-study-receiver').onchange=event=>{receiverIndex=Number(event.target.value);const rx=result.scene.receivers[receiverIndex];[0,1,2].forEach(j=>$('#solid-study-point-'+j).value=rx.point_mm[j]);};
      $('#solid-study-probe').onclick=async()=>{if(reading)return;const inputs=[0,1,2].map(j=>$('#solid-study-point-'+j));if(inputs.some(i=>i.value===''))return toast('请填写完整点位。');const point=inputs.map(i=>Number(i.value)),id=selected,rx=result.scene.receivers[Number($('#solid-study-receiver').value)];reading=true;error='';redraw();try{const next=await api('/api/solid/studies/'+id+'/probe',{point_mm:point,direction:rx.direction});if(selected===id)probe=next;}catch(err){error=err.message;probe=null;}finally{reading=false;redraw();}};
    }
    if(result)$('#solid-propagation-read').onclick=async()=>{
      if(reading)return;const id=selected,scene=result.scene,receivers=scene.receivers;
      try{if(receivers.length<3||receivers.length>8)throw new Error('沿程比较需要 3–8 个预设接收点。');
        for(const rx of receivers){if(rx.point_mm[1]!==receivers[0].point_mm[1]||rx.point_mm[2]!==receivers[0].point_mm[2])throw new Error('预设点需保持相同 Y/Z 坐标，以比较同一传播路径。');const plan=planPlatePropagation(scene,rx.point_mm);if(!plan.available||!plan.passes_end_echo_screen)throw new Error(plan.reason??'建议窗未覆盖或可能含远端回波，请先调整结构、记录或接收点。');}
        reading=true;error='';propagationProbes=null;redraw();const collected=[];
        for(const rx of receivers){if(selected!==id)return;collected.push(await api('/api/solid/studies/'+id+'/probe',{point_mm:rx.point_mm,direction:[1,0,0]}));}
        comparePropagationProbes(collected);if(selected===id)propagationProbes=collected;
      }catch(err){if(selected===id)error=err.message;}finally{reading=false;redraw();}
    };
    if(propagationProbes)$('#solid-propagation-export').onclick=async()=>{const button=$('#solid-propagation-export'),saved=propagationProbes;button.disabled=true;try{
      const resources={},paths={css:'/styles.css',data:'/solid-data.js',dispersion:'/plate-dispersion.js',harmonics:'/solid-harmonics.js',propagation:'/solid-propagation.js',viewer:'/solid-harmonic-viewer.js',propagationViewer:'/solid-propagation-viewer.js'};
      await Promise.all(Object.entries(paths).map(async([key,path])=>{const response=await fetch(path);if(!response.ok)throw new Error('报告资源读取失败。');resources[key]=await response.text();}));
      const html=buildPropagationReport(saved,resources),a=document.createElement('a'),url=URL.createObjectURL(new Blob([html],{type:'text/html;charset=utf-8'}));a.href=url;a.download=saved[0].study_id+'-propagation.html';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    }catch(err){toast(err.message);}finally{button.disabled=false;}};
    if(probe){const viewer=mountHarmonicViewer($('#solid-harmonic-result'),probe);
      $('#solid-harmonic-export').onclick=async()=>{const button=$('#solid-harmonic-export');button.disabled=true;try{if(!viewer.getAnalysis())throw new Error('请先设置有效的方向和分析窗。');const paths={css:'/styles.css',data:'/solid-data.js',dispersion:'/plate-dispersion.js',propagation:'/solid-propagation.js',harmonics:'/solid-harmonics.js',viewer:'/solid-harmonic-viewer.js'},resources={};
        await Promise.all(Object.entries(paths).map(async([key,path])=>{const response=await fetch(path);if(!response.ok)throw new Error('报告资源读取失败。');resources[key]=await response.text();}));
        const html=buildSolidHarmonicReport({...viewer.getProbe(),analysis_window_us:viewer.getAnalysis().window_us},resources),a=document.createElement('a'),url=URL.createObjectURL(new Blob([html],{type:'text/html;charset=utf-8'}));a.href=url;a.download=probe.study_id+'-harmonics.html';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('已导出当前点的五组完整波形。');
      }catch(err){toast(err.message);}finally{button.disabled=false;}};
    }
  }
  return {markup,mount,refresh,leave:clear=>{if(clear)clearTimeout(timer);}};
}
