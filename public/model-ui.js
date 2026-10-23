import { createResearchUI } from './research-ui.js';
import { createMaterialUI } from './material-ui.js';
import { createNonlinearUI } from './nonlinear-ui.js';
export function createModelUI({ state, api, render, toast, escape: e, heading, jobTable, submitStudy }) {
  const expandedLaws = new Set(['harmonics']);
  const nonlinear = () => state.model?.case.schema === 'workai.murnaghan.guided_plate.scene.v1';
  const research = () => nonlinear() || state.model?.case.schema === 'workai.recovered.guided_plate.scene.v1';
  const volume = () => state.model?.case.schema === 'workai.physics_ai.volume.case.v1';
  const get = path => path.split('.').reduce((value, key) => value?.[key], state.model.case);
  const input = (label, path, unit, min, max, fixed = false) => `<label class="field"><span>${e(label)} <small>${e(unit)}</small></span><input type="number" step="any" data-model-path="${path}" value="${e(get(path))}" ${min !== undefined ? `min="${min}"` : ''} ${max !== undefined ? `max="${max}"` : ''} ${fixed ? 'readonly' : ''}></label>`;
  const select = (label, path, options) => `<label class="field"><span>${e(label)}</span><select data-model-path="${path}" data-string>${options.map(([value, title]) => `<option value="${value}" ${get(path) === value ? 'selected' : ''}>${title}</option>`).join('')}</select></label>`;
  const saveButton = () => `<button class="button primary" data-model-action="save" ${state.savingModel ? 'disabled' : ''}>${state.savingModel ? '正在校验…' : '校验并保存'}</button>`;
  const modelStatus = () => `<div class="model-status" role="status">${state.modelError ? `<span class="model-error">${e(state.modelError)}</span>` : state.modelDirty ? '参数已修改，尚未保存。' : state.model.validated ? '参数已按当前模型规则校验并保存。' : '新建模板参数，尚未校验保存；不是原网站的历史模型。'}</div>`;
  function section(title, body, note = '') { return `<section class="panel form-panel"><div class="panel-heading"><h2>${title}</h2></div><div class="panel-body">${body}${note ? `<div class="section-note">${note}</div>` : ''}</div></section>`; }
  function nonlinearFocus() {
    // Preserve the reader's place when health/job responses refresh the page.
    document.querySelectorAll('.physics-laws details[data-law]').forEach(detail => {
      if (detail.open) expandedLaws.add(detail.dataset.law); else expandedLaws.delete(detail.dataset.law);
    });
    return `<section class="panel nonlinear-focus" aria-labelledby="nonlinear-title">
      <div class="panel-heading"><div><div class="eyebrow">当前研究重点</div><h2 id="nonlinear-title">非线性导波</h2><p>当前优先：材料非线性 · 基频与二次谐波的产生、累积</p></div><span class="pill ${nonlinear() ? 'good' : 'warn'}">${nonlinear() ? 'Murnaghan 三阶本构' : '当前为线性模板'}</span></div>
      <div class="panel-body"><p class="physics-scope">已新增铝薄板材料非线性模板，采用有限应变、三阶弹性与五组对照；原有线性模板继续保留。求解有效性需以实际输出与数值检查判断。</p>
        <div class="physics-laws">
          <details data-law="harmonics" ${expandedLaws.has('harmonics') ? 'open' : ''}><summary>材料非线性 · 二次谐波</summary><p>先确认基频与二次谐波的导波模态。相位失配 Δk = k₂(2ω) − 2k₁(ω) 接近零，还需要非零模态耦合，才能讨论持续累积。</p><p>对单一模态对，在弱非线性、无损耗、基频近似不衰减且无入射二次谐波时，匹配条件下 A₂ 随传播距离近似线性增长。有限脉冲还受群速度差和波包重叠限制。</p><p>材料三阶弹性常数、几何非线性开关及其对照组需分别明确；Murnaghan 与 Landau 参数须核对定义和单位。</p></details>
          <details data-law="mixing" ${expandedLaws.has('mixing') ? 'open' : ''}><summary>后续方向 · 导波混频</summary><p>和频为 f₁ + f₂，差频为 |f₁ − f₂|。频率关系之外，还需检查带传播方向的波矢匹配、模态耦合与两个波包的时空重叠。</p><p>同时施加两个频率不会自动产生非线性混频。需要已确认的非线性本构或接触机理，并与单源、线性对照比较。</p></details>
          <details data-law="contact" ${expandedLaws.has('contact') ? 'open' : ''}><summary>后续方向 · 呼吸裂纹</summary><p>需定义裂纹接触面、初始间隙、预载荷和接触定律，再判断开闭、谐波与调制响应。</p><p>当前几何中的表面开口缺口没有提供这些接触参数，不能据此认定已建立呼吸裂纹模型。</p></details>
        </div>
        <div class="section-note">结果判读：从同一次响应中提取 A₁、A₂，并注明信号量、单位、时间窗和接收位置。A₂ / A₁² 是有条件的相对指标，不能直接等同于材料绝对非线性系数。</div>
      </div></section>`;
  }
  function geometry() {
    const c = state.model.case, g = c.geometry;
    const length = Number(g.length_mm) > 0 ? Number(g.length_mm) : 1, width = Number(g.width_mm) > 0 ? Number(g.width_mm) : 1;
    const project = point => [80 + Math.min(1, Math.max(0, Number(point[0]) / length)) * 310 + Math.min(1, Math.max(0, Number(point[1]) / width)) * 110, 245 - Math.min(1, Math.max(0, Number(point[1]) / width)) * 130 - Math.min(1, Math.max(0, Number(point[2]) / (Number(g.thickness_mm) || 1))) * 20];
    const markers = (points, color, prefix) => points.map((point, i) => { const [x, y] = project(point.position_mm); return `<circle cx="${x}" cy="${y}" r="6" fill="${color}" stroke="white" stroke-width="2"/><text x="${x + 10}" y="${y - 9}" font-size="11" fill="${color}">${e(point.id || prefix + i)}</text>`; }).join('');
    let notch = '';
    if (c.notch.enabled) { const [x, y] = project([c.notch.x_mm, c.notch.y_mm, g.thickness_mm]); notch = `<path d="M${x - 5} ${y + 15}l15 -30" fill="none" stroke="#b97855" stroke-width="5"/><text x="${x + 12}" y="${y + 15}" font-size="10" fill="#a7795f">缺口</text>`; }
    return `<div class="geometry-view"><svg viewBox="0 0 590 330" role="img" aria-label="平板几何、激励和接收点示意"><defs><linearGradient id="plate-fill" x2="1" y2="1"><stop stop-color="#e2f0ed"/><stop offset="1" stop-color="#bad9d4"/></linearGradient></defs><path d="M80 225L390 225L500 95L190 95Z" fill="url(#plate-fill)" stroke="#75a99f"/><path d="M80 225V245H390V225Z" fill="#a4c6c1" stroke="#75a99f"/><path d="M390 225L500 95V115L390 245Z" fill="#88b4ad" stroke="#75a99f"/><g stroke="#c3dbd6" stroke-width="1"><path d="M142 225l110 -130M204 225l110 -130M266 225l110 -130M328 225l110 -130M116 182h310M153 139h310"/></g>${notch}${markers(c.sources, '#c28352', 'tx')}${markers(c.receivers, '#277fbb', 'rx')}<path d="M80 267H390m-310 -5v10m310 -10v10" stroke="#8da4ad"/><text x="235" y="287" fill="#748d97" text-anchor="middle" font-size="12">X · ${e(g.length_mm)} mm</text><text x="478" y="193" fill="#748d97" font-size="12">Y · ${e(g.width_mm)} mm</text><text x="83" y="209" fill="#748d97" font-size="11">Z · ${e(g.thickness_mm)} mm</text></svg><div class="geometry-legend"><span>● 激励</span><span>● 接收点</span><small>几何示意 · 厚度放大</small></div></div>`;
  }
  const materialUI = createMaterialUI({ state, e, section });
  const researchUI = createResearchUI({ state, e, input, select, heading, section, saveButton, modelStatus, jobTable, nonlinearFocus, materialPanel: materialUI.render });
  const nonlinearUI = createNonlinearUI({ state, e, input, heading, section, saveButton, modelStatus, jobTable, nonlinearFocus, materialPanel: materialUI.render });
  function renderModel() {
    if (nonlinear()) return nonlinearUI.renderModel();
    if (research()) return researchUI.renderModel();
    if (!state.model) return '<div class="loading">正在载入模型参数…</div>';
    const fixed = volume();
    return `${heading('建模', '定义平板几何与材料，为物理场和研究提供模型。', `<div class="row-actions"><button class="button secondary" data-model-action="export">导出参数</button>${saveButton()}</div>`)}${modelStatus()}<div class="model-layout"><div>${section('模型与几何', `<div class="form-grid"><label class="field full"><span>模型名称</span><input id="model-name" maxlength="80" value="${e(state.model.name)}"></label><label class="field full"><span>模型模板</span><select id="model-template"><option value="nonlinear">二维铝板 · 材料非线性</option><option value="research">二维导波 · 原研究模型（线性）</option><option value="plate" ${!fixed ? 'selected' : ''}>三维平板 · 可调几何</option><option value="volume" ${fixed ? 'selected' : ''}>200 × 100 × 2 mm 大体积板</option></select></label>${input('长度 X', 'geometry.length_mm', 'mm', 15, 100, fixed)}${input('宽度 Y', 'geometry.width_mm', 'mm', 8, 60, fixed)}${input('厚度 Z', 'geometry.thickness_mm', 'mm', 1, 5, fixed)}</div>`, fixed ? '当前大体积协议固定几何与材料；兼容计算模型尚未就绪，可保存参数。' : '支持有限三维平板。几何尺寸、网格预算和点位会在保存时一并校验。')}${section('材料 · 各向同性弹性体', `<div class="form-grid">${input('密度', 'material.rho_kg_m3', 'kg/m³', 500, 20000, fixed)}${input('纵波速度', 'material.cl_m_s', 'm/s', 1500, 10000, fixed)}${input('横波速度', 'material.ct_m_s', 'm/s', 500, 6000, fixed)}</div>`)}</div><div>${section('几何视图', geometry(), '点位随模型参数更新。此视图展示几何关系，不是求解后的声场。')}${section('表面开口缺口', `<label class="checkbox-field"><input type="checkbox" data-model-path="notch.enabled" ${state.model.case.notch.enabled ? 'checked' : ''} ${fixed ? 'disabled' : ''}>启用内部、未贯穿的表面缺口</label><div class="form-grid">${input('中心 X', 'notch.x_mm', 'mm', .1, 100, fixed)}${input('中心 Y', 'notch.y_mm', 'mm', .1, 100, fixed)}${input('长度', 'notch.length_mm', 'mm', .1, 100, fixed)}${input('宽度', 'notch.width_mm', 'mm', .2, 2, fixed)}${input('深度', 'notch.depth_mm', 'mm', .1, 5, fixed)}</div>`, fixed ? '大体积模板目前仅接受完整无缺口板。' : '激励和接收点需避开缺口，缺口深度应小于板厚。此处没有裂纹开闭接触设置。')}</div></div>`;
  }
  function pointFields(group, i) {
    const path = `${group}.${i}`, point = state.model.case[group][i], fixed = volume() && group === 'sources';
    return `<div class="point-editor"><div class="point-heading"><strong>${group === 'sources' ? '激励' : '接收点'} ${i + 1}</strong><button class="button text small" data-model-remove="${group}.${i}" ${state.model.case[group].length <= 1 || fixed ? 'disabled' : ''}>移除</button></div><div class="form-grid"><label class="field"><span>点位 ID</span><input data-model-path="${path}.id" data-string value="${e(point.id)}" maxlength="16"></label>${input('位置 X', `${path}.position_mm.0`, 'mm', 0, state.model.case.geometry.length_mm, fixed)}${input('位置 Y', `${path}.position_mm.1`, 'mm', 0, state.model.case.geometry.width_mm, fixed)}${input('位置 Z', `${path}.position_mm.2`, 'mm', 0, state.model.case.geometry.thickness_mm, fixed)}${group === 'sources' ? `${input('方向 X', `${path}.direction.0`, '', -1, 1, fixed)}${input('方向 Y', `${path}.direction.1`, '', -1, 1, fixed)}${input('方向 Z', `${path}.direction.2`, '', -1, 1, fixed)}<label class="field"><span>频率 <small>MHz，逗号分隔</small></span><input data-model-path="${path}.frequencies_mhz" data-array value="${e(point.frequencies_mhz.join(', '))}" ${fixed ? 'readonly' : ''}></label>${volume() ? input('位移幅值', `${path}.amplitude_nm`, 'nm', .1, 10) : input('应力幅值', `${path}.amplitude_pa`, 'Pa', 1, 10000000)}${input('周期数', `${path}.cycles`, '', volume() ? 3 : 2, volume() ? 7 : 8)}${input('延迟', `${path}.delay_us`, 'μs', 0, volume() ? 10 : 100)}${input('相位', `${path}.phase_deg`, '°', volume() ? -180 : -360, volume() ? 180 : 360)}${volume() ? input('激励宽度', `${path}.width_mm`, 'mm', 1, 1, true) + input('激励跨度', `${path}.span_mm`, 'mm', 1, 1, true) : input('激励半径', `${path}.radius_mm`, 'mm', .5, 5)}` : ''}</div></div>`;
  }
  function renderPhysics() {
    if (nonlinear()) return nonlinearUI.renderPhysics();
    if (research()) return researchUI.renderPhysics();
    if (!state.model) return '<div class="loading">正在载入物理场…</div>';
    const c = state.model.case;
    return `${heading('物理场', '非线性导波研究要点与现有平板模型设置。', saveButton())}${modelStatus()}${nonlinearFocus()}<div class="physics-intro"><span class="pill good">固体弹性波</span><span>各向同性材料</span><span>${volume() ? '法向表面位移激励' : '表面应力激励'}</span></div><div class="model-layout"><div>${section('激励源', c.sources.map((_, i) => pointFields('sources', i)).join('') + `<button class="button secondary small" data-model-add="sources" ${c.sources.length >= (volume() ? 1 : 3) ? 'disabled' : ''}>＋ 添加激励</button>`, volume() ? '固定法向位移激励，中心 (10, 50, 2) mm，载频 0.2 MHz。' : '支持 1–3 个顶面激励，每个激励支持 1–3 个频率（0.08–0.35 MHz）；方向向量需归一化。')}${section('边界条件', `<div class="form-grid">${volume() ? select('X 最小端面', 'boundary.x_min', [['free', '自由'], ['absorbing', '吸收']]) + select('X 最大端面', 'boundary.x_max', [['free', '自由'], ['absorbing', '吸收']]) : select('侧边界', 'boundary', [['free', '自由边界'], ['side_impedance', '阻抗边界']])}</div>`, volume() ? 'Y 与 Z 方向表面为自由边界。' : '按服务器现有弹性平板模板施加边界条件。')}</div><div>${section('点位视图', geometry())}${section('接收点', c.receivers.map((_, i) => pointFields('receivers', i)).join('') + `<button class="button secondary small" data-model-add="receivers" ${c.receivers.length >= 8 ? 'disabled' : ''}>＋ 添加接收点</button>`, volume() ? '支持 1–8 个板内或表面接收点。' : '支持 1–8 个顶面接收点，Z 坐标应等于板厚。')}</div></div>`;
  }
  function renderStudy() {
    if (nonlinear()) return nonlinearUI.renderStudy();
    if (research()) return researchUI.renderStudy();
    if (!state.model) return '<div class="loading">正在载入研究参数…</div>';
    const ready = !volume() || state.health?.services.volume?.data?.model_ready;
    return `${heading('研究', '设定时域采样与离散参数，运行模型并跟踪计算任务。', `<div class="row-actions">${saveButton()}<button class="button primary" data-model-action="run" ${!ready || state.modelDirty || !state.model.validated ? 'disabled' : ''}><span aria-hidden="true">▷</span> 开始计算</button></div>`)}${modelStatus()}<div class="section-note physics-study-scope">当前任务运行已有弹性平板模型。非线性本构、接触定律和谐波提取尚待恢复；此入口尚未验证非线性导波求解能力。</div>${!ready ? '<div class="notice study-notice">大体积模型尚未就绪：参数可校验和导出，计算入口会在兼容模型就绪后开放。</div>' : ''}<div class="model-layout"><div>${section('研究 · 时域', `<div class="form-grid"><label class="field full"><span>任务类型</span><select id="study-kind"><option value="plate_smoke" ${state.studyKind !== 'plate_reference' ? 'selected' : ''}>平板验证（plate_smoke）</option><option value="plate_reference" ${state.studyKind === 'plate_reference' ? 'selected' : ''}>平板参考计算（plate_reference）</option></select></label>${input('采集时长', 'acquisition.duration_us', 'μs', 5, 150, volume())}${input('采样间隔', 'acquisition.sample_step_us', 'μs', volume() ? .2 : .05, volume() ? 1 : .5)}</div>`, '采集时长必须包含整数个采样间隔，并覆盖完整激励脉冲。')}</div><div>${section('网格与求解器', volume() ? '<dl class="definition"><dt>粗 / 细网格</dt><dd>4 / 3 mm</dd><dt>最大时间步</dt><dd>0.05 μs</dd><dt>相对容差</dt><dd>10⁻⁶</dd><dt>场采样</dt><dd>101 × 41 × 3，151 帧</dd></dl><div class="section-note">大体积模板参数由现有协议固定。</div>' : `<div class="form-grid">${input('粗网格', 'numerics.coarse_h_mm', 'mm', .8, 3)}${input('细网格', 'numerics.fine_h_mm', 'mm', .5, 2)}${input('最大时间步', 'numerics.max_step_us', 'μs', .01, .25)}${input('相对容差', 'numerics.relative_tolerance', '', 1e-7, 1e-4)}${input('场采样 X', 'numerics.field_nx', '点', 5, 61)}${input('场采样 Y', 'numerics.field_ny', '点', 5, 41)}${input('场采样帧数', 'numerics.field_frames', '帧', 5, 101)}</div>`)}</div></div><section class="panel"><div class="panel-heading"><h2>计算任务</h2><button class="button secondary small" data-action="new-job">连接验证</button></div>${jobTable()}<form id="import-form" class="import-form"><input id="import-id" aria-label="已有任务编号" placeholder="输入已有任务编号：comsol-…" required pattern="comsol-[a-f0-9]{32}"><button class="button secondary" type="submit">查询并加入列表</button></form></section>`;
  }
  function markDirty() { state.modelDirty = true; state.modelError = ''; document.querySelectorAll('.model-status').forEach(node => { node.textContent = '参数已修改，尚未保存。'; }); document.querySelectorAll('[data-model-action="run"]').forEach(node => { node.disabled = true; }); }
  document.addEventListener('input', event => {
    const target = event.target;
    if (target.id === 'model-name') { state.model.name = target.value; markDirty(); }
    const path = target.dataset.modelPath;
    if (!path || target.tagName === 'SELECT') return;
    const keys = path.split('.'), leaf = keys.pop();
    const parent = keys.reduce((value, key) => value[key], state.model.case);
    parent[leaf] = target.type === 'checkbox' ? target.checked : target.hasAttribute('data-string') ? target.value : target.hasAttribute('data-array') ? target.value.split(',').filter(x => x.trim()).map(Number) : target.value === '' ? null : Number(target.value);
    if (research() && path.startsWith('material.')) {
      delete state.model.case.material_reference;
      document.querySelectorAll('.material-reference-state').forEach(node => { node.textContent = '当前为自定义材料数值，未关联此文献预设。'; });
      document.querySelectorAll('[data-model-action="apply-aluminum"]').forEach(node => { node.disabled = false; });
    }
    markDirty();
  });
  document.addEventListener('change', event => {
    const target = event.target;
    if (target.id === 'model-template') { state.model.case = structuredClone(state.templates[target.value]); state.studyKind = research() ? 'research_reference' : 'plate_smoke'; markDirty(); render(); }
    if (target.id === 'study-kind') state.studyKind = target.value;
    if (target.tagName === 'SELECT' && target.dataset.modelPath) {
      const keys = target.dataset.modelPath.split('.'), leaf = keys.pop(); keys.reduce((value, key) => value[key], state.model.case)[leaf] = target.value; markDirty();
    }
    // Keep the focused controls in place: replacing the whole form on blur
    // can swallow the user's following click on Save. Refresh only the view.
    if (target.dataset.modelPath) document.querySelectorAll('.geometry-view').forEach(view => { view.outerHTML = nonlinear() ? nonlinearUI.geometry() : research() ? researchUI.geometry() : geometry(); });
  });
  async function save() {
    if (state.savingModel) return;
    state.savingModel = true; state.modelError = ''; render();
    try {
      state.model = await api('/api/model', { name: state.model.name, case: state.model.case, expected_revision: state.model.revision });
      state.modelDirty = false; toast('模型参数已校验并保存。');
    } catch (error) { state.modelError = error.message; toast(error.message); }
    finally { state.savingModel = false; render(); }
  }
  document.addEventListener('click', event => {
    const button = event.target.closest('button'); if (!button || button.disabled) return;
    const action = button.dataset.modelAction;
    if (action === 'apply-aluminum' && research() && state.materials?.aluminum) {
      state.model.case.material = structuredClone(state.materials.aluminum.linear);
      state.model.case.material_reference = structuredClone(state.materials.aluminum.reference);
      if (nonlinear()) state.model.case.material_reference.usage = 'murnaghan_total_lagrangian';
      markDirty(); render(); toast('已应用文献铝参数，几何尺寸保持当前设置；请校验并保存。');
    }
    if (action === 'export-material' && state.materials?.aluminum) {
      const blob = new Blob([JSON.stringify(state.materials.aluminum, null, 2) + '\n'], { type: 'application/json' });
      const url = URL.createObjectURL(blob), link = document.createElement('a'); link.href = url; link.download = 'workai-aluminum-material-reference.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 30000);
    }
    if (action === 'save') void save();
    if (action === 'run') {
      document.querySelector('#study-description').textContent = nonlinear() ? `${state.model.name} · 材料非线性五组对照 × 三组离散精度，共 15 次真实 COMSOL 求解。` : research() ? `${state.model.name} · 二维 ${state.model.case.geometry.length_mm} × ${state.model.case.geometry.thickness_mm} mm · 线性参考：粗网格、细网格和半时间步三组求解。` : `${state.model.name} · ${state.model.case.geometry.length_mm} × ${state.model.case.geometry.width_mm} × ${state.model.case.geometry.thickness_mm} mm · ${state.studyKind === 'plate_reference' ? '平板参考计算' : '平板验证'}。`;
      document.querySelector('#study-dialog').showModal();
    }
    if (action === 'export') {
      const blob = new Blob([JSON.stringify(state.model.case, null, 2) + '\n'], { type: 'application/json' });
      const url = URL.createObjectURL(blob), link = document.createElement('a'); link.href = url; link.download = 'workai-model-case.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 30000);
    }
    if (button.dataset.modelAdd) {
      const group = button.dataset.modelAdd, points = state.model.case[group], max = group === 'receivers' ? 8 : volume() ? 1 : 3;
      if (points.length >= max) return;
      const point = structuredClone(points[0]); let index = points.length;
      while (points.some(p => p.id === `${group === 'sources' ? 'tx' : 'rx'}${index}`)) index++;
      point.id = `${group === 'sources' ? 'tx' : 'rx'}${index}`; points.push(point); markDirty(); render();
    }
    if (button.dataset.modelRemove) {
      const [group, index] = button.dataset.modelRemove.split('.'); if (state.model.case[group].length <= 1) return;
      state.model.case[group].splice(Number(index), 1); markDirty(); render();
    }
  });
  document.querySelector('#submit-study').addEventListener('click', () => void submitStudy());
  return { renderModel, renderPhysics, renderStudy };
}
