// Material section: reference aluminium, isotropic Murnaghan constants, or a user-defined strain-energy density W(E).
import { escapeHtml } from './inspector-util.js';
import { materialKind } from './scene-model.js';

export const MATERIAL_KINDS = { reference: '多晶铝（物理 AI 训练材料）', murnaghan: '各向同性 Murnaghan（自定义常数）', energy: '自定义应变能密度 W(E)（各向异性，如 CFRP）' };

/** Transversely isotropic stiffness (fibre = axis 1) from engineering constants; returns C11, C12, C22, C23, C55 in the moduli's unit. */
export function tiFromEngineering({ E1, E2, G12, nu12, nu23 }) {
  const S = [[1 / E1, -nu12 / E1, -nu12 / E1], [-nu12 / E1, 1 / E2, -nu23 / E2], [-nu12 / E1, -nu23 / E2, 1 / E2]];
  const [a, b, c] = S, det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
  if (!(Number.isFinite(det) && Math.abs(det) > 0)) throw Error('工程常数无效：柔度矩阵奇异。');
  const inv = (r, s) => { const rows = [0, 1, 2].filter(i => i !== s), cols = [0, 1, 2].filter(j => j !== r); const m = rows.map(i => cols.map(j => S[i][j])); return ((r + s) % 2 ? -1 : 1) * (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det; };
  const round = v => +v.toPrecision(8);
  return { C11: round(inv(0, 0)), C12: round(inv(0, 1)), C22: round(inv(1, 1)), C23: round(inv(1, 2)), C55: round(G12) };
}

const field = (label, path, value, { unit = '', scale = 1, hint = '' } = {}) => `<label class="wb-field"><span>${escapeHtml(label)}${unit ? ` <em>${unit}</em>` : ''}</span>
  <input type="number" step="any" data-path="${path}" data-kind="number" data-scale="${scale}" value="${value === undefined ? '' : +(value / scale).toPrecision(12)}">${hint ? `<small>${escapeHtml(hint)}</small>` : ''}</label>`;

const text = (label, path, value) => `<label class="wb-field wide"><span>${escapeHtml(label)}</span><input type="text" maxlength="60" data-path="${path}" data-kind="text" value="${escapeHtml(value ?? '')}"></label>`;

const VARIABLE_HELP = `<details class="wb-help"><summary>表达式写法</summary><ul>
  <li>W 是单位参考体积的应变能，单位 GPa（1 GPa = 10⁹ J/m³），写在材料坐标系中，轴 1 为纤维方向。</li>
  <li>应变：Green–Lagrange 应变 E = (FᵀF − I)/2 的分量 <code>E11 E22 E33 E12 E13 E23</code>（张量分量，不是工程剪应变）。</li>
  <li>不变量：<code>i1</code> = tr E，<code>i2</code> = ½[(tr E)² − tr E²]，<code>i3</code> = det E，<code>tr2</code> = tr E²，<code>tr3</code> = tr E³，纤维不变量 <code>i4</code> = a·E·a，<code>i5</code> = a·E²·a。</li>
  <li>运算：+ − * / **（或 ^）、括号、exp log sqrt sin cos tanh sinh cosh，以及下表中的参数名。</li>
  <li>W 在零应变处必须无应力（不能有应变的一次项），二阶部分必须正定。</li>
  <li>求解器使用 W 在零应变处的三阶展开，即二阶、三阶弹性常数；下方给出更高阶项在应变 1e-3 时的相对大小。</li></ul></details>`;

export function materialMarkup(scene, presets = {}) {
  const m = scene.material, kind = materialKind(m);
  const presetOptions = Object.entries(presets).map(([k, p]) => `<option value="${k}">${escapeHtml(p.label)}</option>`).join('');
  let body = `<section class="wb-section"><h3>材料模型</h3><div class="wb-grid">
    <label class="wb-field wide"><span>本构类型</span><select data-action="material-model">${Object.entries(MATERIAL_KINDS).map(([k, v]) => `<option value="${k}" ${k === kind ? 'selected' : ''}>${v}</option>`).join('')}</select></label>
    <label class="wb-field wide"><span>从材料库载入</span><select data-action="material-preset"><option value="">选择预设…</option>${presetOptions}</select></label></div></section>`;
  if (kind === 'reference') {
    const rows = [['ρ₀', m.rho_kg_m3, 'kg/m³'], ['λ', m.lambda_pa / 1e9, 'GPa'], ['μ', m.mu_pa / 1e9, 'GPa'], ['l', m.l_pa / 1e9, 'GPa'], ['m', m.m_pa / 1e9, 'GPa'], ['n', m.n_pa / 1e9, 'GPa']];
    body += `<section class="wb-section"><h3>多晶铝（Murnaghan 三阶）</h3><table class="wb-table">${rows.map(([k, v, u]) => `<tr><th>${k}</th><td>${(+v).toPrecision(4)}</td><td>${u}</td></tr>`).join('')}</table>
      <p class="wb-note">物理 AI 网络就是用这种铝训练的；选它时网络的能量层用训练结果。W = ½(λ+2μ)i₁² − 2μi₂ + ⅓(l+2m)i₁³ − 2m·i₁i₂ + n·i₃。</p></section>`;
  } else if (kind === 'murnaghan') {
    body += `<section class="wb-section"><h3>各向同性 Murnaghan 常数</h3><div class="wb-grid">${text('名称', 'material.name', m.name || '')}
      ${field('密度 ρ₀', 'material.rho_kg_m3', m.rho_kg_m3, { unit: 'kg/m³' })}${field('λ', 'material.lambda_pa', m.lambda_pa, { unit: 'GPa', scale: 1e9 })}${field('μ', 'material.mu_pa', m.mu_pa, { unit: 'GPa', scale: 1e9 })}
      ${field('l', 'material.l_pa', m.l_pa, { unit: 'GPa', scale: 1e9 })}${field('m', 'material.m_pa', m.m_pa, { unit: 'GPa', scale: 1e9 })}${field('n', 'material.n_pa', m.n_pa, { unit: 'GPa', scale: 1e9 })}</div>
      <p class="wb-note">只适用于各向同性材料。物理 AI 的能量层改用这组常数，求导与时间层沿用训练结果。</p></section>`;
  } else {
    const params = Object.entries(m.parameters || {});
    body += `<section class="wb-section"><h3>应变能密度 W(E)</h3><div class="wb-grid">
      ${text('材料名称', 'material.name', m.name || '')}
      ${field('密度 ρ₀', 'material.rho_kg_m3', m.rho_kg_m3, { unit: 'kg/m³' })}${field('纤维方向角 θ', 'material.fiber_angle_deg', m.fiber_angle_deg ?? 0, { unit: '°', hint: '绕 z 轴，从 x 轴量起' })}
      <label class="wb-field wide"><span>W(E) <em>GPa</em></span><textarea rows="7" spellcheck="false" maxlength="4000" data-path="material.energy" data-kind="text" class="wb-code">${escapeHtml(m.energy || '')}</textarea></label></div>
      ${VARIABLE_HELP}</section>
      <section class="wb-section"><h3>参数 <em class="wb-unit">GPa（或无量纲）</em></h3><table class="wb-table compact params">${params.map(([k, v]) => `<tr><th><code>${escapeHtml(k)}</code></th>
        <td><input type="number" step="any" data-path="material.parameters.${k}" data-kind="number" data-scale="1" value="${v}"></td><td><button type="button" class="wb-icon" data-action="param-remove" data-name="${escapeHtml(k)}" title="删除参数">×</button></td></tr>`).join('')}
        <tr><td><input type="text" id="param-new-name" placeholder="新参数名" maxlength="30"></td><td><input type="number" step="any" id="param-new-value" placeholder="数值"></td><td><button type="button" class="wb-icon" data-action="param-add" title="添加参数">+</button></td></tr></table>
      <details class="wb-help"><summary>由工程常数计算横观各向同性二阶常数</summary><div class="wb-grid">
        ${['E1', 'E2', 'G12'].map(k => `<label class="wb-field"><span>${k} <em>GPa</em></span><input type="number" step="any" id="eng-${k}" value="${{ E1: 135, E2: 9, G12: 5 }[k]}"></label>`).join('')}
        ${['nu12', 'nu23'].map(k => `<label class="wb-field"><span>${k.replace('nu', 'ν')}</span><input type="number" step="any" id="eng-${k}" value="${{ nu12: .3, nu23: .45 }[k]}"></label>`).join('')}
        <div class="wb-field wide"><button type="button" class="wb-button small" data-action="ti-from-engineering">写入 C11、C12、C22、C23、C55</button><small>纤维为轴 1；C13 = C12，C33 = C22，C44 = (C22 − C23)/2，C66 = C55（模板中已按此书写）。</small></div></div></details></section>`;
  }
  return body + dampingMarkup(scene) + '<div id="damping-derived"></div><div id="material-derived"></div>';
}

/** Rayleigh damping (α, β) from the attenuation of a wave at two frequencies: a in Np/m (or dB/m), c the group velocity in m/s.
 *  The decay rate in time is γ = a·c; γ(ω) = α/2 + βω²/2 through both points. With only the first point the damping is stiffness-proportional. */
export function dampingFromAttenuation({ f1, a1, c1, f2, a2, c2, unit = 'np' }) {
  const neper = unit === 'db' ? Math.log(10) / 20 : 1, ok = v => Number.isFinite(v) && v > 0;
  if (!ok(f1) || !ok(c1) || !(a1 >= 0)) throw Error('需要频率、群速度（正数）和 f 处的衰减（非负）。');
  const g1 = a1 * neper * c1, w1 = (2 * Math.PI * f1 * 1e6) ** 2;
  if (!ok(f2) || !Number.isFinite(a2)) return { mass_per_s: 0, stiffness_s: 2 * g1 / w1 };
  if (!ok(c2) || !(a2 >= 0) || Math.abs(f2 - f1) < 1e-9 * f1) throw Error('第二个衰减点需要不同的频率、群速度（正数）和非负的衰减。');
  const g2 = a2 * neper * c2, w2 = (2 * Math.PI * f2 * 1e6) ** 2, beta = 2 * (g2 - g1) / (w2 - w1), alpha = 2 * g1 - beta * w1, slack = 1e-9 * Math.max(g1, g2);
  if (beta < -slack / Math.max(w1, w2) || alpha < -slack) throw Error('这两个衰减不能由质量比例加刚度比例阻尼得到：频率高的一个的衰减率（衰减 × 群速度）需要在低的一个的 1 倍到（频率比）² 倍之间。');
  return { mass_per_s: Math.max(alpha, 0), stiffness_s: Math.max(beta, 0) };
}

export function dampingMarkup(scene) {
  const d = scene.damping || { mass_per_s: 0, stiffness_s: 0 }, f = scene.source?.frequency_mhz ?? .5, shown = v => +(+v).toPrecision(8);
  return `<section class="wb-section"><h3>材料阻尼 <em class="wb-unit">可选</em></h3><div class="wb-grid">
    <label class="wb-field"><span>质量比例 α <em>1/s</em></span><input type="number" step="any" min="0" id="damping-alpha" value="${shown(d.mass_per_s)}"></label>
    <label class="wb-field"><span>刚度比例 β <em>ns</em></span><input type="number" step="any" min="0" id="damping-beta" value="${shown(d.stiffness_s * 1e9)}"></label></div>
    <p class="wb-note">Rayleigh 阻尼：任何模态在频率 f 处的幅值按 exp(−γt) 衰减，γ = α/2 + β(2πf)²/2；沿传播方向每米衰减 γ/群速度 奈培。两个系数都填 0 即无阻尼。刚度比例项需要自动时间层和冻结网络引擎。</p>
    <details class="wb-help"><summary>由 f 和 2f 处实测的衰减换算</summary><div class="wb-grid">
      <label class="wb-field"><span>频率 f <em>MHz</em></span><input type="number" step="any" id="damping-f1" value="${shown(f)}"></label>
      <label class="wb-field"><span>单位</span><select id="damping-unit"><option value="np">Np/m</option><option value="db">dB/m</option></select></label>
      <label class="wb-field"><span>f 处的衰减</span><input type="number" step="any" min="0" id="damping-a1" placeholder="例如 5"></label>
      <label class="wb-field"><span>f 处模态的群速度 <em>m/s</em></span><input type="number" step="any" id="damping-c1" placeholder="例如 5400"></label>
      <label class="wb-field"><span>2f 处的衰减</span><input type="number" step="any" min="0" id="damping-a2" placeholder="留空：只用刚度比例"></label>
      <label class="wb-field"><span>2f 处模态的群速度 <em>m/s</em></span><input type="number" step="any" id="damping-c2"></label>
      <div class="wb-field wide"><button type="button" class="wb-button small" data-action="damping-from-attenuation">写入 α、β</button>
        <small>衰减和群速度取实际测量的那个模态（群速度可在结果页“频散与 f–k 谱”里查）。两点决定两个系数；其他频率处的衰减按 γ = α/2 + βω²/2 变化，这是模型，不是材料规律。</small></div></div></details></section>`;
}

/** What the damping does at f and 2f, from the server preview (offline/preview.py damping_summary). */
export function dampingDerivedMarkup(d) {
  if (!d) return '';
  const row = r => `<tr><th>${fmt(r.frequency_mhz)} MHz</th><td>${fmt(r.decay_per_s)}</td><td>${fmt(r.loss_factor, 3)}</td><td>${r.q ? fmt(r.q, 3) : '—'}</td><td>${fmt(r.longitudinal_np_m, 3)}（${fmt(r.longitudinal_db_m, 3)} dB/m）</td><td>${fmt(r.shear_np_m, 3)}（${fmt(r.shear_db_m, 3)} dB/m）</td></tr>`;
  return `<section class="wb-section"><h3>阻尼在 f 和 2f 处的效果</h3>
    <table class="wb-table"><tr><th>频率</th><th>衰减率 γ (1/s)</th><th>损耗因子</th><th>Q</th><th>纵波 Np/m</th><th>横波 Np/m</th></tr>${d.at.map(row).join('')}</table>
    <p class="wb-note">体波的沿程衰减供参考；导波各模态的衰减是 γ 除以它自己的群速度，算完后在结果页“频散与 f–k 谱”里列出。</p></section>`;
}

const fmt = (v, digits = 4) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(3) : (+v).toPrecision(digits));

/** Derived constants from the server preview (physics.materials.describe). */
export function materialDerivedMarkup(d) {
  if (!d) return '<section class="wb-section"><p class="wb-note">正在计算材料常数…</p></section>';
  if (!d.ok) return `<section class="wb-section wb-bad-box"><h3>材料有误</h3><p>${escapeHtml(d.error)}</p></section>`;
  const labels = ['1', '2', '3', '4', '5', '6'], c2 = d.voigt_c2_gpa;
  const speeds = [['纤维方向', d.speeds_m_s.fiber], ['面内横向', d.speeds_m_s.in_plane_transverse], ['厚度方向', d.speeds_m_s.thickness]];
  const c3 = d.c3_nonzero_gpa || [], t = d.truncation;
  const names = ['E11 = 1e-3', 'E12 = 1e-3', '一般应变'];
  return `<section class="wb-section"><h3>由 W 得到的常数</h3>
    <p class="wb-note ok-line">✓ 零应变无应力 · 二阶刚度正定（最小特征值 ${fmt(d.eigenvalues_gpa[0])} GPa）</p>
    <div class="wb-sub">二阶刚度 C<sub>IJ</sub>（Voigt，GPa）</div>
    <table class="wb-table matrix"><tr><th></th>${labels.map(l => `<th>${l}</th>`).join('')}</tr>${c2.map((row, i) => `<tr><th>${labels[i]}</th>${row.map(v => `<td class="${Math.abs(v) < 1e-9 ? 'zero' : ''}">${Math.abs(v) < 1e-9 ? '0' : fmt(v)}</td>`).join('')}</tr>`).join('')}</table>
    <div class="wb-sub">三阶常数 C<sub>IJK</sub>（Brugger，GPa）</div>
    ${c3.length ? `<div class="wb-chips">${c3.slice(0, 40).map(r => `<span><b>C${r.index}</b> ${fmt(r.value)}</span>`).join('')}${c3.length > 40 ? `<span>…共 ${c3.length} 个</span>` : ''}</div>` : '<p class="wb-note warn-line">全部为 0：材料非线性只来自几何非线性（相当于各向异性 StVK）；需要实测或文献的三阶常数才能算材料非线性。</p>'}
    ${t ? `<p class="wb-note">三阶展开的余项：应变 1e-3 时为${t.relative_to === 'cubic' ? '三阶项' : '二阶项'}的 ${fmt(t.remainder_relative, 2)} 倍${t.remainder_relative > .01 ? '（偏大，请检查 W 的高阶项）' : ''}。</p>` : ''}
    <div class="wb-sub">相速度（m/s）</div>
    <table class="wb-table"><tr><th>方向</th><th>准纵波</th><th>准剪切 1</th><th>准剪切 2</th></tr>${speeds.map(([k, v]) => `<tr><th>${k}</th>${v.map(x => `<td>${Math.round(x)}</td>`).join('')}</tr>`).join('')}</table>
    <p class="wb-note">全方向最快 ${Math.round(d.fastest_m_s)} m/s（决定时间步），最慢剪切 ${Math.round(d.slowest_shear_m_s)} m/s（决定网格与 2f 分辨率）。</p>
    <div class="wb-sub">本构自检应力 S（MPa，全局坐标）——用来核对 COMSOL 等软件中的同一 W</div>
    ${d.check_stress_mpa.map((s, k) => `<details class="wb-help"><summary>${names[k]}</summary><table class="wb-table matrix">${s.map(row => `<tr>${row.map(v => `<td>${(+v).toFixed(4)}</td>`).join('')}</tr>`).join('')}</table></details>`).join('')}
    <p class="wb-note">${escapeHtml(d.engine || '')}</p></section>`;
}

/** Material-specific actions; generic inputs are bound by the inspector. */
export function bindMaterial(root, scene, presets, onChange) {
  const set = material => { scene.material = structuredClone(material); onChange('structure'); };
  const damp = (alpha, beta) => {
    if (!(alpha >= 0) || !(beta >= 0)) return onChange({ error: Error('阻尼系数需要非负数值。') });
    if (alpha > 0 || beta > 0) scene.damping = { mass_per_s: alpha, stiffness_s: beta }; else delete scene.damping;
    onChange('structure');
  };
  const number = id => { const el = root.querySelector(id); return el && el.value !== '' ? Number(el.value) : NaN; };
  for (const id of ['#damping-alpha', '#damping-beta']) root.querySelector(id)?.addEventListener('change', () => damp(number('#damping-alpha') || 0, (number('#damping-beta') || 0) * 1e-9));
  root.querySelector('[data-action="damping-from-attenuation"]')?.addEventListener('click', () => {
    try {
      const f1 = number('#damping-f1'), d = dampingFromAttenuation({ f1, a1: number('#damping-a1'), c1: number('#damping-c1'), f2: 2 * f1, a2: number('#damping-a2'), c2: number('#damping-c2'), unit: root.querySelector('#damping-unit').value });
      damp(d.mass_per_s, d.stiffness_s);
    } catch (error) { onChange({ error }); }
  });
  root.querySelector('[data-action="material-model"]')?.addEventListener('change', ev => {
    const kind = ev.target.value, pick = { reference: 'aluminium', murnaghan: 'murnaghan', energy: 'cfrp_ud' }[kind];
    if (presets[pick]) set(presets[pick].material);
  });
  root.querySelector('[data-action="material-preset"]')?.addEventListener('change', ev => { const p = presets[ev.target.value]; if (p) set(p.material); });
  root.querySelectorAll('[data-action="param-remove"]').forEach(b => b.addEventListener('click', () => { delete scene.material.parameters[b.dataset.name]; onChange('structure'); }));
  root.querySelector('[data-action="param-add"]')?.addEventListener('click', () => {
    const name = root.querySelector('#param-new-name').value.trim(), value = Number(root.querySelector('#param-new-value').value);
    try {
      if (!/^[A-Za-z][A-Za-z0-9_]{0,29}$/.test(name)) throw Error('参数名需以字母开头，只含字母、数字和下划线。');
      if (!Number.isFinite(value)) throw Error('参数需要数值。');
      scene.material.parameters ??= {}; if (name in scene.material.parameters) throw Error(`参数 ${name} 已存在。`);
      scene.material.parameters[name] = value; onChange('structure');
    } catch (error) { onChange({ error }); }
  });
  root.querySelector('[data-action="ti-from-engineering"]')?.addEventListener('click', () => {
    try {
      const read = k => { const v = Number(root.querySelector(`#eng-${k}`).value); if (!Number.isFinite(v)) throw Error(`${k} 需要数值。`); return v; };
      const C = tiFromEngineering({ E1: read('E1'), E2: read('E2'), G12: read('G12'), nu12: read('nu12'), nu23: read('nu23') });
      scene.material.parameters = { ...(scene.material.parameters || {}), ...C }; onChange('structure');
    } catch (error) { onChange({ error }); }
  });
}
