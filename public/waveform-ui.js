import { renderHarmonics } from './nonlinear-ui.js';
export function createWaveformUI({ state, e }) {
  function render() {
    if (state.loadingWaveforms) return '<section class="panel"><div class="loading">正在校验研究输入、完成回执和三组 A 扫文件…</div></section>';
    if (state.waveformError) return `<div class="notice error-notice" role="alert">${e(state.waveformError)} <button class="button text small" data-action="reload-waveforms">重试读取</button></div>`;
    const data = state.waveforms;
    if (!data) return '';
    const nonlinear = data.constitutive === 'murnaghan_green_lagrange';
    const names = ['线性 A', '几何 A', '材料 + 几何 A', '几何 2A', '材料 + 几何 2A'];
    const fineTraces = data.levels.fine;
    const chosen = fineTraces.find(t => `${t.shot}:${t.rx}` === state.waveTrace) || (nonlinear ? fineTraces.find(t => t.shot === 2 && t.rx === 0) : null) || fineTraces[0];
    const component = ['u_nm', 'v_nm', 'w_nm'].includes(state.waveComponent) ? state.waveComponent : 'v_nm';
    const traces = Object.fromEntries(['coarse', 'fine', 'timecheck'].map(level => [level, data.levels[level].find(t => t.shot === chosen.shot && t.rx === chosen.rx)]));
    const maxTime = chosen.points.at(-1).time_us || 1;
    const amplitude = Math.max(...Object.values(traces).flatMap(t => t.points.map(p => Math.abs(p[component])))) || 1;
    const x = time => 60 + time / maxTime * 500, y = value => 125 - value / amplitude * 94;
    const colors = { coarse: '#148778', fine: '#648fc3', timecheck: '#b67e4e' };
    const grid = [-1, -.5, 0, .5, 1].map(fraction => `<path d="M60 ${y(amplitude * fraction)}H560" stroke="#e7edef"/><text x="51" y="${y(amplitude * fraction) + 4}" text-anchor="end" fill="#82949e" font-size="10">${(amplitude * fraction).toPrecision(3)}</text>`).join('');
    const ticks = [0, .25, .5, .75, 1].map(t => `<text x="${x(t * maxTime)}" y="242" text-anchor="middle" fill="#82949e" font-size="11">${(t * maxTime).toFixed(2)}</text>`).join('');
    const curves = Object.entries(traces).map(([level, trace]) => `<polyline fill="none" stroke="${colors[level]}" stroke-width="1.6" ${level === 'coarse' ? 'stroke-dasharray="4 3"' : ''} points="${trace.points.map(p => `${x(p.time_us)},${y(p[component])}`).join(' ')}"/>`).join('');
    function difference(left, right) {
      let numerator = 0, denominator = 0;
      for (let i = 0; i < traces[right].points.length; i++) { const a = traces[left].points[i][component], b = traces[right].points[i][component]; numerator += (a - b) ** 2; denominator += b ** 2; }
      return denominator === 0 ? '参考分量为零，未定义' : `${(100 * Math.sqrt(numerator / denominator)).toFixed(3)}%`;
    }
    return `${renderHarmonics(data, e)}<section class="panel waveform-panel"><div class="panel-heading"><div><h2>时域 A 扫 · ${nonlinear ? '非线性对照' : '线性导波'}</h2><p>${nonlinear ? '有限应变模型' : '原研究模型'} · 真实 COMSOL 输出 · 位移 / nm</p></div><span class="pill good">文件与模型身份已校验</span></div><div class="panel-body"><div class="toolbar"><label for="trace-select">接收位置</label><select id="trace-select">${fineTraces.map(t => `<option value="${t.shot}:${t.rx}" ${t === chosen ? 'selected' : ''}>${nonlinear ? names[t.shot] : '激励 ' + (t.shot + 1)} · 接收 ${t.rx + 1} · X=${e(t.x_mm)} mm</option>`).join('')}</select><label for="component-select">分量</label><select id="component-select">${[['u_nm', 'u · 沿板'], ['v_nm', 'v · 厚度'], ['w_nm', 'w · 板外']].map(([value, label]) => `<option value="${value}" ${value === component ? 'selected' : ''}>${label}</option>`).join('')}</select></div><div class="legend"><span><i></i>粗网格</span><span class="fine"><i></i>细网格</span><span class="timecheck"><i></i>细网格 + 半时间步</span></div><div class="chart-wrap"><svg viewBox="0 0 590 270" role="img" aria-label="真实位移 A 扫的网格与时间步对比">${grid}${ticks}${curves}<text x="310" y="265" text-anchor="middle" fill="#82949e" font-size="11">时间 / μs</text><text x="60" y="15" fill="#82949e" font-size="10">位移 / nm</text></svg></div><div class="waveform-metrics"><span>粗 / 细网格相对 L2：<strong>${difference('coarse', 'fine')}</strong></span><span>细网格 / 半时间步相对 L2：<strong>${difference('fine', 'timecheck')}</strong></span></div><div class="chart-caption">${nonlinear ? '此图显示所选本构与幅值对照的原始位移；材料谐波增量见上表。' : '此图为未分离模态的线性位移响应。'}差异以当前分量、完整时间窗计算，并不证明收敛或非线性谐波精度。有限脉冲及多频源的频谱成分不能直接判作材料非线性二次谐波。</div></div></section>`;
  }
  return { render };
}
