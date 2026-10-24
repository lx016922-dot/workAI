import { analyzeTimeWindow } from './solid-data.js';
import { analyzePointPath, parsePointPath, pointPathCsv } from './solid-path.js';

const pathResponseNames = { full: '完整位移', linear: '线性位移', stvk: 'StVK 位移', geometric: '几何非线性增量', material: '材料非线性增量' };
const pathInputIds = ['solid-path-mode', 'solid-path-count', 'solid-path-points', 'solid-path-delay',
  ...['start', 'end'].flatMap(end => [0, 1, 2].map(axis => `solid-path-${end}-${axis}`))];

export function solidPathMarkup(maxPoints=64) {
  return `<section class="panel" id="solid-path-panel"><div class="panel-heading"><h2>多点与沿程谐波</h2><span class="muted small-text">复用当前响应场</span></div><div class="panel-body">
    <p class="section-note">比较多个实体内接收点的 f / 2f，使用上方的接收方向和分析窗。沿程距离从第一个采样点计起，按点列累计直线段长度；连线不代表实际波的传播路线，仅检查采样点是否在实体内。</p>
    <div class="form-grid"><label class="field"><span>点位输入</span><select id="solid-path-mode"><option value="segment">线段等距采样</option><option value="points">自定义点列</option></select></label><label class="field"><span>每毫米时间窗延迟 / μs·mm⁻¹</span><input id="solid-path-delay" type="number" step="any" value="0"></label></div>
    <div id="solid-path-segment"><div class="form-grid solid-vectors">${['start', 'end'].flatMap((end, i) => ['X', 'Y', 'Z'].map((axis, j) => `<label class="field"><span>${i ? '终点' : '起点'} ${axis} / mm</span><input id="solid-path-${end}-${j}" type="number" step="any"></label>`)).join('')}</div><label class="field"><span>采样点数（2–${maxPoints}）</span><input id="solid-path-count" type="number" min="2" max="${maxPoints}" step="1" value="7"></label></div>
    <div id="solid-path-custom" hidden><label class="field"><span>每行一个点：X, Y, Z / mm（2–${maxPoints} 点）</span><textarea id="solid-path-points" rows="5" spellcheck="false"></textarea></label></div>
    <p class="section-note">延迟为 0 时所有点使用同一时间窗；非零值按沿程距离平移时间窗，需自行指定，不自动假定波型或群速度。超出记录的窗口会拒绝计算。</p>
    <div class="solid-path-actions"><button class="button secondary small" id="solid-path-presets">使用预设接收点列</button><button class="button primary small" id="solid-path-analyze">提取多点波形与谐波</button><button class="button secondary small" id="solid-path-csv" disabled>下载多点 CSV</button></div>
    <p id="solid-path-message" class="section-note" role="status">设置点列后开始提取，无需重新求解。</p>
    <div id="solid-path-analysis" hidden><div class="solid-path-charts"><div><h3>基频幅值 f</h3><div id="solid-path-f"></div></div><div><h3>二次谐波幅值 2f</h3><div id="solid-path-2f"></div></div></div>
      <p class="section-note">相位以整段记录的时间零点为参考，范围 −180° 至 180°，未展开。幅值增长不能单独证明纯模态累积；完整场的 2f 还可能包含短脉冲泄漏。材料增量包含材料与几何的耦合，不由此估计材料 β。</p>
      <div id="solid-path-table" class="table-scroll"></div>
      <label class="field"><span>查看点列中的波形与频谱</span><select id="solid-path-wave-point"></select></label>
      <div id="solid-path-waveform"></div><div id="solid-path-spectrum"></div><p id="solid-path-wave-note" class="section-note"></p>
    </div></div></section>`;
}

export function mountSolidPath(container, result, field, extractSeries, chart, state = null) {
  const pathElement = id => container.querySelector('#' + id);
  let analysis = null, generation = 0, disposed = false;
  const listeners = [];
  const listen = (id, event, callback) => {
    const element = pathElement(id); element.addEventListener(event, callback);
    listeners.push(() => element.removeEventListener(event, callback));
  };
  function invalidate() {
    generation++; analysis = null;
    pathElement('solid-path-csv').disabled = true;
    pathElement('solid-path-analyze').disabled = false;
    pathElement('solid-path-analysis').hidden = true;
    for (const id of ['solid-path-f', 'solid-path-2f', 'solid-path-table', 'solid-path-wave-point', 'solid-path-waveform', 'solid-path-spectrum', 'solid-path-wave-note']) pathElement(id).replaceChildren();
    pathElement('solid-path-message').textContent = '设置已更新，请重新提取多点结果。';
    pathElement('solid-path-message').classList.remove('model-error');
  }
  function mode() {
    pathElement('solid-path-segment').hidden = pathElement('solid-path-mode').value !== 'segment';
    pathElement('solid-path-custom').hidden = pathElement('solid-path-mode').value !== 'points';
  }
  function number(id) {
    const element = pathElement(id);
    if (element.value.trim() === '' || !Number.isFinite(Number(element.value))) throw new Error('请填写完整坐标、方向、时间窗、点数与延迟。');
    return Number(element.value);
  }
  function plot(id, x, y, labelX, labelY, color) {
    const element = pathElement(id), width = Math.max(320, Math.round(element.getBoundingClientRect().width));
    element.innerHTML = chart(x, y, labelX, labelY, color, width);
  }
  function showWave() {
    if (!analysis) return;
    const row = analysis.rows[Number(pathElement('solid-path-wave-point').value)];
    const selected = analyzeTimeWindow(row.projected_m, analysis.times_us, analysis.frequency_mhz, ...row.requested_window_us,analysis.signal);
    plot('solid-path-waveform', analysis.times_us, row.projected_m.map(v => v * 1e9), '时间 / μs', '位移 / nm');
    const spectrum = selected.spectrum, limit = Math.min(spectrum.nyquist_mhz, Math.max(analysis.frequency_mhz * 5, 1));
    const indices = spectrum.frequency_mhz.map((f, i) => f <= limit ? i : -1).filter(i => i >= 0);
    plot('solid-path-spectrum', indices.map(i => spectrum.frequency_mhz[i]), indices.map(i => spectrum.amplitude_m[i] * 1e9), '频率 / MHz', '幅值 / nm', '#5b80b6');
    pathElement('solid-path-wave-note').textContent = `第 ${row.index} 点 [${row.point_mm.join(', ')}] mm · 实际分析窗 ${row.actual_window_us.map(v => v.toFixed(3)).join('–')} μs / ${row.sample_count} 点 · 记录分辨率约 ${spectrum.record_resolution_mhz.toFixed(4)} MHz。波形显示完整记录，频谱使用该点时间窗。`;
  }
  function renderPlots() {
    if (!analysis) return;
    const distance = analysis.rows.map(row => row.distance_mm);
    if(analysis.signal?.kind==='chirp'){for(const [order,id] of [[0,'solid-path-f'],[1,'solid-path-2f']])plot(id,distance,analysis.rows.map(row=>row.bands[order].windowed_rms_m*1e9),'沿程距离 / mm','频带 RMS / nm');showWave();return;}
    for (const [order, id] of [[0, 'solid-path-f'], [1, 'solid-path-2f']]) {
      const available = analysis.rows.every(row => row.harmonics[order].available);
      if (available) plot(id, distance, analysis.rows.map(row => row.harmonics[order].amplitude_m * 1e9), '沿程距离 / mm', '幅值 / nm', order ? '#ad6c36' : '#148778');
      else pathElement(id).innerHTML = '<p>该频率超过 Nyquist，无法提取。</p>';
    }
    showWave();
  }
  function render() {
    const amplitude = h => h?.available ? (h.amplitude_m * 1e9).toExponential(5) : '不可用';
    const phase = h => h?.available && h.phase_deg !== null ? h.phase_deg.toFixed(3) : '—';
    pathElement('solid-path-table').innerHTML = `<table><thead><tr><th>点</th><th>坐标 / mm</th><th>沿程 / mm</th><th>实际窗 / μs</th><th>f / nm</th><th>f 相位 / °</th><th>2f / nm</th><th>2f 相位 / °</th></tr></thead><tbody>${analysis.rows.map(row => `<tr><td>${row.index}</td><td>${row.point_mm.map(v => v.toFixed(4)).join(', ')}</td><td>${row.distance_mm.toFixed(4)}</td><td>${row.actual_window_us.map(v => v.toFixed(3)).join('–')}</td><td>${amplitude(row.harmonics[0])}</td><td>${phase(row.harmonics[0])}</td><td>${amplitude(row.harmonics[1])}</td><td>${phase(row.harmonics[1])}</td></tr>`).join('')}</tbody></table>`;
    if(analysis.signal?.kind==='chirp')pathElement('solid-path-table').innerHTML='<p>扫频不定义固定 f / 2f 系数；上图分别显示激励频带和二倍频带的 Hann 加权 RMS。</p>';
    const selected = state?.selected_index ?? 0;
    pathElement('solid-path-wave-point').innerHTML = analysis.rows.map((row, i) => `<option value="${i}">第 ${row.index} 点 · [${row.point_mm.map(v => v.toFixed(3)).join(', ')}] mm</option>`).join('');
    pathElement('solid-path-wave-point').value = String(Math.min(analysis.rows.length - 1, Math.max(0, selected)));
    pathElement('solid-path-analysis').hidden = false;
    pathElement('solid-path-csv').disabled = false;
    const short = analysis.rows.some(row => row.carrier_cycles < 3);
    const alias = analysis.rows.some(row => row.harmonics.some(h => !h.available));
    pathElement('solid-path-message').textContent = `已提取 ${analysis.rows.length} 点 · 单位方向 [${analysis.direction.map(v => v.toFixed(4)).join(', ')}] · ${pathResponseNames[analysis.response_part] ?? '当前响应'} · 未重新求解。${short ? '部分时间窗不足三个基频周期，幅值易受泄漏影响。' : ''}${alias ? '有谐波超过 Nyquist，表格与 CSV 标记为不可用。' : ''}`;
    renderPlots();
  }
  async function run() {
    invalidate();
    const token = generation;
    pathElement('solid-path-analyze').disabled = true;
    pathElement('solid-path-message').textContent = '正在从当前位移场提取多点…';
    // Allow the busy state to paint. A response switch or edit cancels this read.
    await new Promise(resolve => setTimeout(resolve, 0));
    if (disposed || token !== generation) return;
    try {
      const sampling = pathElement('solid-path-mode').value;
      const options = { mode: sampling, direction: [0, 1, 2].map(j => number('solid-direction-' + j)),
        start_us: number('solid-window-start'), end_us: number('solid-window-end'), delay_us_per_mm: number('solid-path-delay') };
      if (sampling === 'segment') Object.assign(options, { start: [0, 1, 2].map(j => number('solid-path-start-' + j)),
        end: [0, 1, 2].map(j => number('solid-path-end-' + j)), count: number('solid-path-count') });
      else options.points = parsePointPath(pathElement('solid-path-points').value);
      const next = await analyzePointPath(result, field, extractSeries, options);
      if(disposed||token!==generation)return;analysis=next;render();
    } catch (error) {
      if(disposed||token!==generation)return;analysis = null;
      pathElement('solid-path-message').textContent = error.message;
      pathElement('solid-path-message').classList.add('model-error');
    } finally { if (!disposed&&token===generation) pathElement('solid-path-analyze').disabled = false; }
  }
  if(result.recording){pathElement('solid-path-count').max=8;pathElement('solid-path-message').textContent='长记录每批最多 8 点，读取完整采样数据。';}
  if(result.scene.signal?.kind==='chirp'){const headings=container.querySelectorAll('.solid-path-charts h3');headings[0].textContent='激励频带 RMS';headings[1].textContent='二倍频带 RMS';}
  const presets = result.scene.receivers;
  [0, 1, 2].forEach(j => {
    pathElement('solid-path-start-' + j).value = presets[0].point_mm[j];
    pathElement('solid-path-end-' + j).value = presets.at(-1).point_mm[j];
  });
  pathElement('solid-path-points').value = presets.map(p => p.point_mm.join(', ')).join('\n');
  if (state) for (const id of pathInputIds) if (state.inputs?.[id] !== undefined) pathElement(id).value = state.inputs[id];
  mode();
  for (const id of [...pathInputIds, 'solid-direction-0', 'solid-direction-1', 'solid-direction-2', 'solid-window-start', 'solid-window-end']) {
    listen(id, 'input', invalidate); listen(id, 'change', invalidate);
  }
  listen('solid-receiver', 'change', invalidate);
  listen('solid-path-mode', 'change', mode);
  listen('solid-path-presets', 'click', () => {
    invalidate(); pathElement('solid-path-mode').value = 'points';
    pathElement('solid-path-points').value = presets.map(p => p.point_mm.join(', ')).join('\n'); mode();
  });
  listen('solid-path-analyze', 'click', run);
  listen('solid-path-wave-point', 'change', showWave);
  listen('solid-path-csv', 'click', () => {
    if (!analysis) return;
    const link = document.createElement('a'), url = URL.createObjectURL(new Blob([pointPathCsv(analysis)], { type: 'text/csv;charset=utf-8' }));
    link.href = url; link.download = result.job_id + '-' + analysis.response_part + '-path.csv'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  let previousWidth = null;
  const resize = new ResizeObserver(entries => {
    const width = Math.round(entries[0].contentRect.width);
    if (!disposed && width !== previousWidth) { previousWidth = width; renderPlots(); }
  });
  resize.observe(pathElement('solid-path-panel'));
  if (state?.analyzed) run();
  return { snapshot: () => ({ inputs: Object.fromEntries(pathInputIds.map(id => [id, pathElement(id).value])),
    analyzed: analysis !== null, selected_index: Number(pathElement('solid-path-wave-point').value) || 0 }),
    cleanup: () => { disposed = true; generation++; resize.disconnect(); listeners.forEach(remove => remove()); } };
}
