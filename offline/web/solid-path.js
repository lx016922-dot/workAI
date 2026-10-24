import { analyzeTimeWindow, unitVector } from './solid-data.js';

// A point sequence is a sampling arrangement, not a computed ray or a mode.
export function samplePointPath({ mode, start, end, count, points }) {
  const finitePoint = p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite);
  let sampled;
  if (mode === 'segment') {
    if (!finitePoint(start) || !finitePoint(end)) throw new Error('起点和终点都需要三个有限坐标（mm）。');
    if (!Number.isInteger(count) || count < 2 || count > 64) throw new Error('采样点数必须是 2–64 的整数。');
    sampled = Array.from({ length: count }, (_, i) => start.map((v, j) => v + (end[j] - v) * i / (count - 1)));
  } else if (mode === 'points') {
    sampled = points;
  } else throw new Error('请选择线段采样或自定义点列。');
  if (!Array.isArray(sampled) || sampled.length < 2 || sampled.length > 64 || !sampled.every(finitePoint)) {
    throw new Error('点列需要 2–64 个有效坐标，每行三个有限数值（mm）。');
  }
  let distance = 0;
  return sampled.map((point, index) => {
    if (index) {
      const step = Math.hypot(...point.map((v, j) => v - sampled[index - 1][j]));
      if (!Number.isFinite(step) || step < 1e-9) throw new Error(`第 ${index + 1} 点与前一点重合，或间距无效。`);
      distance += step;
      if (!Number.isFinite(distance)) throw new Error('点列总长度无效。');
    }
    return { index: index + 1, point_mm: [...point], distance_mm: distance };
  });
}

export function parsePointPath(text) {
  if (typeof text !== 'string' || text.length > 16000) throw new Error('点列文本过长。');
  return text.trim().split(/\r?\n/).filter(line => line.trim()).map((line, i) => {
    const columns = line.trim().split(/[\s,，]+/);
    const point = columns.map(Number);
    if (columns.length !== 3 || !point.every(Number.isFinite)) throw new Error(`第 ${i + 1} 行需要三个数值，以空格或逗号分隔。`);
    return point;
  });
}

export async function analyzePointPath(result, field, extractSeries, options) {
  const points = samplePointPath(options), direction = unitVector(options.direction);
  const { start_us, end_us, delay_us_per_mm = 0 } = options;
  if (![start_us, end_us, delay_us_per_mm].every(Number.isFinite)) throw new Error('时间窗和沿程延迟必须是有限数值。');
  // Validate the base window even when later shifted windows would be valid.
  if (!(start_us < end_us) || start_us < result.times_us[0] || end_us > result.times_us.at(-1)) throw new Error('首点时间窗必须位于记录内，且起点早于终点。');
  if(result.recording&&points.length>8)throw new Error('长记录每次最多分析 8 个点，请缩短点列后分批提取。');
  const extracted=extractSeries.batch?await extractSeries.batch(points.map(p=>p.point_mm),direction):await Promise.all(points.map(p=>extractSeries(result,field,p.point_mm,direction)));
  const times=extracted[0].times_us??result.times_us;
  const rows = points.map((point,index) => {
    try {
      const shift = point.distance_mm * delay_us_per_mm;
      const requested_window_us = [start_us + shift, end_us + shift];
      const probe = extracted[index];
      const analysis = analyzeTimeWindow(probe.projected_m, times, result.scene.source.frequency_mhz, ...requested_window_us,result.scene.signal);
      return { ...point, requested_window_us, actual_window_us: [analysis.start_us, analysis.end_us],
        sample_count: analysis.sample_count, carrier_cycles: analysis.carrier_cycles,
        harmonics: analysis.harmonics, bands: analysis.bands, xyz_m: probe.xyz_m, projected_m: probe.projected_m };
    } catch (error) {
      throw new Error(`第 ${point.index} 点 [${point.point_mm.join(', ')}] mm：${error.message}`);
    }
  });
  return { schema: 'workai.point_path.v1', job_id: result.job_id, response_part: result.response_part ?? 'full',
    field_sha256: result.field.sha256, mesh_sha256: result.mesh_sha256 ?? result.evidence?.mesh_sha256 ?? null,
    frequency_mhz: result.scene.source.frequency_mhz, direction, times_us: times, signal: result.scene.signal,
    base_window_us: [start_us, end_us], delay_us_per_mm, sampling_mode: options.mode,
    distance_definition: 'cumulative_euclidean_distance_from_first_sample_mm',
    phase_reference: 'record_time_zero', phase_unwrapped: false,
    continuous_path_inside_solid_verified: false, pure_mode_accumulation_verified: false,
    material_beta_estimated: false, rows };
}

export function pointPathCsv(analysis) {
  const lines = [
    '# schema,' + analysis.schema, '# job_id,' + analysis.job_id,
    '# response_part,' + analysis.response_part, '# field_sha256,' + analysis.field_sha256,
    '# mesh_sha256,' + (analysis.mesh_sha256 ?? ''), '# normalized_direction,' + analysis.direction.join(','),
    '# base_window_us,' + analysis.base_window_us.join(','), '# delay_us_per_mm,' + analysis.delay_us_per_mm,
    '# distance_definition,' + analysis.distance_definition, '# phase_reference,' + analysis.phase_reference,
    '# phase_unwrapped,false', '# continuous_path_inside_solid_verified,false',
    '# pure_mode_accumulation_verified,false', '# material_beta_estimated,false',
    '# spectrum,'+(analysis.signal?.kind==='chirp'?'Hann weighted mean removed frequency-band power':'Hann weighted mean removed exact-frequency single-sided complex coefficients'),
    '# section,harmonics',
    'point_index,x_mm,y_mm,z_mm,distance_mm,requested_start_us,requested_end_us,actual_start_us,actual_end_us,sample_count,carrier_cycles,order,frequency_mhz,available,real_m,imaginary_m,amplitude_m,phase_deg'
  ];
  for (const row of analysis.rows) for (const h of row.harmonics) {
    lines.push([row.index, ...row.point_mm, row.distance_mm, ...row.requested_window_us,
      ...row.actual_window_us, row.sample_count, row.carrier_cycles, h.order, h.frequency_mhz, h.available,
      h.available ? h.real_m : '', h.available ? h.imaginary_m : '', h.available ? h.amplitude_m : '', h.phase_deg ?? ''].join(','));
  }
  if(analysis.signal?.kind==='chirp'){lines.push('', '# section,chirp_bands','point_index,band_order,start_mhz,end_mhz,hann_weighted_rms_m');for(const row of analysis.rows)for(const b of row.bands)lines.push([row.index,b.order,b.start_mhz,b.end_mhz,b.windowed_rms_m].join(','));}
  lines.push('', '# section,full_record_waveforms', 'point_index,time_us,u_x_m,u_y_m,u_z_m,projected_displacement_m');
  for (const row of analysis.rows) analysis.times_us.forEach((time, i) => {
    lines.push([row.index, time, ...row.xyz_m[i], row.projected_m[i]].join(','));
  });
  return lines.join('\n');
}
