import test from 'node:test';
import assert from 'node:assert/strict';
import { buildOfflineReport } from '../public/offline-report.js';

function fixture() {
  const job_id = 'comsol-' + '8'.repeat(32), source_sha256 = 'a'.repeat(64);
  const trace = { shot: 0, rx: 0, x_mm: 8, points: [{ time_us: 0, u_nm: 0, v_nm: 0, w_nm: 0 }, { time_us: 1, u_nm: 1, v_nm: -2, w_nm: 0 }] };
  return { verified: true, job_id, source_sha256, constitutive: 'linear_isotropic',
    receipt: { job_id, source_sha256, status: 'completed', completed_at_utc: '2026-10-22T07:23:06Z' },
    scene: { geometry: { length_mm: 20, thickness_mm: 1 }, source: { frequency_mhz: .4, amplitude_pa: 5e6, cycles: 5 } },
    levels: Object.fromEntries(['coarse', 'fine', 'timecheck'].map(k => [k, [structuredClone(trace)]])),
    artifacts: [], harmonics: null };
}
function embedded(html) { return JSON.parse(html.match(/<script id="workai-result"[^>]*>([\s\S]*?)<\/script>/)[1]); }

test('offline HTML preserves trace values and receipt while excluding unrelated session/connection fields', () => {
  const data = fixture(); data.connection = { secret: 'SESSION-ONLY-MARKER' }; data.csrf_token = 'CSRF-ONLY-MARKER';
  const html = buildOfflineReport(data, { generatedAt: '2026-10-22T08:00:00Z' }), result = embedded(html);
  assert.deepEqual(result.levels, data.levels); assert.deepEqual(result.receipt, data.receipt);
  assert.ok(!html.includes('SESSION-ONLY-MARKER') && !html.includes('CSRF-ONLY-MARKER'));
  assert.match(html, /connect-src 'none'/); assert.match(html, /本页不重新验签/);
  assert.match(html, /15:23:06/); assert.match(html, /16:00:00/);
});

test('offline data cannot terminate its script or inject HTML through metadata', () => {
  const data = fixture(), hostile = '</script><img src="https://example.test/" onerror="window.pwned=true"><script>';
  data.receipt.note = hostile; data.scene.notes = hostile + '\u2028\u2029';
  const html = buildOfflineReport(data), result = embedded(html);
  assert.equal(result.receipt.note, hostile); assert.equal(result.scene.notes, data.scene.notes);
  assert.ok(!html.includes('<img src="https://example.test/"')); assert.equal((html.match(/<script /g) || []).length, 2);
  assert.throws(() => buildOfflineReport(data, { css: '</style><script>' }));
});

test('offline export rejects unverified, mismatched, partial or nonfinite result sets', () => {
  for (const change of [d => d.verified = false, d => d.receipt.job_id = 'comsol-' + '1'.repeat(32),
    d => d.receipt.source_sha256 = 'f'.repeat(64), d => d.receipt.status = 'running',
    d => d.levels.fine = [], d => d.levels.fine[0].points[1].time_us = 2,
    d => d.levels.fine[0].points[1].u_nm = NaN, d => d.levels.fine[0].shot = '"><img>',
    d => d.levels.fine[0].x_mm = 9]) {
    const data = fixture(); change(data); assert.throws(() => buildOfflineReport(data));
  }
});
