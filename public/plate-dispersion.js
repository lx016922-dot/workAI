// Symmetric Rayleigh-Lamb branch in cT < phase speed < cL, restricted to
// f*d <= 1.2 MHz mm. This predicts an S0 candidate, not modal purity of a trace.
export function s0Dispersion(frequencyHz, thicknessM, cl, ct) {
  if (![frequencyHz, thicknessM, cl, ct].every(x => Number.isFinite(x) && x > 0) || cl <= ct || frequencyHz * thicknessM > 1200) return null;
  const root = f => {
    const omega = 2 * Math.PI * f, half = thicknessM / 2;
    const determinant = k => {
      const alpha = Math.sqrt(k * k - (omega / cl) ** 2), q = Math.sqrt((omega / ct) ** 2 - k * k);
      return ((q * q - k * k) ** 2 * Math.sin(q * half) * Math.cosh(alpha * half) - 4 * k * k * alpha * q * Math.cos(q * half) * Math.sinh(alpha * half)) / (omega / ct) ** 4;
    };
    const low = omega / cl * (1 + 1e-10), high = omega / ct * (1 - 1e-10), roots = [];
    let left = low, fl = determinant(left);
    for (let j = 1; j <= 512; j++) {
      const right = low + (high - low) * j / 512, fr = determinant(right);
      if (fl * fr < 0) {
        let a = left, b = right, fa = fl;
        for (let step = 0; step < 60; step++) { const mid = (a + b) / 2, fm = determinant(mid); if (fa * fm <= 0) b = mid; else { a = mid; fa = fm; } }
        roots.push((a + b) / 2);
      }
      left = right; fl = fr;
    }
    return roots.length === 1 ? roots[0] : null;
  };
  const k = root(frequencyHz), lo = root(frequencyHz * .9999), hi = root(frequencyHz * 1.0001);
  if (!k || !lo || !hi || hi <= lo) return null;
  return { frequency_hz: frequencyHz, wavenumber_rad_m: k, phase_velocity_m_s: 2 * Math.PI * frequencyHz / k,
    group_velocity_m_s: 2 * Math.PI * frequencyHz * .0002 / (hi - lo) };
}
