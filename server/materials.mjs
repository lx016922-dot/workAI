import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const bytes = readFileSync(new URL('../data/materials/aluminum-polycrystal-blaschke2017.json', import.meta.url));
const record = JSON.parse(bytes);
const sha256 = createHash('sha256').update(bytes).digest('hex');
const linear = {
  rho_kg_m3: record.rho_kg_m3,
  cl_m_s: Math.sqrt((record.lambda_pa + 2 * record.mu_pa) / record.rho_kg_m3),
  ct_m_s: Math.sqrt(record.mu_pa / record.rho_kg_m3),
};
const reference = { id: record.id, record_sha256: sha256, usage: 'linear_constants_only' };

export function aluminumMaterial() {
  return structuredClone({ record, record_sha256: sha256, linear, reference,
    derived: { young_modulus_pa: record.mu_pa * (3 * record.lambda_pa + 2 * record.mu_pa) / (record.lambda_pa + record.mu_pa),
      poisson_ratio: record.lambda_pa / (2 * (record.lambda_pa + record.mu_pa)),
      landau_pa: { A: record.third_order.n_pa, B: record.third_order.m_pa - record.third_order.n_pa / 2, C: record.third_order.l_pa - record.third_order.m_pa + record.third_order.n_pa / 2 } },
  });
}
export function verifyMaterialReference(ref, material) {
  if (!ref || typeof ref !== 'object' || Array.isArray(ref) || Object.keys(ref).length !== 3 ||
      Object.entries(reference).some(([key, value]) => ref[key] !== value)) throw new Error('材料参考版本或使用范围不匹配。');
  if (Object.entries(linear).some(([key, value]) => typeof material[key] !== 'number' || !Number.isFinite(material[key]) || Math.abs(material[key] - value) > Math.abs(value) * 1e-12)) {
    throw new Error('材料数值已偏离文献预设，请移除参考关联或重新应用预设。');
  }
}
