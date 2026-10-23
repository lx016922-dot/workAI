"""Small, self-contained reference-convergence review with stored waveforms."""
import argparse
import html
import json
import re
import secrets
import sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from spectral3d.compare import spectral,digest


def main(args):
    summary=json.loads(args.summary.read_text());coarse=spectral(args.coarse);fine=spectral(args.fine);cases=[]
    for case_id,b in fine.items():
        a=coarse[case_id];receivers=[]
        for i,rx in enumerate(b['scene']['receivers']):receivers.append(dict(rx,predicted_parts_m=a['parts'][:,i].tolist(),fem_parts_m=b['parts'][:,i].tolist()))
        cases.append({'case_id':case_id,'frequency_mhz':b['scene']['source']['frequency_mhz'],'times_us':b['times'].tolist(),'receivers':receivers})
    data={'run_id':'spectral-reference-20261023','comparison_kind':'reference_convergence','prediction_label':'较粗高阶网格（1.6 mm）','accepted_for_solver_routing':False,'cases':cases,
        'coarse_manifest_sha256':digest(args.coarse),'fine_manifest_sha256':digest(args.fine),'summary_sha256':digest(args.summary)}
    pct=lambda x:f'{100*x:.3f}%'
    rows=''.join(f'<tr><td>{html.escape(r["case_id"])}</td><td>{pct(r["max_h_full_vector"])}</td><td>{pct(r["max_h_material_vector"])}</td><td>{pct(r["max_p_material_vector"])}</td><td>{pct(r["max_time_material_vector"])}</td><td>{pct(r["field_h_material_vector"])}</td><td>{"通过" if r["all_component_checks_passed"] else "未通过"}</td></tr>' for r in summary['cases'])
    source='\n'.join(re.sub(r'^export ','',re.sub(r'^import .*;\n','',(REPO/name).read_text(),flags=re.M),flags=re.M) for name in ['public/solid-data.js','public/operator-comparison.js'])
    source+="\nconst record=JSON.parse(document.getElementById('comparison-record').textContent);mountOperatorComparison(document.getElementById('comparison-root'),record);"
    assert '</script' not in source.lower();nonce=secrets.token_hex(18);payload=json.dumps(data,separators=(',',':'),ensure_ascii=False,allow_nan=False).replace('<','\\u003c');css=(REPO/'public/styles.css').read_text()
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none';script-src 'nonce-{nonce}';style-src 'unsafe-inline';img-src data: blob:;connect-src 'none';base-uri 'none';form-action 'none'"><title>workAI · 高阶参考解检查</title><style>{css}\nbody{{min-width:0}}main{{max-width:1250px;margin:auto}}.form-grid{{grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}}.actions{{display:flex;gap:12px;flex-wrap:wrap}}</style></head><body><main><div class="page-heading"><div><div class="eyebrow">WORKAI · REFERENCE QUALITY</div><h1>铝板与缺口的高阶参考解</h1><p>{summary['completed_solves']} 次真实三维计算 · 完整 Murnaghan 超弹性 · 未更新 AI 权重</p></div></div><div class="notice"><strong>先改善训练标签的离散精度</strong><p>原一阶四面体缺口标签在网格加密后仍变化约 30%–47%。新的独立高阶方法复核相同的三个已用工况，保持材料、载荷和采样不变。当前结果支持这三个工况的有限检查，不代表通用 AI 已完成训练。</p></div><section class="panel"><div class="panel-heading"><h2>网格、阶数与时间步</h2></div><div class="table-scroll"><table><thead><tr><th>工况</th><th>网格：位移向量</th><th>网格：材料增量</th><th>阶数：材料增量</th><th>时间步：材料增量</th><th>全场网格：材料增量</th><th>本次检查</th></tr></thead><tbody>{rows}</tbody></table></div><div class="panel-footer">网格：四阶单元 1.6 → 1.0 mm；阶数：1.0 mm 网格四阶 → 五阶；时间：同四阶细网格 CFL 0.30 → 0.15。表内为保留点向量最大差异，全场列使用细网格质量加权 RMS。另检查 X/Y/Z、指定方向和材料 2f 复系数。门槛 5%，绝对信号报告下限 0.0001 nm；低于下限的分量只检查绝对变化，不宣称相对精度。</div></section><section class="panel"><div class="panel-body"><p>当前高阶实现覆盖板和贯穿板宽的缺口；管、孔及任意 CAD 尚需对应的高阶网格与验证。三个工况都已被分析使用，归入开发数据。全场网格检查是离散 RMS，不是每个空间点或任意方向的误差上界；未完成 COMSOL 对标。</p><p>下方可切换工况、点位、响应部分和任意非零方向，查看两种网格的真实波形与频谱。材料增量为 Murnaghan − StVK，保留几何非线性对照。页面只读取已保存数据。</p></div></section><div id="comparison-root"></div></main><script type="application/json" id="comparison-record">{payload}</script><script type="module" nonce="{nonce}">{source}</script></body></html>'''
    with args.output.open('x') as f:f.write(page)
    print(json.dumps({'path':str(args.output),'bytes':args.output.stat().st_size,'sha256':digest(args.output)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--coarse',type=Path,required=True);p.add_argument('--fine',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
