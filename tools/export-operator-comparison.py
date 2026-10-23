"""Self-contained experimental AI/FEM probe comparison from frozen run records."""
import argparse,hashlib,html,json,re,secrets
from pathlib import Path
import numpy as np

def main(args):
    run=args.run;repo=Path(__file__).resolve().parents[1];metrics=json.loads((run/'metrics.json').read_text());is_spectral=metrics.get('schema')=='workai.spectral_operator_metrics.v1';frozen=json.loads((run/('checkpoint-freeze.json' if is_spectral else 'frozen-checkpoint-before-test.json')).read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(run/'best.pt')==metrics['checkpoint_sha256']==frozen['checkpoint_sha256' if is_spectral else 'sha256'] and metrics.get('final_test_evaluated',True) and not metrics['accepted_for_solver_routing']
    tests=[c for c in metrics['cases'] if c['split']=='test'];assert len(tests)>0;records=[]
    with np.load(run/'probe-predictions.npz',allow_pickle=False) as saved:
        for c in tests:
            times=saved[c['case_id']+'_times_us'];receivers=[]
            for i,p in enumerate(c['probes']):
                prefix=c['case_id']+'_P'+str(i+1);prediction=saved[prefix+'_prediction'];truth=saved[prefix+'_fem'];assert prediction.shape==truth.shape==(len(times),3,3) and np.isfinite(prediction).all() and np.isfinite(truth).all()
                receivers.append({'name':p['name'],'point_mm':p['point_mm'],'direction':p['direction'],'predicted_parts_m':prediction.tolist(),'fem_parts_m':truth.tolist()})
            records.append({'case_id':c['case_id'],'frequency_mhz':c['source']['frequency_mhz'],'times_us':times.tolist(),'receivers':receivers})
    data={'run_id':run.name,'checkpoint_sha256':metrics['checkpoint_sha256'],'probe_archive_sha256':digest(run/'probe-predictions.npz'),'accepted_for_solver_routing':False,'prediction_label':'物理降阶 + 学习耦合' if metrics.get('kind')=='modal_energy' else 'AI（实验）','cases':records}
    source='\n'.join(re.sub(r'^export ','',re.sub(r'^import .*;\n','',(repo/name).read_text(),flags=re.M),flags=re.M) for name in ['public/solid-data.js','public/operator-comparison.js'])
    source+="\nconst record=JSON.parse(document.getElementById('comparison-record').textContent);mountOperatorComparison(document.getElementById('comparison-root'),record);"
    assert '</script' not in source.lower();nonce=secrets.token_hex(18);css=(repo/'public/styles.css').read_text();body=json.dumps(data,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    title='物理降阶预测与有限元对照' if metrics.get('kind')=='modal_energy' else 'AI 与有限元波形对照'
    label_scope='高阶参考场已完成本批网格、阶次和时间步检查。' if is_spectral and metrics['acceptance'].get('reference_h_p_time_and_fields') else '标签仍需逐例离散精度验证。'
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none';script-src 'nonce-{nonce}';style-src 'unsafe-inline';img-src data: blob:;connect-src 'none';base-uri 'none';form-action 'none'"><title>workAI · {title}</title><style>{css}\nbody{{min-width:0}}main{{max-width:1200px;margin:auto}}.actions{{display:flex;gap:12px;flex-wrap:wrap}}.form-grid{{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}}.mono{{overflow-wrap:anywhere}}</style></head><body><main><div class="page-heading"><div><div class="eyebrow">WORKAI · EXPERIMENTAL MODEL</div><h1>{title}</h1><p>未见测试结构 · 三分量投影 · 波形与频谱</p></div></div><div class="notice error-notice"><strong>实验模型尚未通过代理验收</strong><p>可切换 {len(tests)} 个测试结构及各自两个保留点位，输入任意非零接收方向。这里只查看已有测试结果；其他点位需回工作台从完整场提取。{label_scope}未建立任意形状的可靠预测能力。</p></div><p class="mono small-text">运行：{html.escape(run.name)}<br>权重 SHA-256：{metrics['checkpoint_sha256']}</p><div id="comparison-root"></div></main><script id="comparison-record" type="application/json">{body}</script><script type="module" nonce="{nonce}">{source}</script></body></html>'''
    with args.output.open('x') as f:f.write(page)
    print(json.dumps({'file':str(args.output),'bytes':args.output.stat().st_size,'sha256':digest(args.output),'test_cases':len(tests),'new_solves_or_training':0}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
