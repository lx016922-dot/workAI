"""Compute the demonstration models one after another in a running workbench (the same request the 计算 button sends).

    python docs/demo/compute.py http://127.0.0.1:8765 [name ...]

A model whose record already exists (same name, completed) is skipped. Prints the job of every model.
"""
import json, sys, time
from urllib.request import Request, build_opener, ProxyHandler
from models import MODELS

address = sys.argv[1]; opener = build_opener(ProxyHandler({}))
def call(path, value=None, token=None):
    request = Request(address + path, None if value is None else json.dumps(value).encode(), {'Content-Type': 'application/json', **({'X-WorkAI-Token': token} if token else {})})
    with opener.open(request, timeout=600) as response: return json.loads(response.read())

token = call('/api/info')['token']
for name in sys.argv[2:] or MODELS:
    document = MODELS[name]; title = document['scene']['name']; done = [j for j in call('/api/jobs') if j.get('name') == title and j.get('phase') == 'completed']
    if done: print(name, done[0]['job_id'], 'already computed', flush=True); continue
    while any(j.get('phase') not in ('completed', 'failed', 'paused') for j in call('/api/jobs')): time.sleep(5)
    job = call('/api/jobs', dict(document, mesh='auto'), token)['job_id']; started = time.time()
    while True:
        status = call('/api/jobs/' + job)
        if status['phase'] in ('completed', 'failed'): break
        time.sleep(3)
    print(name, job, status['phase'], f'{time.time() - started:.0f} s', status.get('message') or '', flush=True)
