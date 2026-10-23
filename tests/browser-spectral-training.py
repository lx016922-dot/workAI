"""Read-only live and exported high-order training review; no calculations."""
import argparse,asyncio,json,shutil
from pathlib import Path
from playwright.async_api import async_playwright,expect

async def main(a):
 a.evidence.mkdir(parents=True,exist_ok=False);errors=[];posts=[]
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path=shutil.which('chromium'),args=['--no-sandbox','--disable-dev-shm-usage']);page=await browser.new_page(viewport={'width':1440,'height':1050});page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:posts.append(r.url) if r.method!='GET' else None)
  await page.goto(a.origin+'/#training');await expect(page.get_by_role('heading',name='高阶物理约束 AI 训练',exact=True)).to_be_visible(timeout=30000)
  response=await page.request.get(a.origin+'/api/training');assert response.ok;data=await response.json();assert data['kind']=='spectral_operator' and data['status']=='completed_experimental' and data['step']==8000 and data['accepted_for_solver_routing'] is False
  await expect(page.locator('#spectral-acceptance')).to_contain_text('本轮模型未通过代理验收' if not data['metrics']['acceptance']['all_gates_passed'] else '本批检查通过')
  await expect(page.locator('svg[aria-label="高阶物理约束训练损失"] polyline')).to_have_count(4);await expect(page.get_by_role('heading',name='独立测试几何',exact=True)).to_be_visible()
  for c in [r for r in data['metrics']['cases'] if r['split']=='test']:
   row=page.locator('tr').filter(has=page.get_by_text(c['case_id'],exact=True));await expect(row).to_contain_text(f"{100*c['field_checks']['full_vector']['relative_l2']:.2f}%")
  await page.screenshot(path=str(a.evidence/'live-desktop.png'),full_page=True);await page.set_viewport_size({'width':390,'height':844});assert await page.evaluate('()=>document.documentElement.scrollWidth<=innerWidth');await page.screenshot(path=str(a.evidence/'live-mobile.png'),full_page=True)
  context=await browser.new_context(offline=True,viewport={'width':1440,'height':1000});offline=await context.new_page();network=[];html=a.report.read_text();url='http://127.0.0.1:3000/spectral-training-offline';offline.on('pageerror',lambda e:errors.append(str(e)));offline.on('request',lambda r:network.append(r.url) if r.url!=url else None)
  async def route(r):
   if r.request.url==url:await r.fulfill(body=html,content_type='text/html')
   else:await r.abort()
  await context.route('**/*',route);await offline.goto(url);await expect(offline.get_by_role('heading',name='高阶物理约束 AI 训练',exact=True)).to_be_visible();await expect(offline.locator('#spectral-acceptance')).to_be_visible();await offline.screenshot(path=str(a.evidence/'offline-desktop.png'),full_page=True)
  assert not errors and not posts and not network,(errors,posts,network);result={'passed':True,'run_id':data['run_id'],'step':data['step'],'checkpoint_sha256':data['metrics']['checkpoint_sha256'],'live_metrics_match':True,'offline_requests':network,'post_requests':posts,'javascript_errors':errors,'mobile_no_overflow':True,'solver_routing_disabled':True};(a.evidence/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));await browser.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--origin',default='http://127.0.0.1:3000');p.add_argument('--report',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);asyncio.run(main(p.parse_args()))
