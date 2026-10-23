"""Read-only acceptance of the real 3-D training run and exported report."""
import argparse,asyncio,json,shutil
from pathlib import Path
from playwright.async_api import async_playwright,expect

async def main(args):
    out=Path(args.evidence);out.mkdir(parents=True,exist_ok=False);origin='http://127.0.0.1:3000';errors=[];posts=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=shutil.which('chromium'),args=['--no-sandbox','--disable-dev-shm-usage']);context=await browser.new_context(viewport={'width':1440,'height':1100});page=await context.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:posts.append(r.url) if r.method=='POST' else None)
        await page.goto(origin+'/#training');await expect(page.get_by_role('heading',name='三维跨工况 AI 训练',exact=True)).to_be_visible(timeout=30000);await expect(page.get_by_role('heading',name='物理约束与适用范围',exact=True)).to_be_visible()
        data=await (await page.request.get(origin+'/api/training')).json();assert data['kind']=='mesh_conditioned_3d' and data['run_id']==args.run and data['status']=='completed_experimental' and data['step']==args.steps;assert data['split_counts']=={'train':args.training_cases,'validation':8,'test':8};assert data['accepted_for_solver_routing'] is False
        await expect(page.locator('#operator-acceptance')).to_contain_text('尚未通过代理验收');await expect(page.locator('svg[aria-label="三维跨工况训练损失曲线"] polyline')).to_have_count(4);await expect(page.get_by_role('heading',name='未见几何的完整场测试',exact=True)).to_be_visible()
        assert '40 mm' not in await page.locator('#main').inner_text();await page.screenshot(path=str(out/'training-desktop.png'),full_page=True);await page.set_viewport_size({'width':390,'height':844});assert await page.evaluate('()=>document.documentElement.scrollWidth<=innerWidth');await page.screenshot(path=str(out/'training-mobile.png'),full_page=True);await context.close()
        offline=await browser.new_context(offline=True,viewport={'width':1440,'height':1100});html=Path(args.report).read_text();url=origin+'/training-offline';network=[]
        async def route(request):
            if request.request.url==url:await request.fulfill(body=html,content_type='text/html')
            else:await request.abort()
        await offline.route('**/*',route);page=await offline.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:network.append(r.url) if r.url!=url and not r.url.startswith('data:') else None);await page.goto(url);await expect(page.locator('#operator-acceptance')).to_contain_text('尚未通过代理验收');await expect(page.get_by_role('heading',name='预测与三维有限元对照',exact=True)).to_be_visible();await expect(page.locator('img')).to_be_visible();await page.screenshot(path=str(out/'training-offline.png'),full_page=True)
        assert not errors and not posts and not network,(errors,posts,network);result={'passed':True,'run_id':args.run,'steps':data['step'],'desktop_mobile_layout':True,'offline_report':True,'failed_acceptance_preserved':True,'javascript_errors':errors,'post_requests':posts,'offline_external_requests':network,'direct_windows_file_open_tested':False};(out/'browser-acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));await browser.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--evidence',required=True);p.add_argument('--report',required=True);p.add_argument('--steps',type=int,default=8000);p.add_argument('--training-cases',type=int,default=16);asyncio.run(main(p.parse_args()))
