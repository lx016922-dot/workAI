"""Read-only check of the actual hybrid training status and review report."""
import argparse,asyncio,json,shutil
from pathlib import Path
from playwright.async_api import async_playwright,expect

async def main(args):
    args.evidence.mkdir(parents=True,exist_ok=False);errors=[];posts=[];origin='http://127.0.0.1:3000'
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=shutil.which('chromium'),args=['--no-sandbox','--disable-dev-shm-usage']);context=await browser.new_context(viewport={'width':1440,'height':1050});page=await context.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:posts.append(r.url) if r.method=='POST' else None)
        await page.goto(origin+'/#training');await expect(page.get_by_role('heading',name='物理降阶与能量耦合训练',exact=True)).to_be_visible(timeout=30000);data=await (await page.request.get(origin+'/api/training')).json();assert data['kind']=='modal_energy' and data['step']==1000 and data['run_id']==args.run and not data['accepted_for_solver_routing'];assert data['split_counts']=={'train':48,'validation':8,'test':0};assert data['evaluation_split_counts']=={'train':48,'validation':8,'test':8};assert len(data['reference_checks']['cases'])==8;assert not data['metrics']['acceptance']['all_gates_passed'];await expect(page.locator('#modal-acceptance')).to_contain_text('尚未通过完整替代验收');await expect(page.get_by_role('heading',name='有限元参考解的网格与时间步检查',exact=True)).to_be_visible();assert '不是纯神经网络瞬态求解器' in await page.locator('#main').inner_text();await page.screenshot(path=str(args.evidence/'training-desktop.png'),full_page=True);await page.set_viewport_size({'width':390,'height':844});assert await page.evaluate('()=>document.documentElement.scrollWidth<=innerWidth');await page.screenshot(path=str(args.evidence/'training-mobile.png'),full_page=True);await context.close()
        context=await browser.new_context(offline=True,viewport={'width':1440,'height':1050});url=origin+'/modal-training-offline';html=args.report.read_text();network=[]
        async def route(request):
            if request.request.url==url:await request.fulfill(body=html,content_type='text/html')
            else:await request.abort()
        await context.route('**/*',route);page=await context.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:network.append(r.url) if r.url!=url else None);await page.goto(url);await expect(page.locator('#modal-acceptance')).to_contain_text('尚未通过完整替代验收');await expect(page.get_by_role('heading',name='有限元参考解的网格与时间步检查',exact=True)).to_be_visible();await page.screenshot(path=str(args.evidence/'training-offline.png'),full_page=True);assert not errors and not posts and not network
        result={'passed':True,'run_id':args.run,'steps':data['step'],'independent_test_cases':8,'reference_checks':8,'hybrid_identity_clear':True,'failed_acceptance_preserved':True,'desktop_mobile_no_overflow':True,'javascript_errors':errors,'post_requests':posts,'offline_external_requests':network};(args.evidence/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));await browser.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);asyncio.run(main(p.parse_args()))
