"""Read-only inspection of the real training run; never starts training or COMSOL."""
import asyncio,json,os,shutil
from pathlib import Path
from playwright.async_api import async_playwright,expect

async def check():
    output=Path('/workspace/workai-remote/validation/20261022-training');output.mkdir(parents=True,exist_ok=True)
    origin=os.environ.get('WORKAI_TEST_ORIGIN','http://127.0.0.1:3000')
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=shutil.which('chromium'),args=['--no-sandbox','--disable-dev-shm-usage'])
        page=await browser.new_page(viewport={'width':1440,'height':1000});errors=[];posts=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:posts.append(r.url) if r.method=='POST' else None)
        await page.goto(origin+'/#training')
        await expect(page.get_by_role('heading',name='已加入的物理约束',exact=True)).to_be_visible(timeout=30000)
        await expect(page.get_by_role('heading',name='40 mm 波形验证',exact=True)).to_be_visible()
        await expect(page.locator('svg[aria-label="物理约束训练损失曲线"] polyline')).to_have_count(4)
        state=await (await page.request.get(origin+'/api/training')).json()
        assert state['available'] and state['step']>0 and state['physical_case_count']==1
        assert state['accepted_for_solver_routing'] is False
        assert 'pid' not in state and 'run_dir' not in state
        await expect(page.locator('#main')).to_contain_text('尚未作为求解器使用')
        if state['status']=='completed_experimental':
            await expect(page.get_by_role('heading',name='材料二次谐波对照',exact=True)).to_be_visible()
            if any(v is False for v in state['metrics']['acceptance'].values()):
                await expect(page.get_by_role('status')).to_contain_text('本轮模型未通过验收')
            assert state['metrics']['selected_step'] <= state['step']
        await page.screenshot(path=str(output/'training-desktop.png'),full_page=True)
        await page.set_viewport_size({'width':390,'height':844})
        assert await page.evaluate('() => document.documentElement.scrollWidth <= innerWidth')
        await page.screenshot(path=str(output/'training-mobile.png'),full_page=True)
        assert not posts and not errors,(posts,errors)
        result={'passed':True,'run_id':state['run_id'],'observed_step':state['step'],'observed_status':state['status'],'mobile_overflow':False,'javascript_errors':errors,'post_requests':posts,'solver_routing_disabled':True}
        (output/'browser-acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
        await browser.close()

asyncio.run(check())
