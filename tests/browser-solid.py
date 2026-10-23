"""Read-only live and offline 3-D result interaction, without launching solves."""
import asyncio,json,shutil
from pathlib import Path
from playwright.async_api import async_playwright,expect

ROOT=Path('/workspace/workai-remote');ORIGIN='http://127.0.0.1:3000'
CASES=json.loads((ROOT/'validation/20261022-solid/final-cases.json').read_text())['cases']
EXPORTS={v['kind']:Path(v['output']) for v in json.loads((ROOT/'validation/20261022-solid/final-offline-exports.json').read_text())}
JOB=CASES[0]['display_job']

async def interact(page,point=('4','1.5','1'),void=('1000','1.5','1')):
    await expect(page.locator('#solid-waveform svg')).to_be_visible(timeout=30000)
    initial=await page.locator('#solid-canvas').evaluate('(c)=>c.toDataURL()')
    await page.locator('#solid-frame').evaluate('(e)=>{e.value=80;e.dispatchEvent(new Event("input",{bubbles:true}));}')
    assert initial!=await page.locator('#solid-canvas').evaluate('(c)=>c.toDataURL()')
    await page.locator('#solid-play').click();before=await page.locator('#solid-frame').input_value();await page.wait_for_timeout(250)
    assert before!=await page.locator('#solid-frame').input_value();await page.locator('#solid-play').click()
    for i,v in enumerate(point):await page.locator(f'#solid-point-{i}').fill(v)
    for i,v in enumerate(['1','2','-1']):await page.locator(f'#solid-direction-{i}').fill(v)
    await page.locator('#solid-extract').click();await expect(page.locator('#solid-probe-message')).to_contain_text('未重新求解')
    await expect(page.locator('#solid-spectrum svg')).to_be_visible()
    waveform=await page.locator('#solid-waveform polyline').get_attribute('points')
    for i,v in enumerate(void):await page.locator(f'#solid-point-{i}').fill(v)
    await page.locator('#solid-extract').click()
    await expect(page.locator('#solid-probe-message')).to_contain_text('实体之外')
    await expect(page.locator('#solid-waveform polyline')).to_have_count(0)
    await expect(page.locator('#solid-csv')).to_be_disabled()
    for i,v in enumerate(point):await page.locator(f'#solid-point-{i}').fill(v)
    await page.locator('#solid-extract').click()
    await page.locator('#solid-frequency-limit').fill('5');await page.locator('#solid-frequency-limit').press('Tab')
    await expect(page.locator('#solid-spectrum-note')).to_contain_text('0–5.00 MHz')
    async with page.expect_download() as download:await page.locator('#solid-csv').click()
    assert (await download.value).suggested_filename.endswith('-point-spectrum.csv')

async def main():
    evidence=ROOT/'validation/20261022-solid';evidence.mkdir(parents=True,exist_ok=True)
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=shutil.which('chromium'),args=['--no-sandbox','--disable-dev-shm-usage'])
        context=await browser.new_context(viewport={'width':1440,'height':1000});page=await context.new_page();errors=[];posts=[]
        page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:posts.append(r.url) if r.method=='POST' else None)
        await page.goto(ORIGIN+'/#solid')
        await expect(page.locator('#solid-job-select')).to_be_visible(timeout=30000)
        for kind in ['pipe','hole','notch','plate']:
            await page.locator('#solid-template').select_option(kind)
            await expect(page.locator('#solid-template')).to_have_value(kind)
        await page.locator('#solid-job-select').select_option(JOB)
        await interact(page)
        await page.locator('#solid-result-container').screenshot(path=str(evidence/'solid-live-desktop.png'))
        await page.set_viewport_size({'width':390,'height':844})
        assert await page.evaluate('()=>document.documentElement.scrollWidth<=innerWidth')
        await page.screenshot(path=str(evidence/'solid-live-mobile.png'),full_page=True)
        assert not posts and not errors,(posts,errors)
        await context.close()
        network=[];checked=[]
        for case in CASES:
            kind=case['kind'];offline=await browser.new_context(offline=True,viewport={'width':1440,'height':1000});test_url=ORIGIN+'/offline-solid-evidence';html=EXPORTS[kind].read_text()
            async def route(request):
                if request.request.url==test_url:await request.fulfill(body=html,content_type='text/html')
                else:await request.abort()
            await offline.route('**/*',route);page=await offline.new_page()
            page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:network.append(r.url) if r.url!=test_url else None)
            await page.goto(test_url)
            await expect(page.locator('#offline-load')).to_contain_text('SHA-256 已核对',timeout=30000)
            await expect(page.get_by_text('所选点对照通过',exact=True)).to_be_visible()
            point=('4','2.5','0') if kind=='pipe' else ('4','1.5','1')
            void={'pipe':('4','0','0'),'hole':('6','2','1'),'notch':('6','2','1.8')}.get(kind,('1000','1.5','1'))
            await interact(page,point,void)
            await page.screenshot(path=str(evidence/f'{kind}-offline-desktop.png'),full_page=True)
            await page.set_viewport_size({'width':390,'height':844});assert await page.evaluate('()=>document.documentElement.scrollWidth<=innerWidth')
            await page.screenshot(path=str(evidence/f'{kind}-offline-mobile.png'),full_page=True)
            checked.append({'kind':kind,'job_id':case['display_job'],'point_mm':point,'rejected_void_mm':void})
            await offline.close()
        assert not network and not errors,(network,errors)
        result={'passed':True,'job_id':JOB,'live_view':True,'offline_cases':checked,'parameter_template_controls':4,'animation_changes_frames':True,'arbitrary_point_and_signed_direction':True,'outside_points_rejected':True,'spectrum_and_csv':True,'mobile_overflow':False,'post_requests':posts,'offline_external_requests':network,'javascript_errors':errors,'offline_exact_html_via_memory_route':True,'direct_file_open_tested':False}
        (evidence/'browser-final-acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));await browser.close()

asyncio.run(main())
