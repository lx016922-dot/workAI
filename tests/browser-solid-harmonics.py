"""Live/offline nonlinear-study acceptance. --launch starts one five-control solve."""
import argparse,asyncio,json,shutil
from pathlib import Path
from playwright.async_api import async_playwright,expect

ROOT=Path('/workspace/workai-remote');ORIGIN='http://127.0.0.1:3000'
async def main(args):
    evidence=ROOT/'validation/20261022-solid-harmonics-f64';evidence.mkdir(parents=True,exist_ok=True)
    errors=[];posts=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=shutil.which('chromium'),args=['--no-sandbox','--disable-dev-shm-usage'])
        context=await browser.new_context(viewport={'width':1440,'height':1100});page=await context.new_page()
        page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:posts.append(r.url) if r.method=='POST' else None)
        await page.goto(ORIGIN+'/#solid');await expect(page.locator('#solid-study-select')).to_be_visible(timeout=30000)
        if args.launch:
            await page.locator('#solid-template').select_option('hole')
            async with page.expect_response(lambda r:r.url==ORIGIN+'/api/solid/studies' and r.request.method=='POST') as reply:
                await page.locator('#solid-study-launch').click()
            response=await reply.value;data=await response.json();assert response.status==202,data;study=data['study_id']
        else:
            study=args.study;await page.locator('#solid-study-select').select_option(study)
        await expect(page.locator('#solid-study-probe')).to_be_visible(timeout=240000)
        await page.locator('#solid-study-probe').click();await expect(page.locator('#harmonic-waveform svg')).to_be_visible(timeout=60000)
        if args.expect_reference:await expect(page.locator('#harmonic-reference-status')).to_contain_text('本窗对照通过')
        for i,value in enumerate(['1','2','-1']):await page.locator(f'#harmonic-direction-{i}').fill(value)
        await page.locator('#harmonic-start').fill('1');await page.locator('#harmonic-end').fill('12');await page.locator('#harmonic-analyze').click()
        await expect(page.locator('#harmonic-metrics table')).to_be_visible();await expect(page.locator('#harmonic-spectrum svg')).to_be_visible()
        if args.expect_reference:await expect(page.locator('#harmonic-reference-status')).to_contain_text('尚无细化对照')
        await page.locator('#harmonic-end').fill('100');await page.locator('#harmonic-analyze').click();await expect(page.locator('#harmonic-message')).to_contain_text('已有记录内')
        assert await page.locator('#harmonic-waveform svg').count()==0
        await page.locator('#harmonic-end').fill('12');await page.locator('#harmonic-analyze').click()
        async with page.expect_download() as download:await page.locator('#harmonic-csv').click()
        csv=await (await download.value).path();text=Path(csv).read_text();assert 'material_A_m' in text and '# analysis_window_us,1,12' in text
        async with page.expect_download() as download:await page.locator('#solid-harmonic-export').click()
        item=await download.value;export=evidence/(('browser-launched' if args.launch else 'browser-readonly')+'-export.html');await item.save_as(export)
        await page.locator('#solid-harmonic-result').screenshot(path=str(evidence/'harmonics-live-desktop.png'))
        await page.set_viewport_size({'width':390,'height':844});assert await page.evaluate('()=>document.documentElement.scrollWidth<=innerWidth')
        await page.screenshot(path=str(evidence/'harmonics-live-mobile.png'),full_page=True)
        for i,value in enumerate(['6','2','1'] if args.launch else ['1000','2','1']):await page.locator(f'#solid-study-point-{i}').fill(value)
        await page.locator('#solid-study-probe').click();await expect(page.locator('.error-notice').filter(has_text='实体之外')).to_be_visible(timeout=60000)
        assert await page.locator('#solid-harmonic-result').count()==0
        requests=[url for url in posts if url==ORIGIN+'/api/solid/studies'];assert len(requests)==(1 if args.launch else 0)
        await context.close()
        offline=await browser.new_context(offline=True,viewport={'width':1440,'height':1100});html=export.read_text();url=ORIGIN+'/offline-harmonic-evidence';network=[]
        async def route(request):
            if request.request.url==url:await request.fulfill(body=html,content_type='text/html')
            else:await request.abort()
        await offline.route('**/*',route);page=await offline.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:network.append(r.url) if r.url!=url else None)
        await page.goto(url);await expect(page.locator('#harmonic-waveform svg')).to_be_visible(timeout=30000)
        await expect(page.locator('#harmonic-start')).to_have_value('1');await expect(page.locator('#harmonic-end')).to_have_value('12')
        a=await page.locator('#harmonic-waveform polyline').get_attribute('points');await page.locator('#harmonic-signal').select_option('linear');assert a!=await page.locator('#harmonic-waveform polyline').get_attribute('points')
        for i,value in enumerate(['0','0','0']):await page.locator(f'#harmonic-direction-{i}').fill(value)
        await page.locator('#harmonic-analyze').click();await expect(page.locator('#harmonic-message')).to_contain_text('零向量')
        await page.locator('#harmonic-direction-0').fill('1');await page.locator('#harmonic-analyze').click();await expect(page.locator('#harmonic-waveform svg')).to_be_visible()
        if args.expect_reference:
            await expect(page.locator('#harmonic-reference-status')).to_contain_text('尚无细化对照')
            await page.locator('#harmonic-start').fill('0');await page.locator('#harmonic-end').fill('14');await page.locator('#harmonic-analyze').click()
            await expect(page.locator('#harmonic-reference-status')).to_contain_text('本窗对照通过')
        await page.screenshot(path=str(evidence/'harmonics-offline-desktop.png'),full_page=True)
        await page.set_viewport_size({'width':390,'height':844});assert await page.evaluate('()=>document.documentElement.scrollWidth<=innerWidth')
        assert not errors and not network,(errors,network)
        if args.field_html:
            await offline.close();field_context=await browser.new_context(offline=True,viewport={'width':1440,'height':1100});field_url=ORIGIN+'/offline-float64-field';field_html=Path(args.field_html).read_text()
            async def field_route(request):
                if request.request.url==field_url:await request.fulfill(body=field_html,content_type='text/html')
                else:await request.abort()
            await field_context.route('**/*',field_route);field_page=await field_context.new_page();field_page.on('pageerror',lambda e:errors.append(str(e)))
            await field_page.goto(field_url);await expect(field_page.locator('#offline-load')).to_contain_text('SHA-256 已核对',timeout=30000)
            await expect(field_page.get_by_text('float64',exact=True)).to_be_visible()
            first=await field_page.locator('#solid-canvas').evaluate('(c)=>c.toDataURL()');await field_page.locator('#solid-frame').evaluate('(e)=>{e.value=80;e.dispatchEvent(new Event("input",{bubbles:true}));}');assert first!=await field_page.locator('#solid-canvas').evaluate('(c)=>c.toDataURL()')
            for i,value in enumerate(['4','1.5','1']):await field_page.locator(f'#solid-point-{i}').fill(value)
            for i,value in enumerate(['1','2','-1']):await field_page.locator(f'#solid-direction-{i}').fill(value)
            await field_page.locator('#solid-extract').click();await expect(field_page.locator('#solid-probe-message')).to_contain_text('未重新求解')
            await field_page.screenshot(path=str(evidence/'float64-field-offline.png'),full_page=True);await field_context.close()
        assert not errors,errors
        receipt={'passed':True,'study_id':study,'five_control_study_submissions':len(requests),'posthoc_point_query':True,'void_or_outside_rejected':True,'direction_and_window_changes':True,'csv_and_offline_html':True,'invalid_window_clears_stale_results':True,'offline_saved_window_preserved':True,'exact_reference_scope_verified':args.expect_reference,'float64_fullfield_verified':bool(args.field_html),'offline_external_requests':network,'javascript_errors':errors,'direct_windows_file_open_tested':False}
        target=evidence/('browser-launch-acceptance.json' if args.launch else 'browser-readonly-acceptance.json');target.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));await browser.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');parser.add_argument('--study');parser.add_argument('--expect-reference',action='store_true');parser.add_argument('--field-html');args=parser.parse_args();assert args.launch or args.study;asyncio.run(main(args))
