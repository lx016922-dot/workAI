"""Read-only browser acceptance for completed propagation studies and reports."""
import argparse,asyncio,importlib.util,json,shutil
from pathlib import Path
from playwright.async_api import async_playwright,expect
spec=importlib.util.spec_from_file_location('local_report',Path(__file__).resolve().parents[1]/'tools/local-report-server.py');local_report=importlib.util.module_from_spec(spec);spec.loader.exec_module(local_report)

ORIGIN='http://127.0.0.1:3000'
async def check_waveguide(page):
    panel=page.locator('#finite-width-reference')
    await expect(panel).to_be_visible()
    for text in ['基频形状与相位检查通过','375.10','959.22','15.030','不是模态能量占比','未计算非线性模态耦合积分']:
        await expect(panel).to_contain_text(text)
    assert await panel.locator('tbody tr').count()==5

async def main(args):
    evidence=Path(args.evidence);evidence.mkdir(parents=True,exist_ok=True);errors=[];posts=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=shutil.which('chromium'),args=['--no-sandbox','--disable-dev-shm-usage'])
        context=await browser.new_context(viewport={'width':1440,'height':1100});page=await context.new_page()
        page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:posts.append(r.url) if r.method=='POST' else None)
        await page.goto(ORIGIN+'/#solid');await expect(page.locator('#solid-study-select')).to_be_visible(timeout=30000)
        await page.locator('#solid-study-select').select_option(args.study);await expect(page.locator('#solid-study-probe')).to_be_visible(timeout=60000)
        await page.locator('#solid-study-probe').click();await expect(page.locator('#harmonic-waveform svg')).to_be_visible(timeout=90000)
        await expect(page.locator('#harmonic-direct-window')).to_be_enabled();await page.locator('#harmonic-direct-window').click()
        if args.expect_validation or args.expect_partial_validation:await expect(page.locator('#harmonic-reference-status')).to_contain_text('部分对照未通过' if args.expect_partial_validation else '本窗对照通过')
        await expect(page.locator('#harmonic-propagation-plan')).to_contain_text('建议窗通过端面回波筛查')
        start=float(await page.locator('#harmonic-start').input_value());end=float(await page.locator('#harmonic-end').input_value());assert abs(end-start-10)<1e-10
        await page.locator('#solid-propagation-read').click();await expect(page.locator('#solid-propagation-export')).to_be_visible(timeout=180000)
        await expect(page.get_by_text('至少一段超出 5%',exact=False)).to_be_visible()
        if args.expect_waveguide:await check_waveguide(page)
        async with page.expect_download() as download:await page.locator('#solid-propagation-export').click()
        exported=evidence/'browser-propagation-export.html';await (await download.value).save_as(exported)
        await page.screenshot(path=str(evidence/'propagation-live-desktop.png'),full_page=True)
        await page.set_viewport_size({'width':390,'height':844});assert await page.evaluate('()=>document.documentElement.scrollWidth<=innerWidth')
        await page.screenshot(path=str(evidence/'propagation-live-mobile.png'),full_page=True)
        assert not [u for u in posts if u==ORIGIN+'/api/solid/studies']
        await context.close();offline=await browser.new_context(offline=True,viewport={'width':1440,'height':1100});network=[];url=ORIGIN+'/propagation-offline'
        html=Path(args.report).read_text() if args.report else exported.read_text()
        async def route(request):
            if request.request.url==url:await request.fulfill(body=html,content_type='text/html')
            else:await request.abort()
        await offline.route('**/*',route);page=await offline.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:network.append(r.url) if r.url!=url else None)
        await page.goto(url);await expect(page.locator('#harmonic-waveform svg')).to_be_visible(timeout=30000)
        await expect(page.get_by_text('至少一段超出 5%',exact=False)).to_be_visible()
        if args.expect_waveguide:await check_waveguide(page)
        if args.expect_validation or args.expect_partial_validation:await expect(page.locator('#harmonic-reference-status')).to_contain_text('部分对照未通过' if args.expect_partial_validation else '本窗对照通过')
        first=await page.locator('#harmonic-waveform polyline').get_attribute('points')
        await page.locator('#propagation-receiver').select_option('2');assert first!=await page.locator('#harmonic-waveform polyline').get_attribute('points')
        if args.expect_validation or args.expect_partial_validation:await expect(page.locator('#harmonic-reference-status')).to_contain_text('本窗对照通过')
        await page.locator('#harmonic-direction-1').fill('1');await page.locator('#harmonic-analyze').click();await expect(page.locator('#harmonic-reference-status')).to_contain_text('尚无细化对照')
        await page.locator('#harmonic-end').fill('100');await page.locator('#harmonic-analyze').click();assert await page.locator('#harmonic-waveform svg').count()==0
        await page.locator('#harmonic-direct-window').click();await expect(page.locator('#harmonic-waveform svg')).to_be_visible()
        await page.locator('#harmonic-direction-1').fill('0');await page.locator('#harmonic-analyze').click()
        async with page.expect_download() as download:await page.locator('#harmonic-csv').click()
        text=Path(await (await download.value).path()).read_text();assert 'material_increment_m' in text and '# point_mm,24,2,0.5' in text
        await page.screenshot(path=str(evidence/'propagation-offline-desktop.png'),full_page=True)
        await page.set_viewport_size({'width':390,'height':844});assert await page.evaluate('()=>document.documentElement.scrollWidth<=innerWidth')
        await page.screenshot(path=str(evidence/'propagation-offline-mobile.png'),full_page=True)
        assert not errors and not network,(errors,network)
        if args.field_html:
            await offline.close()
            with local_report.serve_report(args.field_html) as field_url:
                field_context=await browser.new_context(viewport={'width':1440,'height':1100})
                async def field_route(request):
                    if request.request.url==field_url:await request.continue_()
                    else:await request.abort()
                await field_context.route('**/*',field_route);field_page=await field_context.new_page();field_page.on('pageerror',lambda e:errors.append(str(e)));field_page.on('request',lambda r:network.append(r.url) if r.url!=field_url else None)
                await field_page.goto(field_url,timeout=60000);await expect(field_page.locator('#offline-load')).to_contain_text('SHA-256 已核对',timeout=60000);await field_context.set_offline(True)
                await expect(field_page.get_by_text('float64',exact=True)).to_be_visible();initial=await field_page.locator('#solid-canvas').evaluate('(c)=>c.toDataURL()')
                await field_page.locator('#solid-frame').evaluate('(e)=>{e.value=80;e.dispatchEvent(new Event("input",{bubbles:true}));}');assert initial!=await field_page.locator('#solid-canvas').evaluate('(c)=>c.toDataURL()')
                for i,value in enumerate(['30','1.5','0.5']):await field_page.locator(f'#solid-point-{i}').fill(value)
                for i,value in enumerate(['1','2','-1']):await field_page.locator(f'#solid-direction-{i}').fill(value)
                await field_page.locator('#solid-extract').click();await expect(field_page.locator('#solid-probe-message')).to_contain_text('未重新求解');await expect(field_page.locator('#solid-spectrum svg')).to_be_visible()
                await field_page.screenshot(path=str(evidence/'propagation-field-desktop.png'),full_page=True);await field_page.set_viewport_size({'width':390,'height':844});assert await field_page.evaluate('()=>document.documentElement.scrollWidth<=innerWidth')
                await field_context.close()
        assert not errors and not network,(errors,network)
        receipt={'passed':True,'study_id':args.study,'new_solves_submitted':0,'suggested_window_and_phase_warning':True,'three_receiver_comparison_and_export':True,'offline_receiver_direction_window_and_csv':True,'desktop_mobile_layout':True,'reference_scope_verified':args.expect_validation or args.expect_partial_validation,'partial_numerical_failure_preserved':args.expect_partial_validation,'finite_width_reference_live_and_offline':args.expect_waveguide,'float64_fullfield_arbitrary_point_direction':bool(args.field_html),'fullfield_loopback_stream_then_offline':bool(args.field_html),'javascript_errors':errors,'offline_external_requests':network,'direct_windows_file_open_tested':False}
        (evidence/'browser-acceptance.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));await browser.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--evidence',required=True);p.add_argument('--report');p.add_argument('--field-html');p.add_argument('--expect-validation',action='store_true');p.add_argument('--expect-partial-validation',action='store_true');p.add_argument('--expect-waveguide',action='store_true');asyncio.run(main(p.parse_args()))
