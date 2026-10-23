"""Render frames from an existing verified offline result; never starts a solve."""
import argparse,asyncio,importlib.util,json,shutil
from pathlib import Path
from playwright.async_api import async_playwright,expect
spec=importlib.util.spec_from_file_location('local_report',Path(__file__).with_name('local-report-server.py'));local_report=importlib.util.module_from_spec(spec);spec.loader.exec_module(local_report)

async def render(html_path,output):
    output.mkdir(parents=True,exist_ok=False)
    with local_report.serve_report(html_path) as url:
        await render_loaded(url,html_path,output)

async def render_loaded(url,html_path,output):
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=shutil.which('chromium'),args=['--no-sandbox','--disable-dev-shm-usage'])
        context=await browser.new_context(viewport={'width':1400,'height':1000},device_scale_factor=1)
        async def route(request):
            if request.request.url==url:await request.continue_()
            else:await request.abort()
        await context.route('**/*',route);page=await context.new_page();errors=[];network=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:network.append(r.url) if r.url!=url else None)
        await page.goto(url,timeout=60000);await expect(page.locator('#offline-load')).to_contain_text('SHA-256 已核对',timeout=60000)
        await context.set_offline(True)
        await expect(page.locator('#solid-waveform svg')).to_be_visible()
        count=int(await page.locator('#solid-frame').get_attribute('max'))+1
        frames=list(range(0,count,4))
        if frames[-1]!=count-1:frames.append(count-1)
        for i,frame in enumerate(frames):
            await page.locator('#solid-frame').evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}));}',frame)
            await page.locator('#solid-canvas').evaluate('''c=>{
                const ctx=c.getContext('2d'),r=JSON.parse(document.getElementById('solid-record').textContent),h=c.height,w=c.width;
                ctx.fillStyle='#f3f7f8';ctx.fillRect(0,h-62,w,62);ctx.fillStyle='#31536a';ctx.font='12px sans-serif';
                ctx.fillText(`多晶铝 · ${r.scene.constitutive} · ${r.scene.source.frequency_mhz} MHz / ${r.scene.source.amplitude_mpa} MPa`,15,h-44,w-30);
                ctx.fillText(document.getElementById('solid-color-note').textContent,15,h-26,w-30);
                ctx.fillText('真实有限元位移场 · 动画慢速播放，物理时间见左上角',15,h-8,w-30);
            }''')
            await page.locator('#solid-canvas').screenshot(path=str(output/f'frame-{i:03d}.png'))
        assert not errors and not network,(errors,network)
        receipt={'source':str(html_path),'frames':len(frames),'original_frame_indices':frames,'no_external_requests':True,'loopback_streamed_main_document':True,'offline_after_initial_document':True,'scientific_field_renderer':True,'new_simulations':0}
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));await browser.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('html',type=Path);parser.add_argument('frames',type=Path);args=parser.parse_args()
    asyncio.run(render(args.html,args.frames))
