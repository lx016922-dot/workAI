"""Offline prediction animation/point extraction checked against frozen arrays."""
import argparse,asyncio,csv,io,json,shutil
from pathlib import Path
import numpy as np
from playwright.async_api import async_playwright,expect

async def main(args):
    args.evidence.mkdir(parents=True,exist_ok=False);metrics=json.loads((args.run/'metrics.json').read_text());row=next(c for c in metrics['cases'] if c['case_id']==args.case);errors=[];network=[];outputs=[]
    with np.load(args.run/row['field_file']) as d:parts=d['parts_m'];nodes=d['points_mm'];tetra=d['tetra'];times=d['times_us']
    weights=np.array([.1,.2,.3,.4]);ids=tetra[0];point=weights@nodes[ids];direction=np.array([2.,-3.,1.]);direction/=np.linalg.norm(direction)
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=shutil.which('chromium'),args=['--no-sandbox','--disable-dev-shm-usage'])
        for part,report in [('full',args.full),('material',args.material)]:
            context=await browser.new_context(offline=True,viewport={'width':1440,'height':1000},accept_downloads=True);url='http://127.0.0.1:3000/modal-offline';html=report.read_text()
            async def route(request):
                if request.request.url==url:await request.fulfill(body=html,content_type='text/html')
                else:await request.abort()
            await context.route('**/*',route);page=await context.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:network.append(r.url) if r.url!=url and not r.url.startswith(('blob:','data:')) else None);await page.goto(url);await expect(page.locator('#modal-load')).to_contain_text('已核对预测场 SHA-256',timeout=30000);await expect(page.locator('.notice')).to_contain_text('动画不是有限元参考场');await expect(page.locator('#solid-waveform svg')).to_be_visible()
            first=await page.locator('#solid-canvas').evaluate('(c)=>c.toDataURL()');await page.locator('#solid-frame').evaluate('(e)=>{e.value=80;e.dispatchEvent(new Event("input",{bubbles:true}));}');assert first!=await page.locator('#solid-canvas').evaluate('(c)=>c.toDataURL()');await page.locator('#solid-play').click();before=await page.locator('#solid-frame').input_value();await page.wait_for_timeout(300);assert before!=await page.locator('#solid-frame').input_value();await page.locator('#solid-play').click()
            for i,value in enumerate(point):await page.locator('#solid-point-'+str(i)).fill(str(value))
            for i,value in enumerate([2,-3,1]):await page.locator('#solid-direction-'+str(i)).fill(str(value))
            await page.locator('#solid-extract').click();await expect(page.locator('#solid-probe-message')).to_contain_text('未重新求解');await expect(page.locator('#solid-spectrum svg')).to_be_visible()
            async with page.expect_download() as info:await page.locator('#solid-csv').click()
            target=args.evidence/(part+'-point-spectrum.csv');await (await info.value).save_as(target);lines=[x for x in target.read_text().splitlines() if not x.startswith('#')];rows=list(csv.reader(lines));data=np.array(rows[1:1+len(times)],float);values=parts.sum(2) if part=='full' else parts[:,:,2];expected=np.einsum('tnc,n->tc',values[:,ids],weights);np.testing.assert_allclose(data[:,0],times,rtol=0,atol=1e-12);np.testing.assert_allclose(data[:,1:4],expected,rtol=1e-11,atol=1e-23);np.testing.assert_allclose(data[:,4],expected@direction,rtol=1e-11,atol=1e-23)
            await page.screenshot(path=str(args.evidence/(part+'-desktop.png')),full_page=True);await page.set_viewport_size({'width':390,'height':844});assert await page.evaluate('()=>document.documentElement.scrollWidth<=innerWidth');await page.screenshot(path=str(args.evidence/(part+'-mobile.png')),full_page=True)
            for i in range(3):await page.locator('#solid-direction-'+str(i)).fill('0')
            await page.locator('#solid-extract').click();await expect(page.locator('#solid-probe-message')).to_contain_text('零向量');await expect(page.locator('#solid-csv')).to_be_disabled();await expect(page.locator('#solid-waveform svg')).to_have_count(0)
            await page.locator('#solid-point-0').fill('1000');await page.locator('#solid-direction-0').fill('1');await page.locator('#solid-extract').click();await expect(page.locator('#solid-probe-message')).to_contain_text('实体之外');await expect(page.locator('#solid-csv')).to_be_disabled();outputs.append({'part':part,'frames':len(times),'point_mm':point.tolist(),'direction':direction.tolist(),'CSV_matches_full_field':True});await context.close()
        assert not errors and not network,(errors,network);result={'passed':True,'case':args.case,'animation_and_signed_point_spectrum':outputs,'rejected_input_clears_stale_output':True,'mobile_overflow':False,'external_requests':network,'javascript_errors':errors,'new_computation_requests':0,'direct_Windows_file_open_tested':False};(args.evidence/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));await browser.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--full',type=Path,required=True);p.add_argument('--material',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);asyncio.run(main(p.parse_args()))
