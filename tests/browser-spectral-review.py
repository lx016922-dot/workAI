import argparse
import asyncio
import csv
import json
import shutil
from pathlib import Path
import numpy as np
from playwright.async_api import async_playwright,expect


async def main(args):
    args.evidence.mkdir(parents=True,exist_ok=False);errors=[];posts=[];external=[];origin='http://127.0.0.1:3000'
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=shutil.which('chromium'),args=['--no-sandbox','--disable-dev-shm-usage']);context=await browser.new_context(viewport={'width':1440,'height':1050});page=await context.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:posts.append(r.url) if r.method=='POST' else None)
        await page.goto(origin+'/#training');await expect(page.locator('#spectral-reference-progress')).to_be_visible(timeout=30000)
        data=await (await page.request.get(origin+'/api/training')).json();upgrade=data['reference_upgrade'];assert upgrade['completed_solves']==45 and len(upgrade['cases'])==3 and all(c['all_component_checks_passed'] for c in upgrade['cases']);assert not data['accepted_for_solver_routing'] and not data['metrics']['acceptance']['all_gates_passed']
        await expect(page.locator('#spectral-reference-progress')).to_contain_text('这些新结果尚未加入当前模型训练');await expect(page.locator('#modal-acceptance')).to_contain_text('尚未通过完整替代验收');await page.screenshot(path=str(args.evidence/'training-desktop.png'),full_page=True);await page.set_viewport_size({'width':390,'height':844});assert await page.evaluate('()=>document.documentElement.scrollWidth<=innerWidth');await page.screenshot(path=str(args.evidence/'training-mobile.png'),full_page=True);await context.close()
        context=await browser.new_context(offline=True,viewport={'width':1440,'height':1050},accept_downloads=True);url=origin+'/spectral-review-offline'
        async def route(request):
            if request.request.url==url:await request.fulfill(body=args.report.read_text(),content_type='text/html')
            else:await request.abort()
        await context.route('**/*',route);page=await context.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:external.append(r.url) if r.url!=url else None);await page.goto(url)
        await expect(page.get_by_role('heading',name='铝板与缺口的高阶参考解',exact=True)).to_be_visible();await expect(page.locator('#oc-wave svg')).to_be_visible();record=json.loads(await page.locator('#comparison-record').text_content());assert len(record['cases'])==3
        for i,case in enumerate(record['cases']):
            await page.locator('#oc-case').select_option(str(i));await page.locator('#oc-part').select_option('material')
            for j,receiver in enumerate(case['receivers']):
                await page.locator('#oc-probe').select_option(str(j));direction=np.array(receiver['direction']);direction/=np.linalg.norm(direction);a=np.asarray(receiver['predicted_parts_m'])[:,2]@direction;b=np.asarray(receiver['fem_parts_m'])[:,2]@direction
                expected=100*np.linalg.norm(a-b)/np.linalg.norm(b);text=await page.locator('#oc-metrics tbody tr td').nth(0).inner_text();assert abs(float(text.strip('%'))-expected)<.0051
        for i,value in enumerate([2,-3,1]):await page.locator('#oc-direction-'+str(i)).fill(str(value))
        await page.locator('#oc-apply').click()
        async with page.expect_download() as info:await page.locator('#oc-csv').click()
        target=args.evidence/'reference-waveforms.csv';await (await info.value).save_as(target);lines=list(csv.reader(x for x in target.read_text().splitlines() if not x.startswith('#')));assert lines[0]==['time_us','coarse_reference_m','fine_reference_m'];values=np.array(lines[1:],float);direction=np.array([2,-3,1])/np.sqrt(14);receiver=record['cases'][-1]['receivers'][-1]
        np.testing.assert_allclose(values[:,1],np.asarray(receiver['predicted_parts_m'])[:,2]@direction,rtol=1e-12,atol=1e-24);np.testing.assert_allclose(values[:,2],np.asarray(receiver['fem_parts_m'])[:,2]@direction,rtol=1e-12,atol=1e-24)
        await page.screenshot(path=str(args.evidence/'review-desktop.png'),full_page=True);await page.set_viewport_size({'width':390,'height':844});assert await page.evaluate('()=>document.documentElement.scrollWidth<=innerWidth');await page.screenshot(path=str(args.evidence/'review-mobile.png'),full_page=True)
        for i in range(3):await page.locator('#oc-direction-'+str(i)).fill('0')
        await page.locator('#oc-apply').click();await expect(page.locator('#oc-csv')).to_be_disabled();assert not errors and not posts and not external
        result={'passed':True,'live_reference_cases':3,'new_reference_solves':45,'old_model_acceptance_unchanged':True,'six_probe_metrics_reproduced':True,'arbitrary_direction_csv_matches':True,'desktop_mobile_no_overflow':True,'javascript_errors':errors,'post_requests':posts,'offline_external_requests':external};(args.evidence/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));await browser.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);asyncio.run(main(p.parse_args()))
