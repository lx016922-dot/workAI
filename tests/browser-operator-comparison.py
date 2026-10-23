"""Offline browser checks against retained numerical arrays; no solves or API."""
import argparse,asyncio,csv,io,json,shutil
from pathlib import Path
import numpy as np
from playwright.async_api import async_playwright,expect

async def main(args):
    out=args.evidence;out.mkdir(parents=True,exist_ok=False);metrics=json.loads((args.run/'metrics.json').read_text());cases=[r for r in metrics['cases'] if r['split']=='test'];saved=np.load(args.run/'probe-predictions.npz',allow_pickle=False);errors=[];network=[];url='http://127.0.0.1:3000/operator-comparison-offline';html=args.report.read_text()
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=shutil.which('chromium'),args=['--no-sandbox','--disable-dev-shm-usage']);context=await browser.new_context(offline=True,viewport={'width':1440,'height':1050},accept_downloads=True)
        async def route(request):
            if request.request.url==url:await request.fulfill(body=html,content_type='text/html')
            else:await request.abort()
        await context.route('**/*',route);page=await context.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:network.append(r.url) if r.url!=url and not r.url.startswith(('blob:','data:')) else None);await page.goto(url)
        title='物理降阶预测与有限元对照' if metrics.get('kind')=='modal_energy' else 'AI 与有限元波形对照'
        await expect(page.get_by_role('heading',name=title,exact=True)).to_be_visible();await expect(page.locator('#oc-case option')).to_have_count(len(cases));checked=0
        for i,c in enumerate(cases):
            await page.locator('#oc-case').select_option(str(i));await page.locator('#oc-part').select_option('material')
            for j,probe in enumerate(c['probes']):
                await page.locator('#oc-probe').select_option(str(j));cells=await page.locator('#oc-metrics tbody td').all_text_contents();assert abs(float(cells[0].rstrip('%'))-100*probe['material_waveform_relative_l2'])<.006
                for index,key in [(1,'material_2f_predicted_nm'),(2,'material_2f_fem_nm')]:np.testing.assert_allclose(float(cells[index]),probe[key],rtol=6e-5,atol=1e-12)
                if probe['material_2f_complex_relative_error'] is None:assert cells[3]=='信号不足'
                else:assert abs(float(cells[3].rstrip('%'))-100*probe['material_2f_complex_relative_error'])<.006
                await expect(page.locator('#oc-wave polyline')).to_have_count(2);await expect(page.locator('#oc-spectrum polyline')).to_have_count(2);checked+=1
        # Export an arbitrary signed direction, independently project all XYZ.
        for i,value in enumerate([2,-3,1]):await page.locator('#oc-direction-'+str(i)).fill(str(value))
        await page.locator('#oc-apply').click();await expect(page.locator('#oc-message')).to_contain_text('尚未通过代理验收')
        async with page.expect_download() as info:await page.locator('#oc-csv').click()
        download=await info.value;csv_path=out/'custom-direction-material.csv';await download.save_as(csv_path);rows=list(csv.reader(line for line in csv_path.read_text().splitlines() if not line.startswith('#')));assert rows[0]==['time_us','ai_displacement_m','fem_displacement_m'];array=np.asarray(rows[1:],dtype=float);direction=np.array([2.,-3.,1.]);direction/=np.linalg.norm(direction);last=cases[-1];prefix=last['case_id']+'_P2'
        np.testing.assert_allclose(array[:,0],saved[last['case_id']+'_times_us'],rtol=0,atol=1e-12);np.testing.assert_allclose(array[:,1],saved[prefix+'_prediction'][:,2,:]@direction,rtol=1e-11,atol=1e-23);np.testing.assert_allclose(array[:,2],saved[prefix+'_fem'][:,2,:]@direction,rtol=1e-11,atol=1e-23)
        await page.screenshot(path=str(out/'comparison-desktop.png'),full_page=True);await page.set_viewport_size({'width':390,'height':844});assert await page.evaluate('()=>document.documentElement.scrollWidth<=innerWidth');await page.screenshot(path=str(out/'comparison-mobile.png'),full_page=True)
        for i in range(3):await page.locator('#oc-direction-'+str(i)).fill('0')
        await page.locator('#oc-apply').click();await expect(page.locator('#oc-message')).to_contain_text('零向量');await expect(page.locator('#oc-csv')).to_be_disabled();await expect(page.locator('#oc-wave svg')).to_have_count(0)
        assert not errors and not network,(errors,network);result={'passed':True,'run_id':args.run.name,'held_out_probe_metrics_reproduced':checked,'custom_signed_direction_csv_reproduced':True,'zero_direction_clears_stale_results':True,'desktop_mobile_no_overflow':True,'offline_requests':network,'javascript_errors':errors,'new_computation_requests':0,'direct_windows_file_open_tested':False};(out/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));await browser.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);asyncio.run(main(p.parse_args()))
