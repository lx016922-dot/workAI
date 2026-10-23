"""Check real offline high-order animation and CSV against solver arrays."""
import argparse
import asyncio
import csv
import json
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import numpy as np
from playwright.async_api import async_playwright,expect
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spectral3d.mesh import interpolation


async def main(args):
    args.evidence.mkdir(parents=True,exist_ok=False);manifest=json.loads(args.campaign.read_text());case=next(c for c in manifest['cases'] if c['case_id']==args.case);root=args.campaign.parent
    with np.load(root/case['mesh_file'],allow_pickle=False) as data:mesh=dict(data)
    if 'mapping_kind' in mesh:
        from spectral3d.curved_mesh import shape
        w,_=shape(mesh,2*np.array([.31,.42,.67])-1);point=w@mesh['points'][mesh['elements'][0].ravel()]*1000
    else:point=(mesh['origin'][0]+mesh['size'][0]*np.array([.31,.42,.67]))*1000
    ids,weights=interpolation(mesh,point);direction=np.array([2.,-3.,1.]);direction/=np.linalg.norm(direction)
    saved={}
    for law in ['murnaghan','stvk']:
        control=next(c for c in case['controls'] if c['law']==law)
        with np.load(root/control['directory']/'field.npz',allow_pickle=False) as data:saved[law]=np.einsum('tac,a->tc',data['displacement_m'][:,ids],weights)
    errors=[];requests=[];checks=[]
    files={'/full':args.full,'/material':args.material}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=files.get(self.path)
            if path is None:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(path.stat().st_size));self.end_headers()
            with path.open('rb') as source:shutil.copyfileobj(source,self.wfile)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
    base='http://127.0.0.1:'+str(server.server_address[1])
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=shutil.which('chromium'),args=['--no-sandbox','--disable-dev-shm-usage'])
        for part,path in [('full',args.full),('material',args.material)]:
            context=await browser.new_context(viewport={'width':1440,'height':1000},accept_downloads=True);page=await context.new_page();page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:requests.append(r.url) if not r.url.startswith((base,'data:','blob:')) else None)
            async def only_report(route):
                if route.request.url==base+'/'+part and route.request.method=='GET':await route.continue_()
                else:await route.abort()
            await context.route('**/*',only_report)
            await page.goto(base+'/'+part,timeout=60000);await expect(page.locator('#spectral-load')).to_contain_text('已核对高阶参考场 SHA-256',timeout=60000);await expect(page.locator('.notice').first).to_contain_text('高阶超弹性有限元参考场');await expect(page.locator('#solid-waveform svg')).to_be_visible()
            first=await page.locator('#solid-canvas').evaluate('(c)=>c.toDataURL()');await page.locator('#solid-frame').evaluate('(e)=>{e.value=80;e.dispatchEvent(new Event("input",{bubbles:true}));}');assert first!=await page.locator('#solid-canvas').evaluate('(c)=>c.toDataURL()')
            await page.locator('#solid-play').click();before=await page.locator('#solid-frame').input_value();await page.wait_for_timeout(400);assert before!=await page.locator('#solid-frame').input_value();await page.locator('#solid-play').click()
            for i,value in enumerate(point):await page.locator('#solid-point-'+str(i)).fill(str(value))
            for i,value in enumerate([2,-3,1]):await page.locator('#solid-direction-'+str(i)).fill(str(value))
            await page.locator('#solid-extract').click();await expect(page.locator('#solid-spectrum svg')).to_be_visible()
            async with page.expect_download() as info:await page.locator('#solid-csv').click()
            target=args.evidence/(part+'.csv');await (await info.value).save_as(target);rows=list(csv.reader(x for x in target.read_text().splitlines() if not x.startswith('#')));count=len(saved['murnaghan']);data=np.array(rows[1:1+count],float)
            expected=saved['murnaghan'] if part=='full' else saved['murnaghan']-saved['stvk']
            # Material interpolation after nodal subtraction differs only in rounding.
            np.testing.assert_allclose(data[:,1:4],expected,rtol=1e-9,atol=1e-21);np.testing.assert_allclose(data[:,4],expected@direction,rtol=1e-9,atol=1e-21)
            # Independently reproduce the exported full-record Hann FFT.
            spectrum=np.array(rows[count+3:],float);values=data[:,4];window=np.hanning(count);centered=values-np.sum(window*values)/window.sum();size=1<<(count-1).bit_length();amp=abs(np.fft.rfft(centered*window,n=size))*2/window.sum();amp[[0,-1]]/=2
            np.testing.assert_allclose(spectrum[:,1],amp,rtol=1e-10,atol=1e-21)
            await page.screenshot(path=str(args.evidence/(part+'-desktop.png')),full_page=True);await page.set_viewport_size({'width':390,'height':844});assert await page.evaluate('()=>document.documentElement.scrollWidth<=innerWidth');await page.screenshot(path=str(args.evidence/(part+'-mobile.png')),full_page=True)
            for i in range(3):await page.locator('#solid-direction-'+str(i)).fill('0')
            await page.locator('#solid-extract').click();await expect(page.locator('#solid-csv')).to_be_disabled();await expect(page.locator('#solid-waveform svg')).to_have_count(0)
            g=case['scene']['geometry'];void=([g['length_mm']/2,0,0] if g['kind']=='pipe' else [g['hole_x_mm'],g['hole_y_mm'],g['thickness_mm']/2] if g['kind']=='hole' else [g['notch_x_mm'],g['width_mm']/2,g['thickness_mm']-.1*g['notch_depth_mm']])
            for i,value in enumerate(void):await page.locator('#solid-point-'+str(i)).fill(str(value))
            await page.locator('#solid-direction-0').fill('1');await page.locator('#solid-extract').click();await expect(page.locator('#solid-probe-message')).to_contain_text('实体之外');await expect(page.locator('#solid-csv')).to_be_disabled()
            checks.append({'part':part,'frames':count,'custom_point_mm':point.tolist(),'direction':[2,-3,1],'csv_matches_high_order_field':True,'spectrum_matches_numpy':True});await context.close()
        assert not errors and not requests,(errors,requests)
        result={'passed':True,'checks':checks,'void_and_zero_direction_rejected':True,'desktop_and_mobile':True,'external_requests':requests,'javascript_errors':errors,'new_solves':0,'standalone_html_served_on_loopback':True,'direct_file_open_tested':False}
        (args.evidence/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));await browser.close()
    server.shutdown();server.server_close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--full',type=Path,required=True);p.add_argument('--material',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);asyncio.run(main(p.parse_args()))
