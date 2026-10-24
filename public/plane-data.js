// Plane-strain result contract, binary decoding and line analysis for the browser.
// Analysis functions mirror dispersion/analysis.py (Hann-weighted exact-frequency coefficients).
import { amplitudeSpectrum } from './solid-data.js';
export { amplitudeSpectrum };

export const PLANE_SCHEMA='workai.plane_strain.result.v1', PLANE_ENGINE='workai_spectral_plane_strain_2d';
const finite=v=>typeof v==='number'&&Number.isFinite(v);

export function validatePlaneResult(r) {
  if(r?.schema!==PLANE_SCHEMA||r.engine!==PLANE_ENGINE||r.dimension!=='2d_plane_strain'||r.accepted_for_solver_routing!==false)throw new Error('二维结果格式不正确。');
  const m=r.mesh,n=m?.nodes_mm?.length,t=r.times_us?.length,size=(m?.degree+1)**2;
  if(!(n>=4&&n<=60000&&t>=21&&t<=20001&&Number.isInteger(m.degree)&&m.degree>=2&&m.degree<=8&&Array.isArray(m.elements)&&m.elements.length>=1&&m.elements.length*size<=400000))throw new Error('二维结果超出支持规模。');
  if(!m.nodes_mm.every(p=>p.length===2&&p.every(finite))||!r.times_us.every(finite))throw new Error('二维坐标或时间无效。');
  if(!m.elements.every(ids=>ids.length===size&&ids.every(i=>Number.isInteger(i)&&i>=0&&i<n)))throw new Error('二维单元连接无效。');
  if(!Array.isArray(m.boundary_lines)||!m.boundary_lines.every(l=>l.length===2&&l.every(i=>Number.isInteger(i)&&i>=0&&i<n)))throw new Error('二维边界线无效。');
  const step=r.scene?.acquisition?.sample_step_us;
  if(!(step>0)||!r.times_us.every((x,i)=>Math.abs(x-i*step)<1e-7))throw new Error('时间网格不均匀。');
  if(!Array.isArray(r.receivers)||!r.receivers.every(p=>p.u_x_m?.length===t&&p.u_z_m?.length===t&&p.u_x_m.every(finite)&&p.u_z_m.every(finite)&&p.point_mm?.length===3&&p.direction?.length===3))throw new Error('接收点波形无效。');
  const total=r.lines?.shape?.[1]??0;
  if(!Array.isArray(r.receiver_lines)||!r.receiver_lines.every(l=>Array.isArray(l.columns)&&l.columns.length===l.count&&l.columns.every(c=>Number.isInteger(c)&&c>=0&&c<total)&&l.distance_mm?.length===l.count&&l.points_mm?.length===l.count&&l.direction?.length===3))throw new Error('接收线信息无效。');
  checkDescriptor(r.lines,[t,total,2],8,'接收线');
  if(r.field){
    const f=r.field;checkDescriptor(f,[f.shape?.[0],n,2],4,'位移场');
    if(!Array.isArray(f.frame_indices)||f.frame_indices.length!==f.shape[0]||!f.frame_indices.every((v,i)=>Number.isInteger(v)&&v>=0&&v<t&&(i===0||v>f.frame_indices[i-1])))throw new Error('位移场帧索引无效。');
  }
  return r;
}
function checkDescriptor(d,shape,bytes,label){
  const expected={4:'little_endian_float32',8:'little_endian_float64'}[bytes];
  if(!d||d.dtype!==expected||d.unit!=='m'||String(d.shape)!==String(shape)||d.uncompressed_bytes!==shape.reduce((a,b)=>a*b,1)*bytes||d.uncompressed_bytes>256*1024*1024||!/^[a-f0-9]{64}$/.test(d.sha256||'')||!/^[a-f0-9]{64}$/.test(d.uncompressed_sha256||''))throw new Error(label+'描述不正确。');
}
async function digest(value){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',value)),v=>v.toString(16).padStart(2,'0')).join('');}
export async function decodePlaneBinary(descriptor,bytes) {
  if(!globalThis.crypto?.subtle||typeof DecompressionStream==='undefined')throw new Error('浏览器不支持本地摘要校验或解压。');
  if(bytes.byteLength!==descriptor.compressed_bytes)throw new Error('压缩数据长度不匹配。');
  if(await digest(bytes)!==descriptor.sha256)throw new Error('数据 SHA-256 不匹配。');
  const reader=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip')).getReader(),chunks=[];let total=0;
  while(true){const {done,value}=await reader.read();if(done)break;total+=value.byteLength;if(total>descriptor.uncompressed_bytes){await reader.cancel();throw new Error('解压数据超过声明长度。');}chunks.push(value);}
  if(total!==descriptor.uncompressed_bytes)throw new Error('解压数据长度不匹配。');
  const raw=new Uint8Array(total);let offset=0;for(const c of chunks){raw.set(c,offset);offset+=c.length;}
  if(await digest(raw)!==descriptor.uncompressed_sha256)throw new Error('解压数据摘要不匹配。');
  const floats=descriptor.dtype==='little_endian_float64'?new Float64Array(raw.buffer):new Float32Array(raw.buffer);
  if(!floats.every(Number.isFinite))throw new Error('数据含无效数值。');return floats;
}
export async function decodePlaneResult(result,fieldBytes,lineBytes){
  validatePlaneResult(result);
  return {field:fieldBytes&&result.field?await decodePlaneBinary(result.field,fieldBytes):null,lines:await decodePlaneBinary(result.lines,lineBytes)};
}

/** Hann-weighted, mean-removed, exact-frequency single-sided complex coefficients (same as the Python analysis). */
export function harmonicCoefficients(times,values,frequency,start=times[0],end=times.at(-1),orders=[1,2]) {
  if(values.length!==times.length||times.length<4||!Number.isFinite(frequency)||frequency<=0)throw new Error('波形与时间网格需一致且载频为正。');
  const step=times[1]-times[0];if(!(start<end)||start<times[0]-1e-9||end>times.at(-1)+1e-9)throw new Error('分析时间窗必须位于记录内，且起点早于终点。');
  const idx=[];for(let i=0;i<times.length;i++)if(times[i]>=start-1e-9&&times[i]<=end+1e-9)idx.push(i);
  if(idx.length<8)throw new Error('分析时间窗至少需要 8 个采样点。');
  const count=idx.length,window=idx.map((_,i)=>.5*(1-Math.cos(2*Math.PI*i/(count-1)))),sum=window.reduce((a,b)=>a+b,0);
  const mean=idx.reduce((a,i,j)=>a+values[i]*window[j],0)/sum,nyquist=.5/step;
  const harmonics=orders.map(order=>{const f=order*frequency;if(f>nyquist)return {order,frequency_mhz:f,available:false};
    let real=0,imaginary=0;idx.forEach((i,j)=>{const v=(values[i]-mean)*window[j],angle=2*Math.PI*f*times[i];real+=v*Math.cos(angle);imaginary-=v*Math.sin(angle);});
    const scale=(Math.abs(f-nyquist)<1e-12?1:2)/sum;real*=scale;imaginary*=scale;const amplitude=Math.hypot(real,imaginary);
    return {order,frequency_mhz:f,available:true,real_m:real,imaginary_m:imaginary,amplitude_m:amplitude,phase_deg:amplitude>1e-300?Math.atan2(imaginary,real)*180/Math.PI:null};});
  return {start_us:times[idx[0]],end_us:times[idx.at(-1)],sample_count:count,carrier_cycles:(times[idx.at(-1)]-times[idx[0]])*frequency,harmonics};
}

/** Projected trace of one line point: column index into the lines array, direction from the line. */
export function lineTrace(result,lines,line,pointIndex) {
  const total=result.lines.shape[1],frames=result.lines.shape[0],column=line.columns[pointIndex],d=line.direction,out=new Float64Array(frames);
  for(let t=0;t<frames;t++){const base=(t*total+column)*2;out[t]=lines[base]*d[0]+lines[base+1]*d[2];}
  return out;
}
export function lineHarmonics(result,lines,line,frequency,options) {
  const times=result.times_us,rows=[];
  for(let i=0;i<line.count;i++){
    let start,end;
    if(options.mode==='moving'){if(!(options.group_velocity>0)||!(options.burst_us>0))throw new Error('移动窗需要正的群速度与脉冲时长。');const travel=line.distance_mm[i]*1e-3/options.group_velocity*1e6,margin=.25*options.burst_us;start=options.delay_us+travel-margin;end=options.delay_us+travel+options.burst_us+margin;}
    else {start=options.start_us;end=options.end_us;}
    start=Math.max(start,times[0]);end=Math.min(end,times.at(-1));
    const analysis=harmonicCoefficients(times,lineTrace(result,lines,line,i),frequency,start,end);
    rows.push({index:i,distance_mm:line.distance_mm[i],window_us:[analysis.start_us,analysis.end_us],carrier_cycles:analysis.carrier_cycles,harmonics:analysis.harmonics});
  }
  return rows;
}
export function linearFit(x,y){const n=x.length,mx=x.reduce((a,b)=>a+b,0)/n,my=y.reduce((a,b)=>a+b,0)/n;let sxx=0,sxy=0,syy=0;for(let i=0;i<n;i++){sxx+=(x[i]-mx)**2;sxy+=(x[i]-mx)*(y[i]-my);syy+=(y[i]-my)**2;}const slope=sxy/sxx,intercept=my-slope*mx;let residual=0;for(let i=0;i<n;i++)residual+=(y[i]-slope*x[i]-intercept)**2;return {slope,intercept,r_squared:syy>0?1-residual/syy:null};}
export function accumulationFit(rows){
  const x=rows.map(r=>r.distance_mm*1e-3),a1=rows.map(r=>r.harmonics[0].amplitude_m),a2=rows.map(r=>r.harmonics[1].available?r.harmonics[1].amplitude_m:NaN);
  if(a2.some(v=>!Number.isFinite(v)))return {available:false};
  const ratio=a2.map((v,i)=>v/a1[i]**2);
  return {available:true,distance_mm:rows.map(r=>r.distance_mm),a1,a2,ratio,fit_a2:linearFit(x,a2),fit_ratio:linearFit(x,ratio)};
}
function unwrap(phases){const out=[phases[0]];for(let i=1;i<phases.length;i++){let d=phases[i]-phases[i-1];while(d>Math.PI)d-=2*Math.PI;while(d<-Math.PI)d+=2*Math.PI;out.push(out[i-1]+d);}return out;}
export function lineVelocities(rows,frequency) {
  const x=rows.map(r=>r.distance_mm*1e-3),phase=unwrap(rows.map(r=>Math.atan2(r.harmonics[0].imaginary_m,r.harmonics[0].real_m))),fit=linearFit(x,phase),k=-fit.slope;
  return {phase_velocity_m_s:k!==0?2*Math.PI*frequency*1e6/Math.abs(k):null,wavenumber_rad_per_mm:k*1e-3,phase_fit_r_squared:fit.r_squared};
}
/** In-place radix-2 complex FFT (arrays padded to a power of two by the caller). */
export function fft(re,im,inverse=false){const n=re.length;for(let i=1,j=0;i<n;i++){let bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){[re[i],re[j]]=[re[j],re[i]];[im[i],im[j]]=[im[j],im[i]];}}
  for(let length=2;length<=n;length*=2){const angle=(inverse?2:-2)*Math.PI/length,wr=Math.cos(angle),wi=Math.sin(angle);for(let base=0;base<n;base+=length){let ur=1,ui=0;for(let j=0;j<length/2;j++){const a=base+j,b=a+length/2,vr=re[b]*ur-im[b]*ui,vi=re[b]*ui+im[b]*ur;re[b]=re[a]-vr;im[b]=im[a]-vi;re[a]+=vr;im[a]+=vi;const next=ur*wr-ui*wi;ui=ur*wi+ui*wr;ur=next;}}}
  if(inverse)for(let i=0;i<n;i++){re[i]/=n;im[i]/=n;}}
const pow2=n=>{let s=1;while(s<n)s*=2;return s;};
/** |U(f,k)| of a uniformly spaced line: a wave towards +x appears at positive k (exp(i(kx - wt)) convention). */
export function fkSpectrum(result,lines,line,pad=2) {
  const times=result.times_us,nt=times.length,nx=line.count,dt=(times[1]-times[0])*1e-6,dx=(line.distance_mm[1]-line.distance_mm[0])*1e-3;
  if(nx<4||!line.distance_mm.every((v,i)=>i===0||Math.abs(v-line.distance_mm[i-1]-dx*1e3)<1e-6*Math.abs(dx*1e3)+1e-9))throw new Error('f–k 分析需要至少 4 个等距点。');
  const NT=pow2(nt*pad),NX=pow2(nx*pad),data=Array.from({length:nx},(_,i)=>lineTrace(result,lines,line,i));
  // Transform over x first (inverse sign so that +k is +x propagation), then over t.
  const re=Array.from({length:NX},()=>new Float64Array(NT)),im=Array.from({length:NX},()=>new Float64Array(NT));
  for(let t=0;t<nt;t++){const xr=new Float64Array(NX),xi=new Float64Array(NX),wt=.5*(1-Math.cos(2*Math.PI*t/(nt-1)));
    for(let i=0;i<nx;i++)xr[i]=data[i][t]*wt*.5*(1-Math.cos(2*Math.PI*i/(nx-1)));
    fft(xr,xi,true);for(let k=0;k<NX;k++){re[k][t]=xr[k]*NX;im[k][t]=xi[k]*NX;}}
  const half=NT/2+1,magnitude=new Float32Array(NX*half);
  for(let k=0;k<NX;k++){fft(re[k],im[k],false);for(let f=0;f<half;f++)magnitude[k*half+f]=Math.hypot(re[k][f],im[k][f]);}
  const frequency=Array.from({length:half},(_,i)=>i/(NT*dt)*1e-6),order=Array.from({length:NX},(_,i)=>i).sort((a,b)=>kOf(a,NX,dx)-kOf(b,NX,dx));
  return {frequency_mhz:frequency,wavenumber_rad_per_mm:order.map(i=>kOf(i,NX,dx)*1e-3),magnitude:(f,j)=>magnitude[order[j]*half+f],rows:order.length,columns:half};
}
const kOf=(i,n,dx)=>2*Math.PI*(i<n/2?i:i-n)/(n*dx);
