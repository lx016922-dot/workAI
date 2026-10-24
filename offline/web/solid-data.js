// The same interpolation and spectrum routines are used online and offline.
export function unitVector(v) {
  if(!Array.isArray(v)||v.length!==3||!v.every(Number.isFinite))throw new Error('方向需要三个有限数值。');
  const length=Math.hypot(...v);if(!Number.isFinite(length)||length<1e-12)throw new Error('方向不能为零向量，且长度必须有限。');return v.map(x=>x/length);
}
export function locateSolidPoint(result,point) {
  if(!Array.isArray(point)||point.length!==3||!point.every(Number.isFinite))throw new Error('坐标需要三个有限数值（mm）。');
  const nodes=result.nodes_mm;
  for(let index=0;index<result.tetra.length;index++) {
    const ids=result.tetra[index],p=ids.map(i=>nodes[i]);
    if([0,1,2].some(j=>point[j]<Math.min(...p.map(v=>v[j]))-1e-7||point[j]>Math.max(...p.map(v=>v[j]))+1e-7))continue;
    const sub=(a,b)=>a.map((x,j)=>x-b[j]),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],dot=(a,b)=>a.reduce((sum,x,j)=>sum+x*b[j],0);
    const a=sub(p[1],p[0]),b=sub(p[2],p[0]),c=sub(p[3],p[0]),q=sub(point,p[0]),det=dot(a,cross(b,c));
    if(Math.abs(det)<1e-16)continue;
    const weights=[dot(q,cross(b,c))/det,dot(a,cross(q,c))/det,dot(a,cross(b,q))/det];weights.unshift(1-weights.reduce((x,y)=>x+y,0));
    if(weights.every(w=>w>=-1e-7&&w<=1+1e-7)){const clipped=weights.map(w=>Math.max(0,Math.min(1,w))),sum=clipped.reduce((a,b)=>a+b,0);return {tetra_index:index,ids,weights:clipped.map(w=>w/sum),point_mm:point};}
  }
  throw new Error('该点位于实体之外或孔洞/管腔内。请修改坐标；不会自动吸附到其他位置。');
}
export function pointSeries(result,field,point,direction) {
  const location=locateSolidPoint(result,point),d=unitVector(direction),n=result.nodes_mm.length;
  const xyz=result.times_us.map((_,t)=>[0,1,2].map(axis=>location.ids.reduce((sum,id,j)=>sum+location.weights[j]*field[(t*n+id)*3+axis],0)));
  return {location,direction:d,xyz_m:xyz,projected_m:xyz.map(v=>v.reduce((sum,x,j)=>sum+x*d[j],0))};
}
export function amplitudeSpectrum(values,stepUs) {
  const count=values.length;if(count<4||!values.every(Number.isFinite)||!(stepUs>0))throw new Error('频谱需要完整、均匀采样的有限波形。');
  let size=1;while(size<count)size*=2;
  const re=new Float64Array(size),im=new Float64Array(size),window=values.map((_,i)=>.5*(1-Math.cos(2*Math.PI*i/(count-1))));
  const sumW=window.reduce((a,b)=>a+b,0),mean=values.reduce((sum,v,i)=>sum+window[i]*v,0)/sumW;
  for(let i=0;i<count;i++)re[i]=(values[i]-mean)*window[i];
  for(let i=1,j=0;i<size;i++) {let bit=size>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){[re[i],re[j]]=[re[j],re[i]];[im[i],im[j]]=[im[j],im[i]];}}
  for(let length=2;length<=size;length*=2) {
    const angle=-2*Math.PI/length,wr=Math.cos(angle),wi=Math.sin(angle);
    for(let base=0;base<size;base+=length) {let ur=1,ui=0;
      for(let j=0;j<length/2;j++){const a=base+j,b=a+length/2,vr=re[b]*ur-im[b]*ui,vi=re[b]*ui+im[b]*ur;
        re[b]=re[a]-vr;im[b]=im[a]-vi;re[a]+=vr;im[a]+=vi;if((j+1)%64===0){ur=Math.cos(angle*(j+1));ui=Math.sin(angle*(j+1));}else{const next=ur*wr-ui*wi;ui=ur*wi+ui*wr;ur=next;}}
    }
  }
  const frequency_mhz=[],amplitude_m=[],power_spectral_density_m2_per_mhz=[],sumW2=window.reduce((s,w)=>s+w*w,0);
  for(let i=0;i<=size/2;i++){const side=i===0||i===size/2?1:2;frequency_mhz.push(i/(size*stepUs));amplitude_m.push(Math.hypot(re[i],im[i])/sumW*side);power_spectral_density_m2_per_mhz.push(side*(re[i]**2+im[i]**2)*stepUs/sumW2);}
  return {frequency_mhz,amplitude_m,power_spectral_density_m2_per_mhz,window:'Hann',weighted_mean_removed:true,fft_size:size,bin_spacing_mhz:1/(size*stepUs),record_resolution_mhz:1/(count*stepUs),nyquist_mhz:.5/stepUs};
}
export function analyzeTimeWindow(values,times,frequency,start=times[0],end=times.at(-1),signal=null) {
  if(values.length!==times.length||times.length<4||!times.every(Number.isFinite)||!values.every(Number.isFinite)||!Number.isFinite(start)||!Number.isFinite(end)||!(start<end)||start<times[0]||end>times.at(-1)||!Number.isFinite(frequency)||frequency<=0)throw new Error('分析时间窗必须位于记录内，且起点早于终点。');
  const step=times[1]-times[0];
  if(!(step>0)||!times.every((t,i)=>Math.abs(t-times[0]-i*step)<1e-8))throw new Error('分析需要均匀时间采样。');
  const indices=times.map((t,i)=>t>=start-1e-9&&t<=end+1e-9?i:-1).filter(i=>i>=0);
  if(indices.length<8)throw new Error('分析时间窗至少需要 8 个采样点。');
  const samples=indices.map(i=>values[i]),selectedTimes=indices.map(i=>times[i]),count=samples.length;
  const window=samples.map((_,i)=>.5*(1-Math.cos(2*Math.PI*i/(count-1)))),sum=window.reduce((a,b)=>a+b,0),mean=samples.reduce((a,v,i)=>a+v*window[i],0)/sum;
  const spectrum=amplitudeSpectrum(samples,step),chirp=signal?.kind==='chirp',harmonics=(chirp?[]:[1,2]).map(order=>{
    const f=frequency*order;if(f>spectrum.nyquist_mhz)return {order,frequency_mhz:f,available:false};
    let real=0,imaginary=0;
    samples.forEach((v,i)=>{const angle=2*Math.PI*f*selectedTimes[i],value=(v-mean)*window[i];real+=value*Math.cos(angle);imaginary-=value*Math.sin(angle);});
    const scale=(Math.abs(f-spectrum.nyquist_mhz)<1e-12?1:2)/sum;real*=scale;imaginary*=scale;
    const amplitude=Math.hypot(real,imaginary);
    return {order,frequency_mhz:f,available:true,real_m:real,imaginary_m:imaginary,amplitude_m:amplitude,phase_deg:amplitude>1e-30?Math.atan2(imaginary,real)*180/Math.PI:null};
  });
  const bands=chirp?[1,2].map(order=>{const lo=order*Math.min(frequency,signal.end_frequency_mhz??frequency),hi=order*Math.max(frequency,signal.end_frequency_mhz??frequency);let power=0;for(let i=0;i<spectrum.frequency_mhz.length;i++)if(spectrum.frequency_mhz[i]>=lo&&spectrum.frequency_mhz[i]<=hi)power+=spectrum.power_spectral_density_m2_per_mhz[i]*spectrum.bin_spacing_mhz;return {order,start_mhz:lo,end_mhz:hi,windowed_rms_m:Math.sqrt(power),available:hi<=spectrum.nyquist_mhz};}):[];
  return {indices,times_us:selectedTimes,values_m:samples,spectrum,harmonics,bands,start_us:selectedTimes[0],end_us:selectedTimes.at(-1),sample_count:count,carrier_cycles:(selectedTimes.at(-1)-selectedTimes[0])*frequency};
}
export function validateSolidResult(r) {
  if(r?.schema!=='workai.hyperelastic.result.v1'||r.engine!=='workai_explicit_hyperelastic_3d')throw new Error('三维结果格式不正确。');
  return validateDisplacementLayout(r);
}
export function validateDisplacementLayout(r) {
  const n=r.nodes_mm?.length,t=r.times_us?.length;
  const nodeLimit=r.engine==='workai_graphwave_offline'?30000:12000;
  if(!(n>=4&&n<=nodeLimit&&t>=21&&t<=2001&&r.tetra?.length<=40000&&r.surface?.length<=50000))throw new Error('三维结果超出支持规模。');
  if(!r.nodes_mm.every(p=>p.length===3&&p.every(Number.isFinite))||!r.times_us.every(Number.isFinite))throw new Error('三维坐标或时间无效。');
  for(const [values,width] of [[r.tetra,4],[r.surface,3]])if(!values.every(v=>v.length===width&&new Set(v).size===width&&v.every(i=>Number.isInteger(i)&&i>=0&&i<n)))throw new Error('三维网格连接无效。');
  const bytesPerValue={little_endian_float32:4,little_endian_float64:8}[r.field?.dtype];
  if(!bytesPerValue||r.field?.unit!=='m'||r.field?.layout!=='time,node,xyz'||String(r.field.shape)!==String([t,n,3])||r.field.uncompressed_bytes!==t*n*3*bytesPerValue||r.field.uncompressed_bytes>192*1024*1024)throw new Error('位移场尺寸或单位不正确。');
  const step=r.scene.acquisition.sample_step_us;
  if(r.recording){
    const m=r.recording;
    if(!['workai.long_record.v1','workai.graphnet.long_record.v1'].includes(m.schema)||m.animation_only!==true||m.analysis_source!=='complete_float64_chunks'||!Number.isInteger(m.frame_count)||m.frame_count<2||m.frame_count>200001||m.sample_step_us!==step||Math.abs((m.frame_count-1)*step-m.duration_us)>1e-7||Math.abs(r.times_us.at(-1)-m.duration_us)>1e-7||r.times_us[0]!==0||r.times_us.some((v,i)=>i&&v<=r.times_us[i-1]))throw new Error('完整记录与动画时间信息不一致。');
  }else if(!r.times_us.every((x,i)=>Math.abs(x-i*step)<1e-7))throw new Error('时间网格不均匀，不能生成频谱。');
  return r;
}
export async function decodeSolidField(result,bytes) {
  validateSolidResult(result);return decodeDisplacementField(result,bytes);
}
export async function decodeDisplacementField(result,bytes) {
  validateDisplacementLayout(result);if(bytes.byteLength!==result.field.compressed_bytes)throw new Error('压缩位移场长度不匹配。');
  const digest=async value=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',value)),v=>v.toString(16).padStart(2,'0')).join('');
  if(!globalThis.crypto?.subtle)throw new Error('浏览器不支持本地摘要校验，请用现代 Chrome/Edge 打开文件或 HTTPS 页面。');
  if(await digest(bytes)!==result.field.sha256)throw new Error('位移场 SHA-256 不匹配。');
  const reader=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip')).getReader(),chunks=[];let total=0;
  while(true){const {done,value}=await reader.read();if(done)break;total+=value.byteLength;if(total>result.field.uncompressed_bytes){await reader.cancel();throw new Error('解压位移场超过声明长度。');}chunks.push(value);}
  if(total!==result.field.uncompressed_bytes)throw new Error('解压位移场长度不匹配。');
  const raw=new Uint8Array(total);let offset=0;for(const chunk of chunks){raw.set(chunk,offset);offset+=chunk.length;}
  if(await digest(raw)!==result.field.uncompressed_sha256)throw new Error('解压位移场摘要不匹配。');
  const floats=result.field.dtype==='little_endian_float64'?new Float64Array(raw.buffer):new Float32Array(raw.buffer);
  if(!floats.every(Number.isFinite))throw new Error('位移场含无效数值。');return floats;
}
