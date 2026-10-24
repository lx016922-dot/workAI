import { validateDisplacementLayout, decodeDisplacementField, unitVector } from './solid-data.js';

export function validateSpectralResult(result) {
  if(result?.schema!=='workai.offline.graphwave.result.v1'||result.engine!=='workai_graphwave_offline'||result.neural_model?.numerical_correction!==false||!/^[a-f0-9]{64}$/.test(result.neural_model.weights_sha256))throw new Error('离线网络结果格式不正确。');
  validateDisplacementLayout(result);
  const mesh=result.spectral,p=mesh?.degree,n=p+1,count=result.nodes_mm.length;
  if(!Number.isInteger(p)||p<2||p>6||!Array.isArray(mesh.gll)||mesh.gll.length!==n||!mesh.gll.every(Number.isFinite)||Math.abs(mesh.gll[0]+1)>1e-12||Math.abs(mesh.gll[p]-1)>1e-12||mesh.gll.some((x,i)=>i&&x<=mesh.gll[i-1])||!Array.isArray(mesh.elements)||!mesh.elements.length||mesh.elements.length*n**3>100000||(mesh.mapping_kind!=='isoparametric'&&(mesh.origins_mm?.length!==mesh.elements.length||mesh.sizes_mm?.length!==mesh.elements.length)))throw new Error('高阶网格信息无效。');
  for(let e=0;e<mesh.elements.length;e++){
    const ids=mesh.elements[e],origin=mesh.origins_mm?.[e],size=mesh.sizes_mm?.[e];
    if(!Array.isArray(ids)||ids.length!==n**3||new Set(ids).size!==n**3||!ids.every(i=>Number.isInteger(i)&&i>=0&&i<count)||(mesh.mapping_kind!=='isoparametric'&&(origin?.length!==3||!origin.every(Number.isFinite)||size?.length!==3||!size.every(x=>Number.isFinite(x)&&x>0))))throw new Error('高阶单元连接无效。');
    for(let i=0;i<n;i++)for(let j=0;j<n;j++)for(let k=0;k<n;k++){
      if(mesh.mapping_kind==='isoparametric'){const mapped=spectralMap(result,e,[mesh.gll[i],mesh.gll[j],mesh.gll[k]]);if(!(determinant3(mapped.jacobian)>1e-12))throw new Error('曲面单元存在非正几何雅可比。');continue;}
      const point=result.nodes_mm[ids[(i*n+j)*n+k]];
      if([i,j,k].some((q,axis)=>Math.abs(point[axis]-(origin[axis]+(mesh.gll[q]+1)*size[axis]/2))>1e-7))throw new Error('高阶节点与几何映射不一致。');
    }
  }
  return result;
}

export function spectralPointSeries(result,field,point,direction){
  if(!Array.isArray(point)||point.length!==3||!point.every(Number.isFinite))throw new Error('坐标需要三个有限数值（mm）。');
  const d=unitVector(direction),mesh=result.spectral,n=mesh.degree+1;
  let e,weights;
  if(mesh.mapping_kind==='isoparametric')({element:e,weights}=spectralCurvedLocation(result,point));
  else{
    e=mesh.origins_mm.findIndex((origin,index)=>point.every((value,axis)=>value>=origin[axis]-1e-9&&value<=origin[axis]+mesh.sizes_mm[index][axis]+1e-9));
    if(e<0)throw new Error('该点位于实体之外或缺口空域内；不会自动吸附。');
    const local=point.map((value,axis)=>2*(value-mesh.origins_mm[e][axis])/mesh.sizes_mm[e][axis]-1);
    weights=spectralShape(mesh,local).weights;
  }
  const ids=mesh.elements[e],count=result.nodes_mm.length;
  const xyz=result.times_us.map((_,t)=>[0,1,2].map(axis=>ids.reduce((sum,id,j)=>sum+weights[j]*field[(t*count+id)*3+axis],0)));
  return {location:{element_index:e,ids,weights,point_mm:point},direction:d,xyz_m:xyz,projected_m:xyz.map(v=>v.reduce((sum,value,axis)=>sum+value*d[axis],0))};
}

export async function decodeSpectralField(result,bytes){validateSpectralResult(result);return decodeDisplacementField(result,bytes);}

function determinant3(a){return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);}
function solve3(a,b){const det=determinant3(a);if(!Number.isFinite(det)||Math.abs(det)<1e-20)throw new Error('退化曲面映射。');return [0,1,2].map(axis=>determinant3(a.map((row,i)=>row.map((x,j)=>j===axis?b[i]:x)))/det);}
function spectralShape(mesh,local){
  const n=mesh.degree+1,rows=[],derivatives=[];
  for(const value of local){
    rows.push(mesh.gll.map((x,i)=>mesh.gll.reduce((p,y,j)=>i===j?p:p*(value-y)/(x-y),1)));
    derivatives.push(mesh.gll.map((x,i)=>mesh.gll.reduce((sum,other,j)=>i===j?sum:sum+mesh.gll.reduce((p,y,k)=>(k===i||k===j)?p:p*(value-y)/(x-y),1)/(x-other),0)));
  }
  const weights=[],gradient=[];
  for(let i=0;i<n;i++)for(let j=0;j<n;j++)for(let k=0;k<n;k++){weights.push(rows[0][i]*rows[1][j]*rows[2][k]);gradient.push([derivatives[0][i]*rows[1][j]*rows[2][k],rows[0][i]*derivatives[1][j]*rows[2][k],rows[0][i]*rows[1][j]*derivatives[2][k]]);}
  return {weights,gradient};
}
function spectralMap(result,element,local){
  const shape=spectralShape(result.spectral,local),point=[0,0,0],jacobian=Array.from({length:3},()=>[0,0,0]);
  result.spectral.elements[element].forEach((id,a)=>{const node=result.nodes_mm[id];for(let i=0;i<3;i++){point[i]+=shape.weights[a]*node[i];for(let j=0;j<3;j++)jacobian[i][j]+=node[i]*shape.gradient[a][j];}});
  return {...shape,point,jacobian};
}
const curvedBounds=new WeakMap();
function spectralCurvedLocation(result,point){
  let bounds=curvedBounds.get(result);
  if(!bounds){bounds=result.spectral.elements.map(ids=>[0,1,2].map(axis=>{const values=ids.map(id=>result.nodes_mm[id][axis]);return [Math.min(...values),Math.max(...values)];}));curvedBounds.set(result,bounds);}
  for(let e=0;e<bounds.length;e++){
    if(!point.every((v,a)=>v>=bounds[e][a][0]-.03*(bounds[e][a][1]-bounds[e][a][0])-1e-9&&v<=bounds[e][a][1]+.03*(bounds[e][a][1]-bounds[e][a][0])+1e-9))continue;
    let local=[0,0,0];
    for(let iteration=0;iteration<18;iteration++){
      const mapped=spectralMap(result,e,local),residual=mapped.point.map((v,a)=>v-point[a]);
      if(Math.hypot(...residual)<5e-13){if(local.every(v=>Math.abs(v)<=1+1e-8))return {element:e,weights:mapped.weights};break;}
      let delta;try{delta=solve3(mapped.jacobian,residual);}catch{break;}
      local=local.map((v,a)=>v-delta[a]);if(local.some(v=>!Number.isFinite(v)||Math.abs(v)>4))break;
    }
  }
  throw new Error('该点位于实体之外、管腔或孔洞内；不会自动吸附。');
}
