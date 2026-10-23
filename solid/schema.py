"""Strict, numeric-only solid input. All user geometry lengths are millimetres."""
import copy
import math
import re

MATERIAL = {'rho_kg_m3':2700.,'lambda_pa':58.1e9,'mu_pa':26.1e9,'l_pa':-143e9,'m_pa':-297e9,'n_pa':-345e9}

def require(ok, message):
    if not ok: raise ValueError(message)

def number(value, lo, hi, name):
    require(type(value) in (int,float) and math.isfinite(value) and lo<=value<=hi, f'{name} 必须在 {lo}–{hi} 范围内。')
    return float(value)

def vector(value, name, unit=False):
    require(isinstance(value,list) and len(value)==3, f'{name} 需要三个数值。')
    v=[number(x,-10000,10000,name) for x in value]
    if unit:
        norm=math.sqrt(sum(x*x for x in v));require(norm>1e-12,f'{name} 不能为零向量。')
        if abs(norm-1)>1e-12:v=[x/norm for x in v]
    return v

def exact(obj, keys, name):
    require(isinstance(obj,dict) and set(obj)==set(keys),f'{name} 字段不正确。')

def default_scene(kind='plate'):
    geometry={'kind':kind,'length_mm':12.,'width_mm':4.,'thickness_mm':2.,'outer_radius_mm':3.,'inner_radius_mm':2.,'hole_radius_mm':.7,'hole_x_mm':6.,'hole_y_mm':2.,'notch_x_mm':6.,'notch_width_mm':1.,'notch_depth_mm':.6}
    return {'schema':'workai.hyperelastic.solid.v1','name':{'plate':'铝板','pipe':'铝管','hole':'带孔铝板','notch':'缺口铝板'}.get(kind,'铝结构'),
            'geometry':geometry,'material':copy.deepcopy(MATERIAL),'constitutive':'murnaghan',
            'source':{'face':'xmin','center_mm':[0.,2.,1.],'radius_mm':0.,'direction':[1.,0.,0.],'amplitude_mpa':1.,'frequency_mhz':.2,'cycles':2.,'delay_us':0.,'phase_deg':0.},
            'boundary':{'fixed_face':'none'},'acquisition':{'duration_us':14.,'sample_step_us':.05},
            'numerics':{'h_mm':.7,'cfl':.3,'max_elements':40000},
            'receivers':[{'name':'P1','point_mm':[3.,2.,1.],'direction':[1.,0.,0.]},{'name':'P2','point_mm':[9.,2.,1.],'direction':[1.,1.,0.]}]}

def validate_scene(value):
    s=copy.deepcopy(value)
    exact(s,['schema','name','geometry','material','constitutive','source','boundary','acquisition','numerics','receivers'],'场景')
    require(s['schema']=='workai.hyperelastic.solid.v1','场景版本不支持。')
    require(isinstance(s['name'],str) and 1<=len(s['name'])<=100,'名称需要 1–100 个字符。')
    require(s['constitutive'] in ['murnaghan','stvk','linear'],'材料本构不支持。')
    g=s['geometry'];geometry_keys=list(default_scene()['geometry'])
    if g.get('kind')=='cad':geometry_keys+=['asset_id','stl_unit_mm']
    exact(g,geometry_keys,'几何')
    require(g['kind'] in ['plate','pipe','hole','notch','cad'],'几何类型不支持。')
    for key in g:
        if key not in ['kind','asset_id']:g[key]=number(g[key],.001,1000,key)
    if g['kind']=='cad':require(isinstance(g['asset_id'],str) and re.fullmatch(r'asset-[a-f0-9]{64}',g['asset_id']) is not None,'导入模型编号无效。')
    require(g['inner_radius_mm']<g['outer_radius_mm'],'管内半径必须小于外半径。')
    if g['kind']=='hole':
        r=g['hole_radius_mm'];require(r<g['hole_x_mm']<g['length_mm']-r and r<g['hole_y_mm']<g['width_mm']-r,'通孔必须完整位于板内。')
    if g['kind']=='notch':
        require(g['notch_width_mm']/2<g['notch_x_mm']<g['length_mm']-g['notch_width_mm']/2,'缺口不能越过端面。')
        require(g['notch_depth_mm']<g['thickness_mm'],'缺口深度必须小于板厚。')
    exact(s['material'],MATERIAL,'材料')
    # This version uses the sourced aluminum card, not arbitrary unknown constants.
    for k,v in MATERIAL.items():require(number(s['material'][k],-1e12,1e12,k)==v,'本版本采用已记录的多晶铝文献常数。')
    tx=s['source'];tx_keys=['face','center_mm','radius_mm','direction','amplitude_mpa','frequency_mhz','cycles','delay_us','phase_deg']
    if 'patch_normal' in tx:tx_keys+=['patch_normal']
    exact(tx,tx_keys,'激励')
    require(tx['face'] in ['xmin','xmax','ymin','ymax','zmin','zmax','surface_patch'],'激励面无效。')
    if 'patch_normal' in tx:tx['patch_normal']=vector(tx['patch_normal'],'加载区域外法线',True)
    tx['center_mm']=vector(tx['center_mm'],'激励中心');tx['direction']=vector(tx['direction'],'激励方向',True)
    for k,lo,hi in [('radius_mm',0,1000),('amplitude_mpa',.00001,10),('frequency_mhz',.001,10),('cycles',1,20),('delay_us',0,1000),('phase_deg',-360,360)]:tx[k]=number(tx[k],lo,hi,k)
    if tx['face']=='surface_patch':require(tx['radius_mm']>0 and 'patch_normal' in tx,'曲面激励需要正的孔径半径和区域外法线。')
    exact(s['boundary'],['fixed_face'],'边界');require(s['boundary']['fixed_face'] in ['none','xmin','xmax','ymin','ymax','zmin','zmax'],'固定面无效。')
    require(s['boundary']['fixed_face']!=tx['face'],'激励面与固定面不能相同。')
    acq=s['acquisition'];exact(acq,['duration_us','sample_step_us'],'采样')
    acq['duration_us']=number(acq['duration_us'],.1,1000,'时长');acq['sample_step_us']=number(acq['sample_step_us'],.001,10,'采样间隔')
    ratio=acq['duration_us']/acq['sample_step_us'];require(abs(ratio-round(ratio))<1e-7 and 20<=round(ratio)<=2000,'时长必须为采样间隔的整数倍，20–2000 个间隔。')
    require(acq['sample_step_us']*tx['frequency_mhz']<=.1,'基频至少需要 10 个输出采样点/周期。')
    require(tx['delay_us']+tx['cycles']/tx['frequency_mhz']<=acq['duration_us'],'计算时长需要覆盖完整激励脉冲。')
    n=s['numerics'];exact(n,['h_mm','cfl','max_elements'],'数值设置')
    n['h_mm']=number(n['h_mm'],.02,100,'网格尺寸');n['cfl']=number(n['cfl'],.05,.35,'CFL 系数')
    require(type(n['max_elements']) is int and 100<=n['max_elements']<=40000,'单元预算为 100–40000。')
    ct=math.sqrt(MATERIAL['mu_pa']/MATERIAL['rho_kg_m3'])
    require(n['h_mm']<=ct/(tx['frequency_mhz']*1e6*2*8)*1000,'网格需满足二次谐波剪切波长至少 8 个单元的初始筛选；仍需网格收敛验证。')
    require(isinstance(s['receivers'],list) and 1<=len(s['receivers'])<=32,'需要 1–32 个初始接收点。')
    for rx in s['receivers']:
        exact(rx,['name','point_mm','direction'],'接收点');require(isinstance(rx['name'],str) and 1<=len(rx['name'])<=60,'接收点名称无效。')
        rx['point_mm']=vector(rx['point_mm'],'接收坐标');rx['direction']=vector(rx['direction'],'接收方向',True)
    return s
