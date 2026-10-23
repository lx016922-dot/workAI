"""Only fixed JSON operations for the local web service."""
import argparse,json,sys
from pathlib import Path
from .schema import default_scene,validate_scene

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['templates','validate','validate-study','inspect']);parser.add_argument('--root',default='/workspace/workai-remote');args=parser.parse_args()
    try:
        if args.action=='templates':value={k:default_scene(k) for k in ['plate','pipe','hole','notch']}
        elif args.action=='validate':value=validate_scene(json.load(sys.stdin))
        elif args.action=='validate-study':
            from .study import validate_study
            value=validate_study(json.load(sys.stdin))
        else:
            import gmsh
            from .cad import read_asset,import_cad
            from .schema import exact,number
            data=json.load(sys.stdin);exact(data,['asset_id','stl_unit_mm'],'几何检查')
            scale=number(data['stl_unit_mm'],.001,1000,'STL 单位比例');path,metadata=read_asset(args.root,data['asset_id'])
            gmsh.initialize(['workai','-nopopup']);gmsh.option.setNumber('General.Terminal',0)
            try:value={**metadata,'bounds_mm':import_cad(gmsh,path,metadata,scale),'stl_unit_mm':scale}
            finally:gmsh.finalize()
        print(json.dumps({'ok':True,'value':value},ensure_ascii=False,allow_nan=False))
    except Exception as error:
        print(json.dumps({'ok':False,'error':str(error)},ensure_ascii=False));sys.exit(2)
