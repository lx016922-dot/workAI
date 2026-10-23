"""Create a source-only archive while retaining every previously shipped path."""
import argparse,hashlib,json,zipfile
from pathlib import Path


def main(a):
 repo=Path(__file__).resolve().parents[1];previous=json.loads(a.previous.read_text());required={r['path'] for r in previous['files_manifest']};paths=set(required)
 folders=['data','docs','modal3d','operator3d','public','server','solid','spectral3d','spectralop','tests','tools','training','waveguide']
 extensions={'.py','.js','.mjs','.json','.md','.html','.css','.svg','.sh','.ps1','.java','.txt','.xml'}
 for folder in folders:
  for p in (repo/folder).rglob('*'):
   if p.is_file() and p.suffix.lower() in extensions and not any(part in ['__pycache__','.pytest_cache','node_modules','venv','.venv'] for part in p.relative_to(repo).parts):paths.add(str(p.relative_to(repo)))
 records=[]
 with zipfile.ZipFile(a.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for relative in sorted(paths):
   path=repo/relative
   if not path.is_file() or path.is_symlink() or any(part in ['private','.git','.codex','.agents'] for part in Path(relative).parts) or 'private-key' in relative or 'client-private' in relative:raise ValueError('Invalid source path: '+relative)
   raw=path.read_bytes();z.writestr(relative,raw);records.append({'path':relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
 result={'archive':str(a.output),'files':len(records),'bytes':a.output.stat().st_size,'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'all_previous_paths_preserved':required.issubset(paths),'previous_path_count':len(required),'source_only':True,'files_manifest':records}
 with a.manifest.open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k!='files_manifest'}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);main(p.parse_args())
