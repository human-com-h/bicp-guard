"""Download and safely extract immutable, checksummed research artifacts."""
import argparse, hashlib, json, os, shutil, stat, sys, urllib.request, zipfile
from pathlib import Path, PurePosixPath
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def extract(path,asset,destination):
    destination=Path(destination).resolve()
    with zipfile.ZipFile(path) as z:
        if set(z.namelist())!=set(asset['files']) or len(z.namelist())!=len(asset['files']):raise ValueError('Unexpected archive members')
        for info in z.infolist():
            n=PurePosixPath(info.filename)
            if n.is_absolute() or '..' in n.parts or '\\' in info.filename or ':' in info.filename or stat.S_ISLNK(info.external_attr>>16):raise ValueError('Unsafe archive path')
            target=destination.joinpath(*n.parts).resolve()
            if not target.is_relative_to(destination):raise ValueError('Archive escapes destination')
            b=z.read(info);metadata=asset['files'][info.filename]
            if hashlib.sha256(b).hexdigest()!=metadata['sha256'] or len(b)!=metadata['bytes']:raise ValueError('Archive member checksum mismatch')
            if target.exists() and sha(target)!=metadata['sha256']:raise ValueError('Preserve modified artifact: '+str(target))
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():target.write_bytes(b)

def fetch(assets='all',local_dir=None):
    manifest=json.loads((ROOT/'resources/artifacts.json').read_text(encoding='utf-8'))
    cache=ROOT/'artifacts/cache';cache.mkdir(parents=True,exist_ok=True)
    selected=[a for a in manifest['assets'] if assets=='all' or ('inputs' in a['name'])==(assets=='inputs')]
    for a in selected:
        path=cache/a['name']
        if not path.exists():
            temporary=path.with_suffix('.partial')
            if local_dir:shutil.copyfile(Path(local_dir)/a['name'],temporary)
            else:
                req=urllib.request.Request(a['url'],headers={'User-Agent':'BiCP-artifact-fetcher'})
                with urllib.request.urlopen(req,timeout=90) as source,temporary.open('wb') as target:shutil.copyfileobj(source,target)
            if sha(temporary)!=a['sha256']:raise ValueError('Downloaded archive checksum mismatch')
            os.replace(temporary,path)
        if path.stat().st_size!=a['bytes'] or sha(path)!=a['sha256']:raise ValueError('Cached archive checksum mismatch')
        destination=ROOT/'artifacts'/('frozen' if 'inputs' in a['name'] else 'runs')
        extract(path,a,destination)
        print(json.dumps({'asset':a['name'],'sha256':a['sha256'],'files':len(a['files']),'verified':True}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',choices=('all','inputs','runs'),default='all');p.add_argument('--local-dir',type=Path);a=p.parse_args();fetch(a.assets,a.local_dir)
