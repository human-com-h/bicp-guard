"""Fail closed on manuscripts, credentials, unreviewed documents and unexpected files."""
import argparse,hashlib,io,json,re,subprocess,sys,zipfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
TOKEN_PATTERNS=[rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'github_pat_[A-Za-z0-9_]{40,}',rb'hf_[A-Za-z0-9]{30,}',rb'KGAT_[A-Za-z0-9_]{20,}',rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']

def inspect(name,data,policy):
    issues=[];parts=PurePosixPath(name.replace('\\','/')).parts;lower=name.lower()
    if any(p.lower() in ('paper','manuscript','submission','review_response','分析汇总') for p in parts):issues.append('manuscript/research-only path')
    if Path(name).name.lower() in ('main.pdf','main.tex','preprint_zh.md','arxiv_source.tar.gz','kaggle.json','.env'):issues.append('forbidden filename')
    if hashlib.sha256(data).hexdigest() in {r['sha256'] for r in policy['forbidden_fingerprints']}:issues.append('known internal manuscript fingerprint')
    if any(re.search(p,data) for p in TOKEN_PATTERNS):issues.append('credential/private-key pattern')
    if (b'\\document'+b'class') in data or (b'\\begin'+b'{abstract}') in data:issues.append('full LaTeX manuscript content')
    if lower.endswith(('.pdf','.tex','.docx','.doc','.bbl','.bib')) and name not in policy['allowed_document_paths']:issues.append('document format needs explicit path/content approval')
    if data.startswith(b'%PDF') and name not in policy['allowed_document_paths']:issues.append('unreviewed PDF payload')
    if lower.endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for n in z.namelist():issues.extend(name+'!'+n+': '+x for x in inspect(n,z.read(n),policy))
    return issues

def audit(history=False,asset_dir=None):
    policy=json.loads((ROOT/'resources/publication_policy.json').read_text(encoding='utf-8'));issues=[];count=0
    manifest_path=ROOT/'resources/release_files.json'
    tracked=subprocess.run(['git','ls-files','-z'],cwd=ROOT,capture_output=True)
    files=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.relative_to(ROOT).parts for x in ('.git','artifacts','outputs','__pycache__','.venv','.pytest_cache'))]
    if manifest_path.exists():
        expected=json.loads(manifest_path.read_text(encoding='utf-8'))
        actual={p.relative_to(ROOT).as_posix() for p in files if p!=manifest_path}
        if actual!=set(expected):issues.append('reviewed file list differs: '+str(sorted(actual^set(expected))))
        for n,h in expected.items():
            if (ROOT/n).is_file() and hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h:issues.append('reviewed bytes changed: '+n)
    for p in files:
        name=p.relative_to(ROOT).as_posix();issues.extend(name+': '+v for v in inspect(name,p.read_bytes(),policy));count+=1
    if history:
        result=subprocess.run(['git','rev-list','--objects','--all'],cwd=ROOT,capture_output=True,text=True,check=True)
        for line in result.stdout.splitlines():
            obj,_,name=line.partition(' ')
            kind=subprocess.check_output(['git','cat-file','-t',obj],cwd=ROOT,text=True).strip()
            if kind=='blob':
                data=subprocess.check_output(['git','cat-file','blob',obj],cwd=ROOT);issues.extend('history '+obj+' '+name+': '+v for v in inspect(name,data,policy));count+=1
    if asset_dir:
        for asset in json.loads((ROOT/'resources/artifacts.json').read_text(encoding='utf-8'))['assets']:
            p=Path(asset_dir)/asset['name'];data=p.read_bytes()
            if hashlib.sha256(data).hexdigest()!=asset['sha256']:issues.append('asset checksum mismatch '+p.name)
            issues.extend(p.name+': '+v for v in inspect(p.name,data,policy));count+=1
    result={'passed':not issues,'items_checked':count,'history_checked':history,'release_assets_checked':asset_dir is not None,'issues':issues}
    print(json.dumps(result,indent=2,ensure_ascii=False));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--history',action='store_true');p.add_argument('--asset-dir',type=Path);a=p.parse_args();raise SystemExit(0 if audit(a.history,a.asset_dir)['passed'] else 1)
