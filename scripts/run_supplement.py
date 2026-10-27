"""Verify the accepted supplementary input archive; launch only when requested."""
import argparse,hashlib,json,subprocess,sys,zipfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
def verify():
    manifest=json.loads((ROOT/'resources/supplement_v2_2_manifest.json').read_text(encoding='utf-8'))
    archive=ROOT/'resources/supplement_v2_2_inputs.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['artifacts'][archive.relative_to(ROOT).as_posix()]
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())==set(manifest['frozen_input_files'])
        for name,h in manifest['frozen_input_files'].items():
            p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts
            assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    return archive,manifest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--input-dir',type=Path,default=ROOT/'artifacts/frozen/supplement_v2_2');a=p.parse_args()
    archive,m=verify()
    if a.check_only:print(json.dumps({'passed':True,'frozen_files':len(m['frozen_input_files']),'model_execution':False}));raise SystemExit(0)
    if a.output is None:p.error('--output is required for a new collection')
    output=a.output.resolve();base=a.input_dir.resolve()
    if output.exists() or output.is_relative_to(base):p.error('Choose a new output directory outside the sealed input directory')
    if not base.exists():
        with zipfile.ZipFile(archive) as z:z.extractall(base)
    for name,h in m['frozen_input_files'].items():
        if not (base/name).is_file() or hashlib.sha256((base/name).read_bytes()).hexdigest()!=h:raise ValueError('Modified frozen input: '+name)
    # The original runner enforces exactly two T4s and pinned model revisions.
    raise SystemExit(subprocess.call([sys.executable,'-m','src.run_supplement_v2_2','--root',str(base),'--output',str(output)],cwd=base))
