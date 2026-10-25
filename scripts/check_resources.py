"""Check official model metadata and immutable artifact download availability."""
import argparse,json,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.fetch_artifacts import fetch

def check(download=False):
    config=json.loads((ROOT/'configs/deployment_model.json').read_text(encoding='utf-8'))
    url=f"https://huggingface.co/{config['model_id']}/resolve/{config['model_revision']}/config.json"
    with urllib.request.urlopen(url,timeout=60) as r:metadata=json.load(r)
    assert metadata.get('model_type')=='qwen2'
    result={'official_model_config_accessible':True,'model_revision':config['model_revision'],'weights_downloaded':False}
    if download:fetch('all');result['release_archives_verified']=True
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');a=p.parse_args();check(a.download)
