"""Run an original sealed experiment without rewriting code, prompts, or parameters."""
import argparse, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.fetch_artifacts import sha

def command(experiment,output):
    root=ROOT/'artifacts/frozen'/experiment
    manifest=json.loads((ROOT/'resources/frozen_bundles.json').read_text(encoding='utf-8'))[experiment]
    for n,h in manifest.items():
        if not (root/n).is_file() or sha(root/n)!=h:raise ValueError('Missing/modified frozen input '+n+'; run scripts/fetch_artifacts.py --assets inputs')
    output=Path(output).resolve()
    if output.exists() or output.is_relative_to(root):raise ValueError('Choose a NEW output directory outside the sealed input root')
    modules={'v1':'src.run_bicp_parallel','mechanism':'src.run_mechanism_v4','calibration':'src.run_deployment_calibration_v1','pilot':'src.run_deployment_pilot_v2','longitudinal':'src.run_deployment_rl_v1'}
    args=[sys.executable,'-m',modules[experiment],'--root',str(root),'--output',str(output)]
    if experiment=='v1':args+=['--protocol',str(root/'configs/bicp_run_protocol.json'),'--phase','collect','--mode','full','--splits','test','--sources','injecagent,conditional','--calibration',str(root/'configs/bicp_frozen_calibration.json'),'--workers','2']
    return root,args

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=('v1','mechanism','calibration','pilot','longitudinal'),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    root,args=command(a.experiment,a.output)
    if a.check_only:print(json.dumps({'input_integrity':'passed','command':args,'cwd':str(root)}))
    else:raise SystemExit(subprocess.call(args,cwd=root))
