"""Optional real 7B loading/generation check; requires a supported CUDA GPU."""
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.run_mechanism_diagnostic import ImmutableGenerator,owned_generate
from src.deployment_probe_v1 import DeploymentScorer
from scripts.fetch_artifacts import sha

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    cfg=json.loads((ROOT/'configs/deployment_model.json').read_text(encoding='utf-8'))
    generator=ImmutableGenerator(cfg,a.gpu)
    result=owned_generate(generator,[{'role':'user','content':'Say ready.'}],[],max_new_tokens=8)
    scorer=DeploymentScorer(generator.tokenizer,generator.model,max_input_tokens=8192,model_binding=sha(ROOT/'configs/deployment_model.json'))
    contexts={v:'Select a tool.\nAction:' for v in ('original','observation_masked','user_masked')}
    measured=scorer.score(contexts,' list_files\n',' list_files\nAction Input: {}',alpha_name=1.,alpha_arg=2.,strategy='serial')
    assert measured['cost']['model_forward_calls']==3
    print(json.dumps({'model_revision':cfg['model_revision'],'generation':result['output'],'probe':measured['features'],'forward_calls':3,'scope':'loading/scoring smoke test, not a benchmark'}))
