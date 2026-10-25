"""CPU/CUDA causal scoring equivalence with a tiny deterministic toy model."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.deployment_probe_v1 import DeploymentScorer,axis_features
from src.mechanism_probe import SequenceScorer,TorchBackend

def run():
    import torch
    torch.manual_seed(42);torch.set_num_threads(2)
    class Tokenizer:
        pad_token_id=0
        def encode(self,text,**kwargs):return [ord(c)%31+1 for c in text]
        def __call__(self,text,**kwargs):return {'input_ids':self.encode(text),'offset_mapping':[(i,i+1) for i in range(len(text))]}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embedding=torch.nn.Embedding(32,16);self.head=torch.nn.Linear(16,32)
        def get_input_embeddings(self):return self.embedding
        def forward(self,input_ids,attention_mask=None,**kwargs):
            values=self.embedding(input_ids)
            if attention_mask is not None:values=values*attention_mask[:,:,None]
            return SimpleNamespace(logits=self.head(values.cumsum(dim=1)))
    template=Model().eval();state=template.state_dict();result=[]
    for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
        model=Model();model.load_state_dict(state);model.to(device).eval();tokenizer=Tokenizer()
        scorer=DeploymentScorer(tokenizer,model);reference=SequenceScorer(tokenizer,TorchBackend(model))
        contexts={v:'Trusted task '+v+'\nAction:' for v in ('original','observation_masked','user_masked')};name=' Send\n';full=name+'Action Input: {"to":"office"}'
        baseline={kind:reference.score(contexts,target) for kind,target in [('name',name),('full',full)]}
        scores={v:{k:baseline[k]['views'][v] for k in baseline} for v in contexts}
        sliced=scorer.score(contexts,name,full,alpha_name=1.,alpha_arg=2.,strategy='serial');errors=[]
        for v in contexts:
            for k in ('name','full'):errors.extend(abs(a-b) for a,b in zip(scores[v][k]['token_logprobs'],sliced['scores'][v][k]['token_logprobs']))
        assert max(errors)<1e-5 and sliced['cost']['model_forward_calls']==3
        expected=axis_features(scores,1.,2.)
        assert all(abs(expected[k]-sliced['features'][k])<1e-5 for k in ('Risk_name','DeltaRisk_arg'))
        result.append({'device':device,'token_checks':len(errors),'max_abs_error':max(errors),'passed':True})
    return {'passed':True,'torch':torch.__version__,'cuda_build':torch.version.cuda,'checks':result,'generation_seed':42,'scope':'software numeric equivalence only; no 7B weights, NF4 benchmark or latency result'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args();result=run()
    if a.output:
        if a.output.exists():raise ValueError('Preserve old verification')
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
