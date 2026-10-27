"""Two isolated T4 replicas; sequential Qwen/Mistral phases preserve VRAM."""
import argparse,copy,json,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
from .deployment_collection_v1 import read,write,rows,sha,CollectionScorer,diagnostic6,compare
from .deployment_rl_v1 import POLICY_SHA,MODEL_SHA
from .run_mechanism_diagnostic import ImmutableGenerator
from .run_baseline import environment
from .supplement_v2_2 import MistralReActTokenizer,contract_pair,mechanism_block,native_trajectory,score_one

def verify(root):
    b=read(root/'configs/supplement_binding.json')
    for n,h in b['file_sha256'].items():
        p=(root/n).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=h:raise ValueError('Bound supplement changed: '+n)
    if sha(root/'configs/frozen_policy.json')!=POLICY_SHA or sha(root/'configs/deployment_model.json')!=MODEL_SHA:raise ValueError('Formal policy/model changed')
    return b,read(root/'data/supplement/schedule.json')

def install_guard(root,b):
    allowed={(root/n).resolve() for n in b['file_sha256']}|{(root/'configs/supplement_binding.json').resolve()}
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        try:p=Path(args[0]).resolve()
        except (OSError,ValueError,TypeError):return
        if p.is_relative_to(root) and p not in allowed:
            if p.suffix=='.pyc' and '__pycache__' in p.parts:return
            raise PermissionError('Unbound runtime project read: '+p.name)
    sys.addaudithook(guard)

def worker(root,out,phase,gpu):
    b,schedule=verify(root);install_guard(root,b);out.mkdir(parents=True,exist_ok=False)
    name='deployment_model.json' if phase=='qwen' else 'mistral_model.json';cfg=read(root/'configs'/name)
    generator=ImmutableGenerator(copy.deepcopy(cfg),gpu)
    if phase=='mistral':generator.tokenizer=MistralReActTokenizer(generator.tokenizer)
    scorer=CollectionScorer(generator.tokenizer,generator.model,max_input_tokens=8192,model_binding=sha(root/'configs'/name))
    write(out/'environment.json',environment(cfg));write(out/'model.json',{'configuration':cfg,'gpu':gpu,
        'class':type(generator.model).__name__,'device_map':{str(k):str(v) for k,v in generator.model.hf_device_map.items()},
        'new_generation_calls_before_collection':0,'tokenizer_projection':'official-template-ReAct-tool-to-user-v1' if phase=='mistral' else 'original-Qwen-template'})
    # Verify separate six-forward and sliced three-forward paths per replica.
    candidate=next(x for x in schedule[phase] if x['kind']==('contract_pair' if phase=='qwen' else 'thought_replay'))
    item=candidate['items']['v3']['historical'] if phase=='qwen' else candidate['item']
    scored=score_one(item,scorer,prepare=phase=='qwen');contexts=scored['rendered_contexts']
    reference=diagnostic6(scorer,contexts,item['tool_name_target'],item['full_call_target'])
    health=compare(reference,scored['score'],{'token_logprob_max_abs':.1,'axis_feature_max_abs':.05})
    write(out/'healthcheck.json',{'precision':health,'reference':reference,'sliced':scored,'new_generation_calls':0})
    if not health['passed']:raise ValueError('Replica score precision failure')
    runtime=None
    if phase=='qwen':
        from .public_native_runtime_v2_2 import open_public_native
        runtime=open_public_native(root)
    policy=read(root/'configs/frozen_policy.json');suffix=read(root/'configs/suffix_policy.json');errors=0;count=0
    with (out/'records.jsonl').open('w',encoding='utf-8') as f:
        for block in schedule[phase][gpu::2]:
            try:
                if block['kind']=='contract_pair':r=contract_pair(block,scorer,policy,suffix)
                elif block['kind']=='native_trajectory':
                    def checkpoint(x):write(out/'checkpoints'/('block_'+str(block['block'])+'.json'),copy.deepcopy(x))
                    r=native_trajectory(block,generator,scorer,policy,runtime[block['suite']],root,checkpoint=checkpoint)
                else:r=mechanism_block(block,scorer)
            except Exception as e:errors+=1;r={'kind':block['kind'],'unit_id':block['unit_id'],'error':type(e).__name__+':'+str(e),'traceback':traceback.format_exc()}
            r.update(block=block['block'],phase=phase,worker=gpu,actual_model_measurement=True,
                execution_binding_sha256=sha(root/'configs/supplement_binding.json'),model_configuration_sha256=sha(root/'configs'/name))
            f.write(json.dumps(copy.deepcopy(r),ensure_ascii=False,allow_nan=False)+'\n');f.flush();count+=1
            write(out/'progress.json',{'recorded':count,'planned':len(schedule[phase][gpu::2]),'errors':errors,'last_block':block['block']})
            print(json.dumps({'phase':phase,'gpu':gpu,'block':block['block'],'kind':block['kind'],'error':r.get('error')}),flush=True)
    write(out/'completion.json',{'recorded':count,'record_errors':errors})
    if errors:raise RuntimeError('Errors retained, no replacement samples')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('qwen','mistral'));p.add_argument('--gpu',type=int,choices=(0,1));a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
    if a.phase:worker(root,out,a.phase,a.gpu);return
    b,s=verify(root)
    if out.exists():raise ValueError('Preserve prior supplement')
    out.mkdir(parents=True)
    for name in ('supplement_binding.json','deployment_model.json','mistral_model.json','frozen_policy.json','suffix_policy.json'):(out/name).write_bytes((root/'configs'/name).read_bytes())
    import torch
    if torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in (0,1)):raise ValueError('Authorized 2xT4 required')
    from huggingface_hub import snapshot_download
    records=[];exits={};phase_errors={}
    for phase,name in [('qwen','deployment_model.json'),('mistral','mistral_model.json')]:
        try:
            cfg=read(root/'configs'/name)
            snapshot_download(repo_id=cfg['model_id'],revision=cfg['model_revision'],allow_patterns=['*.json','model-*.safetensors','*.txt','tokenizer.model'])
            jobs=[subprocess.Popen([sys.executable,'-m','src.run_supplement_v2_2','--root',str(root),'--output',str(out/phase/f'worker_{i}'),'--phase',phase,'--gpu',str(i)],cwd=root) for i in (0,1)]
            exits[phase]=[p.wait() for p in jobs]
        except Exception as e:
            phase_errors[phase]={'error':type(e).__name__+':'+str(e),'traceback':traceback.format_exc()};write(out/(phase+'_error.json'),phase_errors[phase])
        for p in sorted((out/phase).glob('worker_*/records.jsonl')):records.extend(rows(p))
        write(out/'progress.json',{'last_phase':phase,'recorded':len(records),'worker_exit_codes':exits,'phase_errors':phase_errors})
    records.sort(key=lambda x:(x['phase'],x['block']))
    (out/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records),encoding='utf-8')
    complete=all(sorted(r['block'] for r in records if r['phase']==ph)==list(range(len(s[ph]))) for ph in ('qwen','mistral'))
    errors=sum('error' in r for r in records)
    write(out/'completion.json',{'collection_complete':complete,'record_errors':errors,'worker_exit_codes':exits,'phase_errors':phase_errors,
        'actual_model_measurement':True,'independently_accepted':False,'native_utility_evaluated':False,'thresholds_refitted_on_test':False,
        'formal_policy_unchanged':sha(root/'configs/frozen_policy.json')==POLICY_SHA,'actual_external_tool_executions':0,
        'completed_at_utc':datetime.now(timezone.utc).isoformat()})
    if not complete or errors or phase_errors or any(any(v) for v in exits.values()):raise RuntimeError('Incomplete supplement retained for independent acceptance')
if __name__=='__main__':main()
