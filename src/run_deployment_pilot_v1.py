"""Two isolated T4 replicas, fixed 28 paired units. No oracle/label fitting."""
import argparse,copy,json,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
from .deployment_pilot_v1 import Store,verify,static_unit,workspace_unit,install_public_read_guard
from .deployment_collection_v1 import write,read,rows,sha,prepare_public,render,diagnostic6,compare
from .run_deployment_calibration_v1 import model

def worker(root,output,gpu):
    binding,schedule,policy=verify(root);install_public_read_guard(root,binding);store=Store(root)
    output.mkdir(parents=True,exist_ok=False);generator,scorer=model(root,gpu,output)
    # Health checks use the ORIGINAL fixed calibration preflight proposals.
    # Only the frozen serial path is evaluated; no optimization or policy selection.
    checks=[]
    for i,raw in enumerate(read(root/'data/pilot/healthcheck_public.json')):
        item,prep=prepare_public(raw,scorer.tokenizer);contexts={k:v+'Action:' for k,v in render(scorer.tokenizer,item['views']).items()}
        name,full=item['tool_name_target'],item['full_call_target'];reference=diagnostic6(scorer,contexts,name,full)
        score=scorer.score(contexts,name,full,alpha_name=1.,alpha_arg=2.,strategy='serial')
        precision=compare(reference,score,{'token_logprob_max_abs':.1,'axis_feature_max_abs':.05})
        write(output/'healthchecks'/f'proposal_{i:02d}.json',{'proposal_id':raw['proposal_id'],'preparation':prep,'reference':reference,'score':score,'precision':precision})
        checks.append({'proposal_id':raw['proposal_id'],'precision':precision})
        if not precision['passed']:write(output/'healthchecks.json',{'passed':False,'checks':checks});raise ValueError('Frozen-path replica precision failed')
    write(output/'healthchecks.json',{'passed':True,'checks':checks,'strategy':'serial_three_views','policy_sha256':sha(root/'configs/frozen_policy.json')})
    runtime_parts=None;errors=0
    with (output/'records.jsonl').open('w',encoding='utf-8') as f:
        for block in schedule[gpu::2]:
            try:
                if block['source']=='workspace_development':
                    from .run_agentdojo_bicp import open_runtime
                    if runtime_parts is None:runtime_parts=open_runtime(root)
                    record=workspace_unit(block,store,generator,scorer,policy,runtime_parts)
                else:record=static_unit(block,store,generator,scorer,policy)
            except Exception as e:
                errors+=1;record={'schedule_block':block['block'],'unit_id':block['unit_id'],'source':block['source'],
                    'error':type(e).__name__+':'+str(e),'traceback':traceback.format_exc(),'actual_external_tool_executions':0}
            record.update(worker_gpu=gpu,execution_binding_sha256=sha(root/'configs/pilot_execution.json'),
                frozen_policy_sha256=sha(root/'configs/frozen_policy.json'),actual_model_measurement=True)
            f.write(json.dumps(copy.deepcopy(record),ensure_ascii=False,allow_nan=False)+'\n');f.flush()
            print(json.dumps({'gpu':gpu,'block':block['block'],'error':record.get('error')}),flush=True)
    write(output/'completion.json',{'record_errors':errors,'recorded_blocks':len(schedule[gpu::2])})
    if errors:raise RuntimeError('Pilot errors preserved; no silent unit substitution')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,choices=(0,1));a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
    if a.gpu is not None:worker(root,out,a.gpu);return
    binding,schedule,policy=verify(root)
    if out.exists():raise ValueError('Preserve prior pilot run')
    out.mkdir(parents=True)
    for name in ('pilot_execution.json','deployment_model.json','frozen_policy.json'):(out/name).write_bytes((root/'configs'/name).read_bytes())
    (out/'schedule_public.json').write_bytes((root/'data/pilot/schedule_public.json').read_bytes())
    import torch
    if torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):raise ValueError('Authorized two T4s required')
    from huggingface_hub import snapshot_download
    m=read(root/'configs/deployment_model.json');snapshot_download(repo_id=m['model_id'],revision=m['model_revision'],allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
    jobs=[subprocess.Popen([sys.executable,'-m','src.run_deployment_pilot_v1','--root',str(root),'--output',str(out/'workers'/f'worker_{i}'),'--gpu',str(i)],cwd=root) for i in range(2)]
    exits=[p.wait() for p in jobs];records=[]
    for path in sorted((out/'workers').glob('*/records.jsonl')):records.extend(rows(path))
    records.sort(key=lambda r:r['schedule_block'])
    (out/'records.jsonl').write_text(''.join(json.dumps(copy.deepcopy(r),ensure_ascii=False,allow_nan=False)+'\n' for r in records),encoding='utf-8')
    complete=[r['schedule_block'] for r in records]==list(range(28))
    write(out/'completion.json',{'collection_complete':complete,'blocks':len(records),'record_errors':sum('error' in r for r in records),
        'worker_exit_codes':exits,'actual_model_measurement':True,'frozen_policy_unchanged':sha(root/'configs/frozen_policy.json')==binding['frozen_policy_sha256'],
        'thresholds_refitted':False,'actual_external_tool_executions':0,'evaluation_labels_loaded':False,'completed_at_utc':datetime.now(timezone.utc).isoformat()})
    if not complete or any(exits):raise RuntimeError('Incomplete pilot preserved for independent acceptance')

if __name__=='__main__':main()
