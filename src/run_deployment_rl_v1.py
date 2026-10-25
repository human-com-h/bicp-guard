"""Two T4 replicas, frozen 922 longitudinal blocks and one native recovery task."""
import argparse,copy,json,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
from .deployment_rl_v1 import verify,Store,longitudinal_unit,workspace_unit,install_public_read_guard
from .deployment_collection_v1 import write,read,rows,sha,prepare_public,render,diagnostic6,compare
from .deployment_pilot_v1 import ARMS
from .run_deployment_calibration_v1 import model

def worker(root,out,gpu):
    binding,schedule,policy=verify(root);install_public_read_guard(root,binding);store=Store(root)
    out.mkdir(parents=True,exist_ok=False);generator,scorer=model(root,gpu,out);checks=[]
    for i,raw in enumerate(read(root/'data/rl/healthcheck_public.json')):
        item,prep=prepare_public(raw,scorer.tokenizer);contexts={k:v+'Action:' for k,v in render(scorer.tokenizer,item['views']).items()}
        ref=diagnostic6(scorer,contexts,item['tool_name_target'],item['full_call_target'])
        score=scorer.score(contexts,item['tool_name_target'],item['full_call_target'],alpha_name=1.,alpha_arg=2.,strategy='serial')
        precision=compare(ref,score,{'token_logprob_max_abs':.1,'axis_feature_max_abs':.05})
        write(out/'healthchecks'/f'proposal_{i:02d}.json',{'proposal_id':raw['proposal_id'],'reference':ref,'score':score,'precision':precision})
        checks.append(precision)
        if not precision['passed']:raise ValueError('Frozen serial path health precision failed')
    write(out/'healthchecks.json',{'passed':True,'checks':checks,'strategy':'serial_three_views','thresholds_refitted':False})
    errors=0;count=0
    with (out/'longitudinal_records.jsonl').open('w',encoding='utf-8') as f:
        for b in schedule[gpu::2]:
            try:r=longitudinal_unit(b,store,scorer,policy)
            except Exception as e:errors+=1;r={'unit_id':b['unit_id'],'schedule_block':b['block'],'error':type(e).__name__+':'+str(e),'traceback':traceback.format_exc(),'scope':'L'}
            r.update(worker_gpu=gpu,execution_binding_sha256=sha(root/'configs/rl_execution.json'),frozen_policy_sha256=sha(root/'configs/frozen_policy.json'),actual_model_measurement=True)
            f.write(json.dumps(copy.deepcopy(r),ensure_ascii=False,allow_nan=False)+'\n');f.flush();count+=1
            write(out/'progress.json',{'scope':'L','recorded':count,'planned':len(schedule[gpu::2]),'errors':errors,'last_block':b['block']})
            print(json.dumps({'scope':'L','worker':gpu,'block':b['block'],'error':r.get('error')}),flush=True)
    if gpu==0:
        from .public_workspace_runtime_v1 import open_public_workspace
        runtime=open_public_workspace(root)
        def checkpoint(snapshot):
            with (out/'workspace_checkpoints.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(copy.deepcopy(snapshot),ensure_ascii=False,allow_nan=False)+'\n');f.flush()
        try:r=workspace_unit(root,{'unit_id':'workspace:user_task_27','arm_order':list(ARMS)},store,generator,scorer,policy,runtime,checkpoint=checkpoint)
        except Exception as e:errors+=1;r={'scope':'R','unit_id':'workspace:user_task_27','error':type(e).__name__+':'+str(e),'traceback':traceback.format_exc()}
        r.update(worker_gpu=gpu,execution_binding_sha256=sha(root/'configs/rl_execution.json'),frozen_policy_sha256=sha(root/'configs/frozen_policy.json'),actual_model_measurement=True)
        write(out/'workspace_record.json',r)
    write(out/'completion.json',{'L_blocks':count,'R_units':int(gpu==0),'record_errors':errors})
    if errors:raise RuntimeError('R+L errors retained; no silent sample substitution')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,choices=(0,1));a=p.parse_args()
    root=a.root.resolve();out=a.output.resolve()
    if a.gpu is not None:worker(root,out,a.gpu);return
    binding,schedule,policy=verify(root)
    if out.exists():raise ValueError('Preserve prior R+L run')
    out.mkdir(parents=True)
    for name in ('rl_execution.json','deployment_model.json','frozen_policy.json'):(out/name).write_bytes((root/'configs'/name).read_bytes())
    (out/'schedule.json').write_bytes((root/'data/rl/schedule.json').read_bytes())
    import torch
    if torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in (0,1)):raise ValueError('Authorized two-T4 runtime required')
    from huggingface_hub import snapshot_download
    cfg=read(root/'configs/deployment_model.json');snapshot_download(repo_id=cfg['model_id'],revision=cfg['model_revision'],allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
    jobs=[subprocess.Popen([sys.executable,'-m','src.run_deployment_rl_v1','--root',str(root),'--output',str(out/'workers'/f'worker_{i}'),'--gpu',str(i)],cwd=root) for i in (0,1)]
    exits=[p.wait() for p in jobs];records=[]
    for path in sorted((out/'workers').glob('*/longitudinal_records.jsonl')):records.extend(rows(path))
    records.sort(key=lambda r:r['schedule_block'])
    (out/'longitudinal_records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records),encoding='utf-8')
    rp=out/'workers/worker_0/workspace_record.json'
    if rp.exists():(out/'workspace_record.json').write_bytes(rp.read_bytes())
    complete=[r['schedule_block'] for r in records]==list(range(922)) and rp.exists()
    errors=sum('error' in r for r in records)+int(not rp.exists() or 'error' in read(rp))
    write(out/'completion.json',{'collection_complete':complete,'L_blocks':len(records),'R_units':int(rp.exists()),'record_errors':errors,
        'worker_exit_codes':exits,'actual_model_measurement':True,'thresholds_refitted':False,'evaluation_labels_loaded':False,
        'actual_external_tool_executions':0,'frozen_policy_unchanged':sha(root/'configs/frozen_policy.json')==binding['frozen_policy_sha256'],
        'completed_at_utc':datetime.now(timezone.utc).isoformat()})
    if not complete or any(exits) or errors:raise RuntimeError('Incomplete R+L retained for independent acceptance')
if __name__=='__main__':main()
