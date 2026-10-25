"""Two-T4 public-only calibration: 12 preflight proposals then 390 blocks."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback

from .deployment_collection_v1 import (read,write,rows,sha,digest,verify_bindings,PublicStore,CollectionScorer,
    preflight_one,choose_strategy,collect_block)
from .run_mechanism_diagnostic import ImmutableGenerator,owned_generate
from .run_baseline import environment

def install_public_read_guard(root,binding):
    root=Path(root).resolve();allowed={(root/p).resolve() for p in binding['file_sha256']}|{(root/'configs/deployment_execution.json').resolve()}
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        try:path=Path(args[0]).resolve()
        except (OSError,ValueError,TypeError):return
        if path.is_relative_to(root) and path not in allowed:
            if path.suffix=='.pyc' and '__pycache__' in path.parts:return
            raise PermissionError('Runtime cannot read unbound/evaluation project data: '+path.name)
    sys.addaudithook(guard)

def model(root,gpu,output):
    config=read(root/'configs/deployment_model.json');generator=ImmutableGenerator(copy.deepcopy(config),gpu)
    scorer=CollectionScorer(generator.tokenizer,generator.model,max_input_tokens=config['max_input_tokens'],model_binding=sha(root/'configs/deployment_model.json'))
    write(output/'environment.json',environment(config))
    write(output/'model.json',{'model_configuration_sha256':sha(root/'configs/deployment_model.json'),
        'model_revision':config['model_revision'],'model_class':type(generator.model).__name__,
        'device_map':{str(k):str(v) for k,v in generator.model.hf_device_map.items()},'gpu':gpu,
        'worker_device_name':generator.torch.cuda.get_device_name(gpu),'labels_loaded':False})
    write(output/'warmup.json',owned_generate(generator,[{'role':'user','content':'Say ready.'}],[],max_new_tokens=3))
    return generator,scorer

def preflight(root,output):
    binding,_=verify_bindings(root);install_public_read_guard(root,binding);store=PublicStore(root)
    output.mkdir(parents=True,exist_ok=False);generator,scorer=model(root,0,output);results=[]
    for index,identity in enumerate(store.preflight['proposal_ids']):
        result=preflight_one(store.proposal(identity),scorer,store.preflight)
        result.update(worker_gpu=0,actual_model_measurement=True,binding_sha256=sha(root/'configs/deployment_execution.json'))
        write(output/f'proposal_{index:02d}.json',result);results.append(result)
        print(json.dumps({'phase':'preflight','proposal':index,'errors':result['errors']}),flush=True)
    selection=choose_strategy(results,store.preflight)
    selection.update(binding_sha256=sha(root/'configs/deployment_execution.json'),model_configuration_sha256=sha(root/'configs/deployment_model.json'),
        completed_at_utc=datetime.now(timezone.utc).isoformat(),actual_model_measurement=True,labels_loaded=False)
    selection['selection_sha256']=digest(selection);write(output/'selection.json',selection)

def worker(root,output,gpu,selection_path):
    binding,schedule=verify_bindings(root);install_public_read_guard(root,binding)
    # Selection is an output, outside the sealed extracted project root.
    selection=read(selection_path);saved=selection['selection_sha256'];base={k:v for k,v in selection.items() if k!='selection_sha256'}
    if digest(base)!=saved or not selection['all_primary_precision_passed'] or selection['binding_sha256']!=sha(root/'configs/deployment_execution.json'):raise ValueError('Preflight selection binding failed')
    store=PublicStore(root);output.mkdir(parents=True,exist_ok=False);generator,scorer=model(root,gpu,output)
    # Both model replicas verify the chosen THREE-view path before collecting.
    checks=[]
    for index,identity in enumerate(store.preflight['proposal_ids']):
        result=preflight_one(store.proposal(identity),scorer,store.preflight,timing=False)
        write(output/'replica_preflight'/f'proposal_{index:02d}.json',result)
        precision=result['collection3'][selection['strategy']]['precision']
        checks.append({'proposal_id':identity,'precision':precision,'errors':result['errors']})
        if not precision['passed']:write(output/'replica_preflight.json',{'passed':False,'checks':checks});raise ValueError('Replica precision check failed')
    write(output/'replica_preflight.json',{'passed':True,'checks':checks,'selection_sha256':saved})
    count=errors=0
    with (output/'records.jsonl').open('w',encoding='utf-8') as stream:
        for block in schedule[gpu::2]:
            try:record=collect_block(block,store,generator,scorer,selection['strategy'])
            except Exception as error:
                errors+=1
                record={'schedule_block':block['block'],'kind':block['kind'],'error_type':type(error).__name__,'error':str(error),
                    'traceback':traceback.format_exc(),'labels_loaded':False,'actual_external_tool_executions':0}
            record.update(worker_gpu=gpu,binding_sha256=sha(root/'configs/deployment_execution.json'),
                selection_sha256=saved,actual_model_measurement=True)
            stream.write(json.dumps(copy.deepcopy(record),ensure_ascii=False,allow_nan=False)+'\n');stream.flush();count+=1
            write(output/'progress.json',{'recorded':count,'planned':len(schedule[gpu::2]),'errors':errors,'last_block':block['block']})
            print(json.dumps({'phase':'collection','gpu':gpu,'block':block['block'],'error':record.get('error')}),flush=True)
    write(output/'completion.json',{'collection_complete':count==len(schedule[gpu::2]),'records':count,'record_errors':errors,'labels_loaded':False})
    if errors:raise RuntimeError('Collection errors retained; independent acceptance required')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('preflight','worker'));parser.add_argument('--gpu',type=int,choices=(0,1));parser.add_argument('--selection',type=Path)
    a=parser.parse_args();root=a.root.resolve();output=a.output.resolve()
    if a.phase=='preflight':preflight(root,output);return
    if a.phase=='worker':worker(root,output,a.gpu,a.selection);return
    binding,schedule=verify_bindings(root)
    if output.exists():raise ValueError('Preserve previous collection')
    output.mkdir(parents=True)
    for name in ('deployment_execution.json','deployment_model.json','deployment_protocol.json'):(output/name).write_bytes((root/'configs'/name).read_bytes())
    (output/'calibration_execution_schedule.json').write_bytes((root/'data/deployment_calibration/calibration_execution_schedule.json').read_bytes())
    import torch
    if torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):raise ValueError('Authorized two-T4 environment required')
    from huggingface_hub import snapshot_download
    config=read(root/'configs/deployment_model.json')
    snapshot_download(repo_id=config['model_id'],revision=config['model_revision'],allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
    command=[sys.executable,'-m','src.run_deployment_calibration_v1','--root',str(root)]
    result=subprocess.run(command+['--output',str(output/'preflight'),'--phase','preflight'],cwd=root)
    if result.returncode:
        write(output/'completion.json',{'collection_complete':False,'stage':'preflight_failed','records':0,'exit_code':result.returncode,'thresholds_fitted':False})
        raise RuntimeError('Preflight failed; no 390-block collection started')
    selection=output/'preflight/selection.json'
    jobs=[subprocess.Popen(command+['--output',str(output/'workers'/f'worker_{i}'),'--phase','worker','--gpu',str(i),'--selection',str(selection)],cwd=root) for i in range(2)]
    exits=[p.wait() for p in jobs];records=[]
    for path in sorted((output/'workers').glob('worker_*/records.jsonl')):records.extend(rows(path))
    records.sort(key=lambda r:r['schedule_block'])
    with (output/'records.jsonl').open('w',encoding='utf-8') as stream:
        for r in records:stream.write(json.dumps(copy.deepcopy(r),ensure_ascii=False,allow_nan=False)+'\n')
    complete=[r['schedule_block'] for r in records]==list(range(390))
    write(output/'completion.json',{'collection_complete':complete,'planned_blocks':390,'recorded_blocks':len(records),'record_errors':sum('error' in r for r in records),
        'worker_exit_codes':exits,'actual_model_measurement':True,'labels_loaded':False,'thresholds_fitted':False,'independently_accepted':False,
        'actual_external_tool_executions':0,'completed_at_utc':datetime.now(timezone.utc).isoformat()})
    if not complete or any(exits):raise RuntimeError('Incomplete/error collection; preserve partial outputs')

if __name__=='__main__':main()
