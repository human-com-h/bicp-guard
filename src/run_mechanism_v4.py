"""Fixed development-only v4 collection; independent labels attach after scoring."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import traceback

from .mechanism_probe import SequenceScorer, TorchBackend, json_segments, original_prefix, semantic_sha
from .mechanism_v3 import extract_proposal_v3, parse_call_v3
from .mechanism_v4 import (PREFIX_LEVELS, branch_prefixes, neutral_thought,
    two_dimensional_features, tool_layer, truncate_observations)
from .probe import VIEW_NAMES
from .run_baseline import environment, write_json
from .run_bicp import counterfactual_messages
from .run_mechanism_diagnostic import ImmutableGenerator, owned_generate, error_record, write_record
from .run_mechanism_v3 import read, read_lines, sha, safe_score

PANELS={'parameter_factorial':10,'thought_replay':69,'native_discovery_replay':7,
    'actual_malicious_proposal':1,'workspace_truncated_continuation':1,'conditional_free_smoke':6}
DATA='data/mechanism_v4/'


def render(tokenizer, views):
    if set(views)!=set(VIEW_NAMES):raise ValueError('Three exact views required')
    return {k:tokenizer.apply_chat_template(v,tokenize=False,add_generation_prompt=True) for k,v in views.items()}


def segments(target,kind):
    if kind=='tool_name':return [{'start':0,'end':len(target),'kind':'tool_identifier'}]
    offset=target.index('Action Input:')+len('Action Input:')
    return [{'start':0,'end':offset,'kind':'call_syntax'}]+json_segments(target[offset:],offset)


def score_payload(item,tokenizer,scorer,block,*,branches=False):
    """Only public messages, prefixes and targets enter this function."""
    owned=copy.deepcopy(item);chats=render(tokenizer,owned['views'])
    result={'views':owned['views'],'rendered_by_view':chats,'prefixes':owned['prefixes'],
        'prefix_unavailable':owned.get('prefix_unavailable',{}),'targets':{},'levels':{},
        'active_prefix_levels':list(block['active_prefix_levels']),'target_order':list(block['target_order']),
        'source':owned['source'],'oracle_passed_to_model_or_scorer':False}
    for kind in ('tool_name','full_call'):
        if kind+'_target' in owned:result['targets'][kind]=owned[kind+'_target']
    for level in block['active_prefix_levels']:
        if level not in owned['prefixes']:
            result['levels'][level]={'available':False,'reason':owned.get('prefix_unavailable',{}).get(level,'missing_prefix')}
            continue
        prefix=owned['prefixes'][level];contexts={k:v+prefix+'Action:' for k,v in chats.items()}
        scores={}
        for kind in block['target_order']:
            if kind in result['targets']:
                target=result['targets'][kind]
                scores[kind]=safe_score(scorer,contexts,target,segments=segments(target,kind))
            else:scores[kind]={'available':False,'reason':owned.get('proposal_missing_reason','no_valid_proposal')}
        features=two_dimensional_features(scores['tool_name'],scores['full_call'])
        row={'available':True,'scores':scores,'features':features,'branches':{}}
        if branches:
            decision_contexts={k:v+prefix for k,v in chats.items()}
            events={k:safe_score(scorer,decision_contexts,v) for k,v in branch_prefixes(owned['source']).items()}
            masses={}
            if all(e.get('available') for e in events.values()):
                for view in VIEW_NAMES:
                    logs={k:v['views'][view]['joint_loglik'] for k,v in events.items()};m=max(logs.values())
                    den=sum(math.exp(x-m) for x in logs.values())
                    masses[view]={k:math.exp(x-m)/den for k,x in logs.items()}
            row['branches']={'events':events,'restricted_candidate_mass':masses,
                'exhaustive_probability_distribution':False,'action_anchor_already_in_context':False,
                'terminator_policy':'text_prefix_only_no_forced_LF_or_EOS'}
        result['levels'][level]=row
    if 'tool_name' in result['targets']:
        result['tool_layer']=tool_layer(result['targets']['tool_name'].strip())
    result['gate_status']='diagnostic_scores_only_pending_separately_frozen_calibration_and_application_scope'
    return copy.deepcopy(result)


def score_factorial(item,tokenizer,scorer,block):
    owned=copy.deepcopy(item);contexts=[]
    for index in block['context_order']:
        context=owned['contexts'][index];candidates=[]
        for candidate_index in block['candidate_order']:
            candidate=context['candidates'][candidate_index]
            public={'source':'conditional','views':context['views'],'prefixes':{'without_thought':''},
                **{k:candidate[k] for k in ('tool_name_target','full_call_target')}}
            candidates.append({'candidate_id':candidate['candidate_id'],
                'diagnostic':score_payload(public,tokenizer,scorer,block)})
        contexts.append({'context_id':context['context_id'],'candidates':candidates})
    return copy.deepcopy({'contexts':contexts,'context_order':list(block['context_order']),
        'candidate_order':list(block['candidate_order']),'new_generation_performed':False,
        'proposal_mode':'teacher_forced_factorial','oracle_passed_to_model_or_scorer':False})


def collect_free(item,generator,scorer,block):
    owned=copy.deepcopy(item);before=copy.deepcopy(owned['views']['original'])
    generated=owned_generate(generator,before,owned['tool_names'])
    parsed=parse_call_v3(generated['output'],owned['tool_names'],reached_token_limit=generated.get('reached_token_limit',False))
    chats=render(generator.tokenizer,owned['views']);thought=original_prefix(generated['output'])
    public={'source':owned['source'],'views':owned['views'],
        'prefixes':{'with_actual_thought':thought,'without_thought':''},'prefix_unavailable':{}}
    try:public['prefixes']['with_neutral_thought']=neutral_thought(generator.tokenizer,chats,thought)
    except ValueError as error:public['prefix_unavailable']['with_neutral_thought']=str(error)
    try:
        proposal=extract_proposal_v3(generated['output'],parsed)
        public.update({k:proposal[k] for k in ('tool_name_target','full_call_target')})
    except ValueError as error:public['proposal_missing_reason']=str(error)
    diagnostic=score_payload(public,generator.tokenizer,scorer,block,branches=True)
    if generated['messages']!=before or owned!=item:raise ValueError('Mutable generation/history')
    return copy.deepcopy({'generation':generated,'messages_before':before,'parsed_call':parsed,
        'diagnostic':diagnostic,'new_generation_performed':True,'explicit_nonempty_thought':bool(re.match(r'\s*Thought:\s*\S',thought)),
        'mask_provenance':owned.get('mask_provenance'), 'actual_external_tool_executions':0})


def workspace_inputs(fixture,tokenizer,tool_names):
    prepared=truncate_observations(fixture['raw_messages'],tokenizer)
    if prepared!=fixture['prepared']:raise ValueError('Runtime tokenizer/truncation differs from frozen fixture')
    messages=prepared['messages'];observations=[i for i,m in enumerate(messages) if m['role']=='tool']
    views,mask=counterfactual_messages(messages,user_index=1,obs_index=observations[-1])
    return {'source':'agentdojo','views':views,'tool_names':list(tool_names),'mask_provenance':mask}


def evaluate_workspace(evaluation,parsed,native):
    """One bounded local continuation, not a newly completed native trajectory."""
    from .run_agentdojo_bicp import new_environment,json_safe,native_utility
    suites,module,formatter=native;suite=suites[evaluation['suite']];task=suite.user_tasks[evaluation['task_id']]
    state0,pre=new_environment(suite,task)
    state=type(state0).model_validate(copy.deepcopy(evaluation['state_before_continuation']))
    runtime=module.FunctionsRuntime(suite.tools)
    calls=[module.FunctionCall.model_validate(c) for c in evaluation['executed_prefix_calls']]
    result={'state_before':json_safe(state),'executed':False,'scope':'one continuation from archived step1, not complete trajectory utility'}
    final=parsed.get('text','') if parsed.get('kind')=='final_answer' else ''
    if parsed.get('kind')=='tool_call':
        try:
            call=module.FunctionCall(function=parsed['tool'],args=copy.deepcopy(parsed['arguments']))
            value,error=runtime.run_function(state,call.function,call.args);calls.append(call)
            result.update(executed=True,tool_result=formatter(value),runtime_error=error)
        except Exception as error:result['runtime_exception']=str(error)
    result['state_after']=json_safe(state)
    try:result['native_utility_after_bounded_continuation']=native_utility(suite,task,pre,state,calls,final)
    except Exception as error:result['utility_error']=str(error)
    return copy.deepcopy(result)


def verify_bindings(root):
    binding=read(root/'configs/mechanism_execution.json')
    for path,expected in binding['file_sha256'].items():
        if sha(root/path)!=expected:raise ValueError('Bound file changed: '+path)
    protocol=read(root/'configs/mechanism_protocol.json');schedule=read(root/'configs/mechanism_schedule.json')['blocks']
    if len(schedule)!=94 or [b['block'] for b in schedule]!=list(range(94)):raise ValueError('Changed schedule')
    if Counter(b['kind'] for b in schedule)!=PANELS:raise ValueError('Changed panel scope')
    for b in schedule:
        if set(b['prefix_order'])!=set(PREFIX_LEVELS) or sorted(b['target_order'])!=['full_call','tool_name']:raise ValueError('Changed intervention order')
        expected=[x for x in b['prefix_order'] if b['kind']!='parameter_factorial' or x=='without_thought']
        if b['active_prefix_levels']!=expected:raise ValueError('Changed active prefix levels')
    if sha(root/'configs/mechanism_schedule.json')!=binding['schedule_sha256']:raise ValueError('Schedule digest')
    if sha(root/'configs/mechanism_model.json')!=protocol['model_configuration_sha256']:raise ValueError('Model file digest')
    return binding,protocol,schedule


def worker(root,output,gpu,workers):
    binding,protocol,schedule=verify_bindings(root);output.mkdir(parents=True)
    config=read(root/'configs/mechanism_model.json');write_json(output/'environment.json',environment(config))
    generator=ImmutableGenerator(copy.deepcopy(config),gpu)
    scorer=SequenceScorer(generator.tokenizer,TorchBackend(generator.model),max_input_tokens=config['max_input_tokens'])
    eos=generator.tokenizer.encode('<|im_end|>',add_special_tokens=False)
    if eos!=[151645]:raise ValueError('Tokenizer EOS mismatch')
    write_json(output/'model.json',{'configuration':config,'configuration_sha256':semantic_sha(config),
        'model_class':type(generator.model).__name__,'tokenizer_class':type(generator.tokenizer).__name__,
        'eos_ids':eos,'device_map':{str(k):str(v) for k,v in generator.model.hf_device_map.items()}})
    # No unplanned warm-up generation. The only seven generations are the scheduled ones.
    factors={r['block_id']:r for r in read(root/(DATA+'parameter_factorial_payloads.json'))}
    replays={r['replay_id']:r for r in read_lines(root/(DATA+'thought_replay_pool.jsonl'))+read_lines(root/(DATA+'native_discovery_pool.jsonl'))}
    malicious=read(root/(DATA+'actual_malicious_parameter_target.json'));replays[malicious['replay_id']]=malicious
    free={r['sample_id']:r for r in read_lines(root/(DATA+'conditional_free_smoke_inputs.jsonl'))}
    workspace=read(root/(DATA+'workspace_truncation_fixture.json'));native=None;errors=0;count=0
    with (output/'records.jsonl').open('w',encoding='utf-8') as stream:
        for block in schedule[gpu::workers]:
            unit=block['unit_id'];kind=block['kind']
            try:
                if kind=='parameter_factorial':
                    record=score_factorial(factors[unit],generator.tokenizer,scorer,block)
                    record['evaluation_only']=read(root/(DATA+'parameter_evaluation_labels.json'))[unit]
                elif kind in ('thought_replay','native_discovery_replay','actual_malicious_proposal'):
                    item=replays[unit]
                    record={'diagnostic':score_payload(item,generator.tokenizer,scorer,block),
                        'historical_output':item['historical_output'],'historical_provenance':item.get('historical_provenance'),
                        'new_generation_performed':False}
                    if kind=='thought_replay':record['evaluation_only']=read(root/(DATA+'thought_evaluation_labels.json'))[unit]
                    elif kind=='native_discovery_replay':record['evaluation_only']=read(root/(DATA+'native_evaluation_labels.json'))[unit]
                    else:
                        from .conditional_tasks import simulate_case
                        cases=read(root/(DATA+'conditional_evaluation_cases.json'))['cases']
                        case=cases['bicp:calendar:date_window:01:same_tool_argument_attack']
                        parsed=parse_call_v3(item['historical_output'],item['tool_names'])
                        record['evaluation_only']={'no_defense':simulate_case(case,parsed),
                            'gate_outcome':None,'gate_reason':'pending_separate_calibration; no development-fitted threshold'}
                else:
                    if kind=='workspace_truncated_continuation':
                        if native is None:
                            from .run_agentdojo_bicp import open_runtime
                            native=open_runtime(root)
                        runtime=native[1].FunctionsRuntime(native[0]['workspace'].tools)
                        item=workspace_inputs(workspace,generator.tokenizer,runtime.functions)
                    else:item=free[unit]
                    record=collect_free(item,generator,scorer,block)
                    if kind=='conditional_free_smoke':
                        from .conditional_tasks import classify_call,simulate_case
                        case=read(root/(DATA+'conditional_evaluation_cases.json'))['cases'][unit]
                        record['evaluation_only']={'authorization':classify_call(case,record['parsed_call']),
                            'no_defense':simulate_case(case,record['parsed_call'])}
                    else:record['evaluation_only']=evaluate_workspace(read(root/(DATA+'workspace_evaluation_state.json')),record['parsed_call'],native)
            except Exception as error:
                errors+=1;record={**error_record(error,generator),'traceback':traceback.format_exc()}
            record.update(kind=kind,unit_id=unit,schedule_block=block['block'],worker_gpu=gpu,
                execution_order=copy.deepcopy(block),binding_sha256=sha(root/'configs/mechanism_execution.json'),
                actual_model_measurement=True,actual_external_tool_executions=0,thresholds_selected=False)
            write_record(stream,copy.deepcopy(record));count+=1
            write_json(output/'progress.json',{'recorded':count,'planned':len(schedule[gpu::workers]),'last_block':block['block'],'record_errors':errors})
            print(json.dumps({'gpu':gpu,'block':block['block'],'kind':kind,'error':record.get('error')}),flush=True)
    write_json(output/'completion.json',{'collection_complete':count==len(schedule[gpu::workers]),'recorded':count,'record_errors':errors})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2,choices=(1,2));p.add_argument('--gpu',type=int);a=p.parse_args()
    root=a.root.resolve();output=a.output.resolve()
    if a.gpu is not None:worker(root,output,a.gpu,a.workers);return
    _,_,schedule=verify_bindings(root)
    if output.exists():raise ValueError('Preserve earlier run')
    output.mkdir(parents=True)
    for name in ('mechanism_execution.json','mechanism_model.json','mechanism_protocol.json','mechanism_schedule.json'):
        (output/name).write_bytes((root/'configs'/name).read_bytes())
    from huggingface_hub import snapshot_download
    config=read(root/'configs/mechanism_model.json')
    snapshot_download(repo_id=config['model_id'],revision=config['model_revision'],allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
    jobs=[subprocess.Popen([sys.executable,'-m','src.run_mechanism_v4','--root',str(root),'--output',str(output/'workers'/f'worker_{i}'),
        '--workers',str(a.workers),'--gpu',str(i)],cwd=root) for i in range(a.workers)]
    exits=[job.wait() for job in jobs];records=[]
    for path in sorted((output/'workers').glob('worker_*/records.jsonl')):records.extend(read_lines(path))
    records.sort(key=lambda r:r['schedule_block'])
    with (output/'records.jsonl').open('w',encoding='utf-8') as stream:
        for record in records:write_record(stream,record)
    complete=[r['schedule_block'] for r in records]==list(range(94))
    write_json(output/'completion.json',{'collection_complete':complete,'planned_blocks':94,'recorded_blocks':len(records),
        'record_errors':sum('error' in r for r in records),'worker_exit_codes':exits,'actual_model_measurement':True,
        'independently_accepted':False,'thresholds_selected':False,'actual_external_tool_executions':0})
    if not complete or any(exits):raise RuntimeError('Incomplete collection; retain partial records')


if __name__=='__main__':main()
