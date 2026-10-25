"""Pilot recovery adapter: observable context-budget fallback and checkpoints.

The frozen guard/scorer/metadata projection remain in their original modules.
"""
import copy,time
from .deployment_pilot_v1 import decide,is_allowed,flat_schema
from .deployment_guard_v1 import prepare_observations,digest as metadata_digest
from .deployment_collection_v1 import digest
from .mechanism_v3 import add_format_contract
from .run_mechanism_diagnostic import owned_generate
from .run_bicp import counterfactual_messages

BUDGET_ERROR='Protected structured metadata exceeds prompt budget; NEED_INFORMATION'

def prepare_workspace(messages,tokenizer,*,prompt_budget=7424):
    try:return prepare_observations(messages,tokenizer,prompt_budget=prompt_budget)
    except ValueError as error:
        if str(error)!=BUDGET_ERROR:raise
    # Recover the exact attempted metadata projection for audit only. It is
    # never supplied to the model when it exceeds the protected budget.
    import yaml
    original=copy.deepcopy(messages);projected=copy.deepcopy(messages);edits=[]
    count=lambda ms:len(tokenizer.encode(tokenizer.apply_chat_template(ms,tokenize=False,add_generation_prompt=True),add_special_tokens=False))
    for i,m in enumerate(original):
        if m['role']!='tool' or m.get('name')!='list_files':continue
        values=yaml.safe_load(m['content']);metadata=[{k:v for k,v in x.items() if k!='content'} for x in values]
        body=yaml.safe_dump(metadata,sort_keys=True,allow_unicode=True);projected[i]['content']=body
        if yaml.safe_load(body)!=metadata:raise AssertionError('Recovery projection changed metadata')
        edits.append({'message_index':i,'objects_before':len(values),'objects_after':len(metadata),
            'metadata_sha256':metadata_digest(metadata),'original_body_sha256':metadata_digest(m['content']),
            'effective_body_sha256':metadata_digest(body),'rule':'all_objects_all_metadata_remove_content_uniformly'})
    if not edits or count(projected)<=prompt_budget:raise AssertionError('Budget fallback without reproducible exhaustion')
    assert all(a==b for a,b in zip(original,projected) if a['role']!='tool')
    return copy.deepcopy({'messages':None,'projected_messages_for_audit':projected,'original_prompt_tokens':count(original),
        'effective_prompt_tokens':count(projected),'prompt_budget':prompt_budget,'edits':edits,'structured_projection':True,
        'protected_messages_unchanged':True,'status':'budget_exhausted','fallback':'NEED_INFORMATION',
        'model_generation_performed':False,'extra_forward_calls':0,'error':BUDGET_ERROR})

def workspace_unit(block,store,generator,scorer,policy,runtime_parts,*,checkpoint=None):
    from .run_agentdojo_bicp import new_environment,initial_messages,json_safe,tool_definitions
    suites,rm,formatter=runtime_parts;suite=suites['workspace'];task=suite.user_tasks['user_task_27']
    runtime=rm.FunctionsRuntime(suite.tools);original,pre=new_environment(suite,task)
    initial=add_format_contract(initial_messages(task,runtime),runtime.functions);arms=[];start=time.perf_counter()
    def persist(current):
        if checkpoint:checkpoint(copy.deepcopy({'unit_id':block['unit_id'],'schedule_block':block['block'],
            'completed_arms':arms,'current_arm':current,'evaluation_labels_loaded':False,'actual_external_tool_executions':0}))
    for arm in block['arm_order']:
        env=original.model_copy(deep=True);messages=copy.deepcopy(initial);events=[];calls=[];final='';terminal='step_limit';arm_start=time.perf_counter()
        persist({'arm':arm,'events':events,'messages':messages,'state':json_safe(env),'phase':'arm_start'})
        for step in range(8):
            event_start=time.perf_counter();t=time.perf_counter();prepared=prepare_workspace(messages,scorer.tokenizer)
            prep_ms=(time.perf_counter()-t)*1000;before=json_safe(env)
            if prepared.get('status')=='budget_exhausted':
                d={'arm':arm,'decision':'NEED_INFORMATION','reason':'context_budget_exhausted','probe':None,
                    'extra_forward_calls':0,'branch_argmax_used':False,'parsed_call':{'kind':'not_generated'},
                    'actual_external_tool_executions':0,'policy_sha256':digest(policy),'timing':{'total_ms':0.}}
                event={'event_type':'budget_fallback','step':step,'messages_before':copy.deepcopy(messages),'effective_messages':None,
                    'preparation':prepared,'generation':None,'decision':d,'state_before':before,'state_after':copy.deepcopy(before),
                    'executed':False,'mask_provenance':None,'simulated_execution_ms':0.,'tool_result':None,'runtime_error':None,
                    'timing':{'input_preparation_ms':prep_ms,'generation_ms':0.,'decision_ms':0.,'simulated_execution_ms':0.,
                        'end_to_end_ms':(time.perf_counter()-event_start)*1000}}
                final='NEED_INFORMATION';terminal='context_budget_need_information'
            else:
                effective=copy.deepcopy(prepared['messages']);generation=owned_generate(generator,effective,list(runtime.functions))
                views={'original':copy.deepcopy(effective)};mask=None
                if any(m['role']=='tool' for m in effective):views,mask=counterfactual_messages(copy.deepcopy(effective),user_index=1)
                specs={r['name']:flat_schema(r) for r in tool_definitions(runtime)}
                item={'source':'workspace_development','tool_names':list(runtime.functions),'schemas':specs,'views':views,'reviewed_tags':store.tags}
                d=decide(generation['output'],item,scorer,policy,arm,reached_token_limit=generation.get('reached_token_limit',False),
                    native_validate=lambda name,args:runtime.functions[name].parameters.model_validate(args))
                event={'event_type':'model_decision','step':step,'messages_before':copy.deepcopy(messages),'effective_messages':effective,
                    'preparation':prepared,'generation':generation,'decision':d,'state_before':before,'executed':False,'state_after':copy.deepcopy(before),
                    'mask_provenance':mask,'simulated_execution_ms':0.,'tool_result':None,'runtime_error':None}
                if d['decision']=='terminal':final=d['terminal_text'];terminal='final_answer'
                elif not is_allowed(d):final='NEED_INFORMATION';terminal='blocked_or_invalid'
                else:
                    p=d['parsed_call'];call=rm.FunctionCall(function=p['tool'],args=copy.deepcopy(p['arguments']));t=time.perf_counter()
                    value,error=runtime.run_function(env,call.function,copy.deepcopy(call.args));result=formatter(value) if error is None else 'Error: '+str(error)
                    event.update(executed=True,state_after=json_safe(env),tool_result=result,runtime_error=error,simulated_execution_ms=(time.perf_counter()-t)*1000)
                    calls.append(json_safe(call));messages.extend([{'role':'assistant','content':generation['output']},{'role':'tool','name':call.function,'content':result}])
                event['timing']={'input_preparation_ms':prep_ms,'generation_ms':generation['timing']['total_ms'],
                    'decision_ms':d['timing']['total_ms'],'simulated_execution_ms':event['simulated_execution_ms'],'end_to_end_ms':(time.perf_counter()-event_start)*1000}
            if not event['executed']:assert event['state_before']==event['state_after']
            events.append(copy.deepcopy(event))
            persist({'arm':arm,'events':events,'messages':messages,'state':json_safe(env),'phase':'step_complete'})
            if not event['executed']:break
        arms.append(copy.deepcopy({'arm':arm,'events':events,'initial_state':json_safe(pre),'final_state':json_safe(env),
            'initial_messages':initial,'messages':messages,'executed_calls':calls,'final_text':final,'terminal':terminal,
            'physical_end_to_end_ms':(time.perf_counter()-arm_start)*1000,'evaluation_performed':False}))
        persist(None)
    return copy.deepcopy({'schedule_block':block['block'],'unit_id':block['unit_id'],'source':block['source'],'mode':block['mode'],
        'arms':arms,'physical_collection_ms':(time.perf_counter()-start)*1000,'actual_external_tool_executions':0,'evaluation_labels_loaded':False,
        'workspace_adapter_version':2})
