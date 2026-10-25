"""R + L frozen deployment collection: public inputs only, no label/utility reads."""
import copy,json,time,sys
from pathlib import Path
from .deployment_collection_v1 import read,rows,sha,digest,assert_public,prepare_public
from .deployment_pilot_v1 import ARMS,policy_check,decide,is_allowed,dispatch_ledger
from .native_adapter_v3 import NativeAdapterV3
from .run_mechanism_diagnostic import owned_generate

POLICY_SHA='95a3f6afcd8fe47aeade2a5fa6499b2ccaadc4d0683ea8968bea02d7dce62e99'
MODEL_SHA='d0ec15276f666ef854691c78b1df3d2fc5b79d34a054a1255cc6f74bee6b14bc'

def install_public_read_guard(root,binding):
    root=Path(root).resolve();allowed={(root/p).resolve() for p in binding['file_sha256']}|{(root/'configs/rl_execution.json').resolve()}
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        try:path=Path(args[0]).resolve()
        except (OSError,ValueError,TypeError):return
        if path.is_relative_to(root) and path not in allowed:
            if path.suffix=='.pyc' and '__pycache__' in path.parts:return
            raise PermissionError('R+L runtime cannot read unbound/evaluation project data: '+path.name)
    sys.addaudithook(guard)

def verify(root):
    root=Path(root);binding=read(root/'configs/rl_execution.json')
    for name,value in binding['file_sha256'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=value:raise ValueError('R+L bound file changed: '+name)
    if sha(root/'configs/frozen_policy.json')!=POLICY_SHA or sha(root/'configs/deployment_model.json')!=MODEL_SHA:raise ValueError('Formal frozen policy/model changed')
    policy=read(root/'configs/frozen_policy.json');policy_check(policy)
    schedule=read(root/'data/rl/schedule.json')['blocks']
    if len(schedule)!=922 or [b['block'] for b in schedule]!=list(range(922)):raise ValueError('922 schedule mismatch')
    if len({b['unit_id'] for b in schedule})!=922 or any(b['arm_order']!=list(ARMS) or b['worker']!=b['block']%2 for b in schedule):raise ValueError('Order/arm/worker tamper')
    return binding,schedule,policy

class Store:
    def __init__(self,root):
        root=Path(root);items=rows(root/'data/rl/public_inputs.jsonl');self.tags=read(root/'data/rl/tool_risks.json')['tools']
        self.items={(r['unit_id'],r['stage']):r for r in items};self.metadata=read(root/'data/rl/continuations.json')
        self.workspace=read(root/'data/rl/workspace_public.json')
        if len(items)!=len(self.items):raise ValueError('Duplicate public stage')
        if len(items)!=994 or sum(k[1]=='stage1' for k in self.items)!=922 or sum(k[1]=='stage2' for k in self.items)!=72:raise ValueError('Public stage count changed')
        for v in (items,self.metadata,self.workspace,self.tags):assert_public(v)
    def item(self,uid,stage):
        x=copy.deepcopy(self.items[uid,stage]);x['reviewed_tags']=copy.deepcopy(self.tags);return x
    def cap(self,uid,stage):return self.metadata['generation_limits'][uid][stage]
    def continuation(self,uid,decision):
        if not is_allowed(decision):return False,'first_stage_not_allowed'
        if (uid,'stage2') not in self.items:return False,'no_public_cached_continuation'
        expected=self.metadata['parent_calls'][uid];p=decision['parsed_call']
        if {'tool':p['tool'],'arguments':p['arguments']}!=expected:raise ValueError('Cached continuation parent does not match public first proposal')
        if self.metadata['stage2_public_sha256'][uid]!=digest(self.items[uid,'stage2']):raise ValueError('Cached continuation byte/event changed')
        return True,'public_cache_parent_verified'

def longitudinal_unit(block,store,scorer,policy,*,checkpoint=None):
    started=time.perf_counter();uid=block['unit_id'];arms=[];prepared={};preparation={}
    for arm in block['arm_order']:
        t=time.perf_counter();state={'dispatches':[],'effects':[]};events=[]
        for stage in ('stage1','stage2'):
            if stage=='stage2':
                reachable,reason=store.continuation(uid,events[0]['decision'])
                if not reachable:break
            else:reason='fixed_first_stage'
            pstart=time.perf_counter()
            if stage not in prepared:prepared[stage],preparation[stage]=prepare_public(store.item(uid,stage),scorer.tokenizer)
            item=copy.deepcopy(prepared[stage]);prep_ms=(time.perf_counter()-pstart)*1000
            before=copy.deepcopy(state);output=item['historical_output']
            decision=decide(output,item,scorer,policy,arm,reached_token_limit=store.cap(uid,stage))
            estart=time.perf_counter();state=dispatch_ledger(before,decision);exec_ms=(time.perf_counter()-estart)*1000
            if not is_allowed(decision) and state!=before:raise AssertionError('Blocked proposal dispatched')
            event=copy.deepcopy({'stage':stage,'cache_reachability':reason,'output':output,'historical_reached_token_limit':store.cap(uid,stage),
                'decision':decision,'preparation':preparation[stage],'effective_public_input_sha256':digest(item),
                'state_before':before,'state_after':state,'executed':is_allowed(decision),
                'timing':{'input_preparation_ms':prep_ms,'new_generation_ms':0.,'decision_ms':decision['timing']['total_ms'],
                    'simulated_dispatch_ms':exec_ms,'end_to_end_ms':prep_ms+decision['timing']['total_ms']+exec_ms}})
            events.append(event)
            if checkpoint:checkpoint(copy.deepcopy({'scope':'L','unit_id':uid,'arm':arm,'events':events,'completed_arms':arms}))
        reachable,reason=store.continuation(uid,events[0]['decision'])
        arms.append(copy.deepcopy({'arm':arm,'events':events,'initial_state':{'dispatches':[],'effects':[]},'final_state':state,
            'continuation_reachable':reachable,'continuation_reason':reason,'new_generation_calls':0,
            'physical_collection_ms':(time.perf_counter()-t)*1000}))
    return copy.deepcopy({'scope':'L','schedule_block':block['block'],'unit_id':uid,'source':block['source'],'arms':arms,
        'new_generation_calls':0,'historical_proposal_fixed':True,'evaluation_labels_loaded':False,'actual_external_tool_executions':0,
        'physical_collection_ms':(time.perf_counter()-started)*1000})

def workspace_unit(root,block,store,generator,scorer,policy,public_runtime,*,checkpoint=None):
    from .public_workspace_runtime_v1 import environment_from_public
    environment_type,runtime,rm,formatter=public_runtime
    adapter=NativeAdapterV3(root,'workspace',runtime,metadata_only_task=True)
    original=environment_from_public(environment_type,store.workspace)
    initial=copy.deepcopy(store.workspace['initial_messages']);arms=[];start=time.perf_counter()
    def persist(current):
        if checkpoint:checkpoint(copy.deepcopy({'scope':'R','unit_id':block['unit_id'],'completed_arms':arms,'current_arm':current,'evaluation_labels_loaded':False}))
    for arm in block['arm_order']:
        env=original.model_copy(deep=True);messages=copy.deepcopy(initial);events=[];calls=[];final='';terminal='step_limit';arm_start=time.perf_counter()
        persist({'arm':arm,'events':events,'messages':messages,'state':env.model_dump(mode='json'),'phase':'arm_start'})
        for step in range(8):
            begin=time.perf_counter();t=time.perf_counter();before=env.model_dump(mode='json');generation=None
            try:prepared=adapter.prepare(messages,scorer.tokenizer);preparation_error=None
            except ValueError as e:prepared=None;preparation_error=str(e)
            prep_ms=(time.perf_counter()-t)*1000
            if prepared is None or not prepared['budget']['fits']:
                d={'decision':'NEED_INFORMATION','reason':'observation_contract' if prepared is None else 'context_budget_exhausted',
                    'probe':None,'extra_forward_calls':0,'parsed_call':{'kind':'not_generated'},'timing':{'total_ms':0.},'branch_argmax_used':False}
                event={'event_type':'budget_or_contract_fallback','step':step,'messages_before':copy.deepcopy(messages),'effective_messages':None,
                    'preparation':prepared,'preparation_error':preparation_error,'generation':None,'decision':d,'state_before':before,
                    'state_after':copy.deepcopy(before),'executed':False,'simulated_execution_ms':0.,'tool_result':None,'runtime_error':None}
                final='NEED_INFORMATION';terminal='context_or_contract_need_information'
            else:
                effective=copy.deepcopy(prepared['messages']);generation=owned_generate(generator,effective,list(runtime.functions),max_new_tokens=512)
                d=adapter.decide(generation['output'],effective,scorer,policy,arm,reached_token_limit=generation.get('reached_token_limit',False))
                event={'event_type':'model_decision','step':step,'messages_before':copy.deepcopy(messages),'effective_messages':effective,
                    'preparation':prepared,'preparation_error':None,'generation':generation,'decision':d,'state_before':before,
                    'executed':False,'state_after':copy.deepcopy(before),'simulated_execution_ms':0.,'tool_result':None,'runtime_error':None}
                if d['decision']=='terminal':final=d['terminal_text'];terminal='final_answer'
                elif not is_allowed(d):final='NEED_INFORMATION';terminal='blocked_or_invalid'
                else:
                    p=d['parsed_call'];call=rm.FunctionCall(function=p['tool'],args=copy.deepcopy(p['arguments']));t=time.perf_counter()
                    value,error=runtime.run_function(env,call.function,copy.deepcopy(call.args));result=formatter(value) if error is None else 'Error: '+str(error)
                    event.update(executed=True,state_after=env.model_dump(mode='json'),tool_result=result,runtime_error=error,simulated_execution_ms=(time.perf_counter()-t)*1000)
                    calls.append(call.model_dump(mode='json'));messages.extend([{'role':'assistant','content':generation['output']},{'role':'tool','name':call.function,'content':result}])
            if not event['executed'] and event['state_before']!=event['state_after']:raise AssertionError('Blocked native call executed')
            event['timing']={'input_preparation_ms':prep_ms,'generation_ms':generation['timing']['total_ms'] if generation else 0.,
                'decision_ms':d['timing']['total_ms'],'simulated_execution_ms':event['simulated_execution_ms'],'end_to_end_ms':(time.perf_counter()-begin)*1000}
            events.append(copy.deepcopy(event));persist({'arm':arm,'events':events,'messages':messages,'state':env.model_dump(mode='json'),'phase':'step_complete'})
            if not event['executed']:break
        arms.append(copy.deepcopy({'arm':arm,'events':events,'initial_state':original.model_dump(mode='json'),'final_state':env.model_dump(mode='json'),
            'initial_messages':initial,'messages':messages,'executed_calls':calls,'final_text':final,'terminal':terminal,
            'physical_collection_ms':(time.perf_counter()-arm_start)*1000,'evaluation_performed':False}));persist(None)
    return copy.deepcopy({'scope':'R','unit_id':block['unit_id'],'arms':arms,'workspace_adapter_version':3,
        'physical_collection_ms':(time.perf_counter()-start)*1000,'actual_external_tool_executions':0,'evaluation_labels_loaded':False})
