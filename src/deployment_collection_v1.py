"""Public-input-only calibration collection and measured preflight primitives.

No oracle, attack/legality label, simulator, threshold fitting or tool execution
is an input to this module. Existing deployment-v1 sources remain frozen.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

from .deployment_probe_v1 import DeploymentScorer,SharedPrefixBackend,clone_cache,axis_features,prepare_slices,required_views,VIEWS
from .probe import aggregate_token_logprobs
from .deployment_guard_v1 import parse_deployment,grounding_check,contract_from_public_user
from .mechanism_probe import SequenceScorer,TorchBackend
from .mechanism_v3 import extract_proposal_v3
from .mechanism_v4 import tool_layer,DISCOVERY
from .parsing import validate_schema
from .run_mechanism_diagnostic import owned_generate

METHODS=('diagnostic6','sliced3','sliced2','batch2','shared_kv')
ALPHAS=(.5,1.,2.)
FORBIDDEN_KEYS={'oracle','oracle_policy','environment','is_attack','attacker_target','legal_call','evaluation_only','evaluation_only_labels'}

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def assert_public(value):
    # Text in U/O and tool descriptions remains ordinary model input. Only the
    # structured payload keys are inspected; quoted hostile text is never code.
    if isinstance(value,dict):
        if FORBIDDEN_KEYS&set(value):raise ValueError('Evaluation-only field in public payload')
        for x in value.values():assert_public(x)
    elif isinstance(value,list):
        for x in value:assert_public(x)

def verify_bindings(root):
    root=Path(root);binding=read(root/'configs/deployment_execution.json')
    for name,value in binding['file_sha256'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=value:raise ValueError('Bound file changed: '+name)
    schedule=read(root/'data/deployment_calibration/calibration_execution_schedule.json')['blocks']
    if len(schedule)!=390 or [x['block'] for x in schedule]!=list(range(390)):raise ValueError('390-block schedule mismatch')
    return binding,schedule

class PublicStore:
    def __init__(self,root):
        self.root=Path(root);folder=self.root/'data/deployment_calibration'
        self.inputs={r['unit_id']:r for r in rows(folder/'calibration_public_inputs.jsonl')}
        self.forced={r['proposal_id']:r for r in rows(folder/'calibration_forced_proposals.jsonl')}
        self.controls={r['proposal_id']:r for r in rows(folder/'calibration_injecagent_high_risk_controls.jsonl')}
        self.schemas=read(folder/'schemas.json');self.tags=read(folder/'tool_risks.json')['tools']
        self.contracts=read(folder/'calibration_grounding_contracts.json')
        self.preflight=read(folder/'future_7b_preflight_schedule.json')
        if (len(self.inputs),len(self.forced),len(self.controls))!=(306,80,4):raise ValueError('Public roster count mismatch')
        for value in (self.inputs,self.forced,self.controls,self.contracts):assert_public(value)
        if any(r['split']!='calibration' for r in [*self.inputs.values(),*self.forced.values(),*self.controls.values()]):raise ValueError('Non-calibration payload')

    def proposal(self,identity):
        if identity in self.controls:return copy.deepcopy(self.controls[identity])
        row=copy.deepcopy(self.forced[identity]);item=copy.deepcopy(self.inputs[row['unit_id']])
        item.update(row);return item

    def item(self,block):
        return copy.deepcopy(self.inputs[block['unit_id']]) if block['kind']=='free_generation' else self.proposal(block['proposal_id'])

def render(tokenizer,views):
    if set(views)!=set(VIEWS):raise ValueError('Three public views required')
    return {k:tokenizer.apply_chat_template(v,tokenize=False,add_generation_prompt=True) for k,v in views.items()}

def replacement_span(original,masked,marker):
    if len(original)!=len(masked):raise ValueError('Message count changed by mask')
    changed=[]
    for i,(a,b) in enumerate(zip(original,masked)):
        if {k:v for k,v in a.items() if k!='content'}!={k:v for k,v in b.items() if k!='content'}:raise ValueError('Mask changed protected metadata')
        if a['content']!=b['content']:changed.append(i)
    if len(changed)!=1:raise ValueError('Mask must change one public source span')
    i=changed[0];text=masked[i]['content']
    if text.count(marker)!=1:raise ValueError('Ambiguous mask placeholder')
    prefix,suffix=text.split(marker)
    raw=original[i]['content']
    if not raw.startswith(prefix) or not raw.endswith(suffix):raise ValueError('Mask changed outside source field')
    end=len(raw)-len(suffix) if suffix else len(raw)
    return i,len(prefix),end,raw[len(prefix):end]

def prepare_public(item,tokenizer,prompt_budget=7424):
    owned=copy.deepcopy(item);views=owned['views'];original=views['original'];mask=owned['mask_provenance']
    oi,start,end,observation=replacement_span(original,views['observation_masked'],mask['obs_placeholder'])
    replacement_span(original,views['user_masked'],mask['user_placeholder'])
    def count(v):return {k:len(tokenizer.encode(t,add_special_tokens=False)) for k,t in render(tokenizer,v).items()}
    before=count(views);edits=[]
    if max(before.values())>prompt_budget:
        # These calibration inputs have no workspace list_files observation.
        # Crop only the exact O span obtained from the fixed mask template;
        # never crop the enclosing InjecAgent user message containing Schemas.
        marker='\n[EXTERNAL_OBSERVATION_TRUNCATED head_tail_v1]\n'
        def cropped(keep):
            n=(keep+1)//2;tail=keep//2
            text=observation[:n]+marker+(observation[-tail:] if tail else '')
            new=copy.deepcopy(views)
            new['original'][oi]['content']=original[oi]['content'][:start]+text+original[oi]['content'][end:]
            u=new['user_masked'][oi]['content']
            if u.count(observation)!=1:raise ValueError('Observation not uniquely isolated in user-masked view')
            new['user_masked'][oi]['content']=u.replace(observation,text,1)
            return new
        low,high=0,len(observation);best=None
        while low<=high:
            mid=(low+high)//2;candidate=cropped(mid)
            if max(count(candidate).values())<=prompt_budget:best=candidate;low=mid+1
            else:high=mid-1
        if best is None:raise ValueError('Protected S/Schema/U exceeds budget; no silent truncation')
        views=best
        edits=[{'message_index':oi,'original_span':[start,end],'original_O_sha256':digest(observation),
            'rule':'exact_O_only_head_tail; public source-template boundary','protected_prefix_suffix_preserved':True}]
    owned['views']=copy.deepcopy(views)
    return owned,{'original_prompt_tokens_by_view':before,'effective_prompt_tokens_by_view':count(views),
        'prompt_budget':prompt_budget,'edits':edits,'raw_public_payload_sha256':digest(item),
        'S_Schema_U_history_preserved':True}

class CompatiblePrefixBackend(SharedPrefixBackend):
    """Preserve DynamicCache type on HF4.51; legacy tuple supported on HF4.40."""
    def score_shared(self,rows):
        import torch
        contexts=[r['input_ids'][:r['input_token_count']] for r in rows];p=0
        for values in zip(*contexts):
            if len(set(values))!=1:break
            p+=1
        if p==0:raise ValueError('No common public context prefix')
        def fork(cache):
            if hasattr(cache,'to_legacy_cache') and hasattr(type(cache),'from_legacy_cache'):
                return type(cache).from_legacy_cache(clone_cache(cache.to_legacy_cache()))
            return clone_cache(cache)
        training=self.model.training;self.model.eval()
        try:
            with torch.inference_mode():
                x=torch.tensor([contexts[0][:p]],device=self.device());pre=self.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=True,return_dict=True)
                cache=fork(pre.past_key_values);last=pre.logits[0,-1].float().clone();del pre
                result=[]
                for row in rows:
                    inputs=torch.tensor([row['input_ids'][p:]],device=self.device());branch=fork(cache)
                    output=self.model(input_ids=inputs,attention_mask=torch.ones((1,len(row['input_ids'])),dtype=torch.long,device=self.device()),
                        position_ids=torch.arange(p,len(row['input_ids']),device=self.device())[None,:],past_key_values=branch,use_cache=True,return_dict=True)
                    vectors=[]
                    for pos in row['prediction_positions']:
                        if pos<p-1:raise ValueError('Target in shared prefix')
                        vectors.append(last if pos==p-1 else output.logits[0,pos-p].float())
                    logits=torch.stack(vectors);targets=torch.tensor(row['candidate_token_ids'],device=logits.device)
                    result.append(torch.log_softmax(logits,dim=-1).gather(1,targets[:,None]).squeeze(1).cpu().tolist())
                    del output,inputs,branch,logits
                return result,{'shared_prefix_tokens':p,'model_forward_calls':1+len(rows),'processed_input_tokens':p+sum(len(r['input_ids'])-p for r in rows),
                    'cache_forked_per_view':True,'cache_type':type(cache).__name__,'shared_prefix_sha256':digest(contexts[0][:p])}
        finally:
            if training:self.model.train()

class CollectionScorer(DeploymentScorer):
    def __init__(self,tokenizer,model,**kwargs):
        super().__init__(tokenizer,model,**kwargs);self.backend=CompatiblePrefixBackend(model)

    def score(self,contexts,name_target,full_target,*,alpha_name=1.,alpha_arg=1.,strategy='serial',original_reuse=None):
        required_views(alpha_name,alpha_arg)
        if strategy!='batch2' or alpha_name==alpha_arg==1 or original_reuse is not None:
            return super().score(contexts,name_target,full_target,alpha_name=alpha_name,alpha_arg=alpha_arg,strategy=strategy,original_reuse=original_reuse)
        # Keep -O and -U in the SAME batch for both collection and alpha=1
        # deployment. Batching [original,-O] then [-U] would change the NF4
        # forward shape of the two deployment views when original cancels.
        started=time.perf_counter()
        dual=super().score(contexts,name_target,full_target,alpha_name=1.,alpha_arg=1.,strategy='batch2')
        memory=self.backend.memory_start();t=time.perf_counter()
        original=prepare_slices(self.tokenizer,contexts,name_target,full_target,views=('original',),max_input_tokens=self.max_input_tokens)['original']
        prepared=time.perf_counter();row=original['full']
        logs=self.backend.score_batch([row['input_ids']],[[1]*len(row['input_ids'])],[row])[0]
        self.backend.synchronize();finished=time.perf_counter();n=original['name_tokens']
        dual['scores']['original']={kind:{**original[kind],**aggregate_token_logprobs(logs[:n] if kind=='name' else logs)} for kind in ('name','full')}
        original_memory=self.backend.memory_end(memory);combined_memory=copy.deepcopy(dual['cost']['memory'])
        for key in ('peak_allocated_bytes','peak_reserved_bytes','peak_incremental_allocated_bytes'):
            if key in original_memory:combined_memory[key]=max(combined_memory.get(key,0),original_memory[key])
        dual['cost']={'preparation_ms':dual['cost']['preparation_ms']+(prepared-t)*1000,
            'forward_ms':dual['cost']['forward_ms']+(finished-prepared)*1000,'total_ms':(finished-started)*1000,
            'model_forward_calls':2,'batch_sizes':[2,1],'view_batch_order':[['observation_masked','user_masked'],['original']],
            'physical_input_tokens':dual['cost']['physical_input_tokens']+len(row['input_ids']),'scored_view_rows':3,'memory':combined_memory,
            'cache_reused':False,'alpha1_forward_shape_preserved':True}
        dual['features']=axis_features(dual['scores'],alpha_name,alpha_arg)
        return copy.deepcopy(dual)

    def three(self,contexts,name,full,strategy='serial'):
        # Non-1 placeholders force storage of all views; no alpha is fitted here.
        result=self.score(contexts,name,full,alpha_name=.5,alpha_arg=2.,strategy=strategy)
        result['alpha_grid_features']={str(an)+':'+str(aa):axis_features(result['scores'],an,aa) for an in ALPHAS for aa in ALPHAS}
        result['collection_only_alpha_placeholders']=[.5,2.]
        return copy.deepcopy(result)

def diagnostic6(scorer,contexts,name,full):
    legacy=SequenceScorer(scorer.tokenizer,scorer.backend,max_input_tokens=scorer.max_input_tokens)
    a=legacy.score(contexts,name);b=legacy.score(contexts,full)
    return {'scores':{v:{'name':a['views'][v],'full':b['views'][v]} for v in VIEWS},
        'cost':{'model_forward_calls':6,'physical_input_tokens':a['cost']['padded_model_input_tokens']+b['cost']['padded_model_input_tokens'],
            'preparation_ms':a['cost']['preparation_ms']+b['cost']['preparation_ms'],
            'forward_ms':a['cost']['forward_ms']+b['cost']['forward_ms']}}

def compact_scores(result):return {v:{k:x[k]['token_logprobs'] for k in ('name','full')} for v,x in result['scores'].items()}

def compare(reference,result,tolerance):
    errors=[];features=[]
    for v,x in result['scores'].items():
        for k in ('name','full'):
            a=reference['scores'][v][k];b=x[k]
            if a['candidate_token_ids']!=b['candidate_token_ids'] or len(a['token_logprobs'])!=len(b['token_logprobs']):raise ValueError('Candidate identity mismatch')
            errors.extend(abs(i-j) for i,j in zip(a['token_logprobs'],b['token_logprobs']))
    grid=[(an,aa) for an in ALPHAS for aa in ALPHAS] if 'original' in result['scores'] else [(1.,1.)]
    for an,aa in grid:
        a=axis_features(reference['scores'],an,aa);b=axis_features(result['scores'],an,aa)
        features.extend(abs(a[k]-b[k]) for k in ('Risk_name','DeltaRisk_arg'))
    if not all(math.isfinite(x) for x in errors+features):raise ValueError('Nonfinite preflight')
    a=max(errors,default=0.);b=max(features,default=0.)
    return {'passed':a<=tolerance['token_logprob_max_abs'] and b<=tolerance['axis_feature_max_abs'],
        'max_token_logprob_error':a,'max_axis_feature_error':b,'token_comparisons':len(errors),'feature_comparisons':len(features)}

def preflight_one(item,scorer,plan,*,timing=True):
    owned,prep=prepare_public(item,scorer.tokenizer);contexts={k:v+'Action:' for k,v in render(scorer.tokenizer,owned['views']).items()}
    name,full=owned['tool_name_target'],owned['full_call_target'];reference=diagnostic6(scorer,contexts,name,full)
    methods={'diagnostic6':lambda:diagnostic6(scorer,contexts,name,full),'sliced3':lambda:scorer.three(contexts,name,full),
        'sliced2':lambda:scorer.score(contexts,name,full),'batch2':lambda:scorer.score(contexts,name,full,strategy='batch2'),
        'shared_kv':lambda:scorer.score(contexts,name,full,strategy='shared_kv')}
    initial={};errors={};collection3={};timings=[]
    for method,fn in methods.items():
        try:
            result=fn();initial[method]=result;check=compare(reference,result,plan['predeclared_precision_tolerance'])
            if not check['passed']:errors[method]={'precision':check}
        except Exception as error:errors[method]={'type':type(error).__name__,'error':str(error)}
    # Validate the actual THREE-view collection shape for batch/cache too.
    for strategy in ('serial','batch2','shared_kv'):
        try:
            result=scorer.three(contexts,name,full,strategy)
            method={'serial':'sliced2','batch2':'batch2','shared_kv':'shared_kv'}[strategy]
            shape=compare(result,initial[method],{'token_logprob_max_abs':1e-6,'axis_feature_max_abs':2e-6}) if method in initial else {'passed':False}
            collection3[strategy]={'score':result,'precision':compare(reference,result,plan['predeclared_precision_tolerance']),
                'collection_online_alpha1_shape_precision':shape}
            if not shape['passed']:collection3[strategy]['precision']['passed']=False
        except Exception as error:collection3[strategy]={'precision':{'passed':False},'error':str(error)}
    if timing:
        for method,fn in methods.items():
            if method not in errors:
                for _ in range(plan['warmups']):fn()
        for repetition in range(plan['paired_repetitions']):
            order=list(METHODS);random.Random(plan['timing_seed']+repetition).shuffle(order)
            for method in order:
                if method in errors:continue
                try:
                    memory=scorer.backend.memory_start();scorer.backend.synchronize();start=time.perf_counter()
                    result=methods[method]();scorer.backend.synchronize();elapsed=(time.perf_counter()-start)*1000
                    check=compare(reference,result,plan['predeclared_precision_tolerance'])
                    if not check['passed']:errors[method]={'repeated_precision':check}
                    timings.append({'repetition':repetition,'method':method,'elapsed_ms':elapsed,'cost':copy.deepcopy(result['cost']),
                        'memory':scorer.backend.memory_end(memory),'precision':check,'token_logprobs':compact_scores(result)})
                except Exception as error:errors[method]={'type':type(error).__name__,'error':str(error)}
    return copy.deepcopy({'proposal_id':owned['proposal_id'],'effective_views':owned['views'],'preparation':prep,
        'contexts':contexts,'targets':{'name':name,'full':full},'initial':initial,'errors':errors,'collection3':collection3,
        'timings':timings,'timing_performed':timing,'labels_loaded':False,'actual_external_tool_executions':0})

def choose_strategy(results,plan):
    if len(results)!=12 or [r['proposal_id'] for r in results]!=plan['proposal_ids']:raise ValueError('12-proposal preflight order mismatch')
    if any('sliced3' in r['errors'] or 'sliced2' in r['errors'] or not r['collection3']['serial']['precision']['passed'] for r in results):
        raise ValueError('Primary prefix-sliced/two-view precision failed; collection must not start')
    eligible={'serial':1.};details={}
    for strategy in ('batch2','shared_kv'):
        ratios=[];passed=True
        for r in results:
            if strategy in r['errors'] or not r['collection3'][strategy]['precision']['passed']:passed=False;break
            a={x['repetition']:x['elapsed_ms'] for x in r['timings'] if x['method']=='sliced2'}
            b={x['repetition']:x['elapsed_ms'] for x in r['timings'] if x['method']==strategy}
            if len(a)!=plan['paired_repetitions'] or set(a)!=set(b):passed=False;break
            ratio=statistics.median(a[k]/b[k] for k in a);ratios.append(ratio)
            if ratio<=1:passed=False
        details[strategy]={'all_12_passed_and_each_proposal_faster':passed,'paired_speedup_by_proposal':ratios}
        if passed:eligible[strategy]=statistics.median(ratios)
    selected=min(eligible,key=lambda k:(-eligible[k],{'serial':0,'batch2':1,'shared_kv':2}[k]))
    return {'strategy':selected,'strategy_frozen_before_collection':True,'eligible_median_paired_speedups':eligible,
        'alternative_details':details,'all_primary_precision_passed':True,'proposal_ids':plan['proposal_ids'],
        'selection_basis':'no labels; all 12 precision checks and per-proposal latency; serial on tie',
        'calibration_views':list(VIEWS),'online_alpha1_views':list(VIEWS[1:]),'thresholds_fitted':False}

def precheck(parsed,item,schemas,tags):
    result={'branch_argmax_used':False,'actual_external_tool_executions':0,'online_extra_forward_calls_before_probe':0,
        'schema':{'valid':None,'reason':'not_applicable_or_not_reached'},'scope_allowed':None,
        'grounding':{'passed':None,'reason':'not_applicable_or_not_reached'}}
    if parsed.get('kind')=='final_answer':return {**result,'decision':'terminal','online_probe_required':False,'preprobe_rejected':False}
    if parsed.get('kind')!='tool_call':return {**result,'decision':'NEED_INFORMATION','reason':'parse','online_probe_required':False,'preprobe_rejected':True}
    tool=parsed['tool'];args=parsed.get('arguments');scope=tool in item['tool_names'];schema=validate_schema(args,schemas[tool]) if tool in schemas and isinstance(args,dict) else {'valid':False}
    result.update(schema=copy.deepcopy(schema),scope_allowed=scope)
    if not scope or not schema['valid']:return {**result,'decision':'NEED_INFORMATION','reason':'scope_or_schema','online_probe_required':False,'preprobe_rejected':True}
    layer=tool_layer(tool,tags)
    if tags.get(tool,{}).get('risk_tag')=='sensitive_or_private_read':layer='protected_sensitive_read'
    result['tool_layer']=layer
    if layer==DISCOVERY:return {**result,'decision':'allow_scoped_discovery','online_probe_required':False,'preprobe_rejected':False}
    contract=None
    if item['source']=='conditional':contract=contract_from_public_user(tool,'\n'.join(m['content'] for m in item['views']['original'] if m['role']=='user'))
    grounded=grounding_check(tool,args,item['views']['original'],contract);result['grounding']=grounded
    rejected=not grounded['passed']
    return {**result,'decision':'NEED_INFORMATION' if rejected else 'pending_frozen_probe','reason':'grounding' if rejected else None,
        'online_probe_required':not rejected,'preprobe_rejected':rejected}

def parse_public_output(text,tools,*,reached_token_limit=False):
    parsed=parse_deployment(text,tools,reached_token_limit=reached_token_limit)
    def finite(value):
        if isinstance(value,float):return math.isfinite(value)
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        if isinstance(value,list):return all(finite(v) for v in value)
        return True
    if not finite(parsed):
        # Preserve the entire raw generation separately, without putting NaN in
        # JSON score logs or allowing permissive JSON decoders to approve it.
        return {'kind':'invalid','reason':'nonfinite_JSON_argument','original_tool':parsed.get('tool')}
    return copy.deepcopy(parsed)

def collect_block(block,store,generator,scorer,strategy):
    raw=store.item(block);item,preparation=prepare_public(raw,scorer.tokenizer)
    generated=None
    if block['kind']=='free_generation':
        generated=owned_generate(generator,item['views']['original'],item['tool_names'])
        parsed=parse_public_output(generated['output'],item['tool_names'],reached_token_limit=generated.get('reached_token_limit',False))
        output=generated['output']
    else:
        parsed=copy.deepcopy(item['parsed_call']);output='Action: '+parsed['tool']+'\nAction Input: '+json.dumps(parsed['arguments'],ensure_ascii=False,separators=(',',':'))
    route=precheck(parsed,item,store.schemas,store.tags);score=None;proposal=None;event=None
    if parsed.get('kind')=='tool_call' and route.get('schema',{}).get('valid') and route.get('scope_allowed'):
        if generated is None:
            proposal={k:item[k] for k in ('tool_name_target','full_call_target')}
        else:proposal=extract_proposal_v3(output,parsed)
        contexts={k:v+'Action:' for k,v in render(scorer.tokenizer,item['views']).items()}
        score=scorer.three(contexts,proposal['tool_name_target'],proposal['full_call_target'],strategy)
        event=digest({'input_ids':score['scores']['original']['full']['input_ids'],
            'candidate_token_ids':score['scores']['original']['full']['candidate_token_ids']})
    return copy.deepcopy({'schedule_block':block['block'],'kind':block['kind'],'proposal_id':block.get('proposal_id','free:'+block.get('unit_id','')),
        'source_sample_id':raw.get('unit_id',raw.get('source_sample_id')),'source':raw['source'],'split':'calibration','group_id':raw['group_id'],
        'generation':generated,'parsed_call':parsed,'routing':route,'preparation':preparation,'effective_views':item['views'],
        'proposal':proposal,'score':score,'event_sha256':event,'scorer_strategy':strategy,
        'score_cost_scope':'offline_three_view_diagnostic; separate from online routing cost',
        'labels_loaded':False,'thresholds_fitted':False,'actual_external_tool_executions':0})
