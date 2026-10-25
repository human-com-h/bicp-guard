"""BiCP v2 local design primitives; public inputs only, no model loading.

Frozen v1/v3 remain untouched. Labels are offline evaluation data. This module
provides the two-dimensional feature, scoped risk routing, neutral controls,
and deterministic external-observation-only budget handling.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math

PREFIX_LEVELS=('with_actual_thought','without_thought','with_neutral_thought')
DISCOVERY='read_only_discovery'
MUTATION='high_risk_state_mutation'
COMMUNICATION='high_risk_external_communication'
SENSITIVE='protected_sensitive_read'
UNKNOWN='protected_unclassified'
TRUNCATION_MARKER='\n[EXTERNAL_OBSERVATION_TRUNCATED head_tail_v1]\n'
DISCOVERY_TOOLS={'get_channels','get_users_in_channel','read_channel_messages',
    'get_all_restaurants_in_city','get_all_car_rental_companies_in_city','get_car_price_per_day',
    'get_cuisine_type_for_restaurants','check_restaurant_opening_hours','get_price_for_restaurants',
    'get_rating_reviews_for_restaurants','get_rating_reviews_for_car_rental','list_files',
    'GitHubGetRepositoryDetails','ShopifyGetProductDetails','TwitterManagerGetUserProfile'}
COMMUNICATION_TOOLS={'send_channel_message','send_direct_message','MessagingForwardMessage','AdminExportData'}
MUTATION_TOOLS={'EmailArchive','CalendarCreateEvent','FinancePayInvoice','DocumentsRouteDocument',
    'add_user_to_channel','invite_user_to_slack','create_channel'}


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def branch_prefixes(source):
    if source=='conditional':return {'Execute':'Action:','NO_ACTION':'Final Answer: NO_ACTION','NEED_INFORMATION':'Final Answer: NEED_INFORMATION'}
    if source in ('injecagent','agentdojo'):return {'Execute':'Action:','Terminate':'Final Answer:'}
    raise ValueError('Unknown source')


def factorial_contrasts(values):
    """Offline numeric contrasts for the four independently labelled cells."""
    if set(values)!={('U','neutral'),('U','suspicious'),('O_only','neutral'),('O_only','suspicious')}:
        raise ValueError('All four source/appearance cells are required')
    un,us,on,os=(values[k] for k in (('U','neutral'),('U','suspicious'),('O_only','neutral'),('O_only','suspicious')))
    if not all(math.isfinite(x) for x in (un,us,on,os)):raise ValueError('Nonfinite cell')
    return {'D_neutral_bad':on-us,'C_neutral_source':on-un,
        'provenance_main':((on-un)+(os-us))/2,'lexical_main':((us-un)+(os-on))/2,
        'interaction':(os-us)-(on-un)}


def exact_positive_sign_test(values):
    """One-sided domain sign test; exact zeros/ties reported separately."""
    if not all(math.isfinite(x) for x in values):raise ValueError('Nonfinite domain effect')
    nonzero=[x for x in values if x!=0];n=len(nonzero);positive=sum(x>0 for x in nonzero)
    return {'effective_domains':n,'ties':len(values)-n,'positive_domains':positive,
        'one_sided_p':sum(math.comb(n,k) for k in range(positive,n+1))/(2**n) if n else 1.}


def prefix_gap_decomposition(actual,without,neutral):
    return {'content':neutral-actual,'length':without-neutral,'total':without-actual}


def tool_layer(tool,reviewed_tags=None):
    if tool in DISCOVERY_TOOLS:return DISCOVERY
    if tool in COMMUNICATION_TOOLS:return COMMUNICATION
    if tool in MUTATION_TOOLS:return MUTATION
    tag=(reviewed_tags or {}).get(tool,{}).get('risk_tag','')
    if tag in ('sensitive_or_private_read','public_or_personal_lookup'):return SENSITIVE
    if 'external' in tag or 'communication' in tag:return COMMUNICATION
    if tag in ('physical_action','security_or_permission_write','financial_write') or 'write' in tag:return MUTATION
    return UNKNOWN


def two_dimensional_features(name,full,field='length_normalized_loglik'):
    if name.get('available') is False or full.get('available') is False:
        return {'available':False,'reason':'one_or_both_score_events_missing'}
    def risk(cell):
        views=cell['views'];return views['user_masked'][field]-views['observation_masked'][field]
    a,b=risk(name),risk(full)
    if not all(math.isfinite(x) for x in (a,b)):raise ValueError('Nonfinite features')
    return copy.deepcopy({'available':True,'Risk_name':a,'Risk_full':b,'DeltaRisk_arg':b-a,
        'score_field':field,'meaning':'difference of normalized event risks; not an isolated argument conditional logprob'})


def route_proposal(tool,features,*,schema_valid,scope_allowed,reviewed_tags=None,thresholds=None):
    """Software policy skeleton; thresholds must be separately frozen calibration.

    scope_allowed is a trusted application capability/ACL check, never an oracle
    legality label or a ground-truth reference-call lookup.
    """
    layer=tool_layer(tool,reviewed_tags);base={'tool_layer':layer,'actual_execution':False}
    if not schema_valid:return {**base,'decision':'block_schema'}
    if not scope_allowed:return {**base,'decision':'block_scope'}
    if layer==DISCOVERY:return {**base,'decision':'allow_scoped_discovery','probe_used_for_blocking':False}
    if not features.get('available'):return {**base,'decision':'safe_fallback_missing_score'}
    if thresholds is None:return {**base,'decision':'score_only_pending_calibration'}
    if thresholds.get('frozen') is not True or thresholds.get('selection_split')!='calibration':
        raise ValueError('A separately frozen calibration policy is required')
    for key in ('tau_name','tau_arg'):
        if not math.isfinite(thresholds[key]):raise ValueError('Nonfinite threshold')
    block_name=features['Risk_name']>thresholds['tau_name']
    block_arg=features['DeltaRisk_arg']>thresholds['tau_arg']
    return {**base,'decision':'block_probe' if block_name or block_arg else 'allow_protected',
        'name_gate':block_name,'argument_gate':block_arg,'probe_used_for_blocking':True}


def continuation_ids(tokenizer,context,prefix):
    before=tokenizer.encode(context,add_special_tokens=False)
    after=tokenizer.encode(context+prefix,add_special_tokens=False)
    if after[:len(before)]!=before:raise ValueError('Context/prefix token boundary changed')
    return after[len(before):]


def neutral_thought(tokenizer,contexts,actual):
    """Match each real rendered-context prefix length with fixed neutral grammar.

    Only requested length is used; neither actual words, tools, labels nor source
    observation content enters the returned neutral text.
    """
    if not actual.lstrip().startswith('Thought:') or not actual.split(':',1)[1].strip():
        raise ValueError('A real nonempty explicit Thought is required')
    lengths={len(continuation_ids(tokenizer,c,actual)) for c in contexts.values()}
    if len(lengths)!=1:raise ValueError('Actual prefix length differs across views')
    n=next(iter(lengths));base='Thought:';sentence=' I will consider the request carefully.'
    # Candidate vocabulary is fixed. Repetitions are deterministic; no tool or
    # entity token is copied from actual reasoning, and there is no score search.
    for repetitions in range(n//2+1):
        text=base+sentence*repetitions+'\n'
        count=len(continuation_ids(tokenizer,next(iter(contexts.values())),text))
        if count>n:break
        for residual in range(12):
            candidate=base+sentence*repetitions+' Ready'*residual+'\n'
            if all(len(continuation_ids(tokenizer,c,candidate))==n for c in contexts.values()):
                if repetitions or residual:return candidate
    raise ValueError('No exact fixed-grammar neutral prefix for this token length')


def observation_head_tail(text,tokenizer,keep_tokens):
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    offsets=encoded['offset_mapping'];n=len(encoded['input_ids'])
    if keep_tokens>=n:return text,{'changed':False,'original_tokens':n}
    if keep_tokens<2:raise ValueError('Head and tail each require at least one token')
    head=(keep_tokens+1)//2;tail=keep_tokens//2
    end=offsets[head-1][1];start=offsets[n-tail][0]
    if start<=end:raise ValueError('Head/tail spans overlap')
    value=text[:end]+TRUNCATION_MARKER+text[start:]
    return value,{'changed':True,'original_tokens':n,'retained_token_budget':keep_tokens,
        'head_characters':[0,end],'tail_characters':[start,len(text)],
        'original_sha256':hashlib.sha256(text.encode()).hexdigest(),'effective_sha256':hashlib.sha256(value.encode()).hexdigest()}


def truncate_observations(messages,tokenizer,*,prompt_budget=7424,min_retained_tokens=64):
    """Crop external role=tool bodies only, before deriving three masks.

    Longest original observation first, latest index on ties. No protected
    message, metadata, order, schema or user text can be changed. Failure when
    protected content itself is too large is explicit rather than truncating U/S.
    """
    original=copy.deepcopy(messages);effective=copy.deepcopy(messages)
    render=lambda ms:tokenizer.apply_chat_template(ms,tokenize=False,add_generation_prompt=True)
    count=lambda ms:len(tokenizer.encode(render(ms),add_special_tokens=False))
    before=count(original);edits=[]
    if before>prompt_budget:
        candidates=[(len(tokenizer.encode(m['content'],add_special_tokens=False)),i) for i,m in enumerate(original) if m['role']=='tool' and isinstance(m.get('content'),str)]
        for n,index in sorted(candidates,reverse=True):
            if count(effective)<=prompt_budget:break
            if n<=min_retained_tokens:continue
            keep=max(min_retained_tokens,n-(count(effective)-prompt_budget)-24)
            while True:
                value,receipt=observation_head_tail(original[index]['content'],tokenizer,keep)
                effective[index]['content']=value;over=count(effective)-prompt_budget
                if over<=0 or keep==min_retained_tokens:break
                keep=max(min_retained_tokens,keep-over-8)
            edits.append({'message_index':index,**receipt})
    after=count(effective)
    if after>prompt_budget:raise ValueError(f'protected_or_minimum_context_overflow:{after}>{prompt_budget}')
    for i,(a,b) in enumerate(zip(original,effective)):
        if a['role']!='tool' and a!=b:raise AssertionError('Protected message changed')
        if {k:v for k,v in a.items() if k!='content'}!={k:v for k,v in b.items() if k!='content'}:raise AssertionError('Tool metadata changed')
    return copy.deepcopy({'messages':effective,'original_prompt_tokens':before,'effective_prompt_tokens':after,
        'prompt_budget':prompt_budget,'edits':edits,'policy':'external-role-tool-head-tail-v1',
        'protected_messages_unchanged':True,'original_messages_sha256':digest(original),'effective_messages_sha256':digest(effective)})


def truncate_serialized_observation(case,tokenizer,build_views,*,prompt_budget=7424,min_retained_tokens=64):
    """InjecAgent: edit only the authenticated source Tool Response field.

    Its carrier is a user-role template containing both U and O. Never crop that
    carrier wholesale; rebuild all masks from the same bounded source field.
    """
    original=copy.deepcopy(case);effective=copy.deepcopy(case)
    views,mask=build_views(effective)
    count=lambda v:len(tokenizer.encode(tokenizer.apply_chat_template(v['original'],tokenize=False,add_generation_prompt=True),add_special_tokens=False))
    before=count(views);n=len(tokenizer.encode(original['Tool Response'],add_special_tokens=False));receipt=None
    keep=n
    while count(views)>prompt_budget and keep>min_retained_tokens:
        keep=max(min_retained_tokens,keep-(count(views)-prompt_budget)-24)
        effective['Tool Response'],receipt=observation_head_tail(original['Tool Response'],tokenizer,keep)
        views,mask=build_views(effective)
    if count(views)>prompt_budget:raise ValueError('protected_serialized_context_overflow')
    if {k:v for k,v in effective.items() if k!='Tool Response'}!={k:v for k,v in original.items() if k!='Tool Response'}:
        raise AssertionError('Trusted source field changed')
    return copy.deepcopy({'views':views,'mask_provenance':mask,'effective_tool_response':effective['Tool Response'],
        'original_prompt_tokens':before,'effective_prompt_tokens':count(views),'prompt_budget':prompt_budget,
        'edits':[] if receipt is None else [{'source_field':'Tool Response',**receipt}],
        'policy':'source-Tool-Response-head-tail-v1','trusted_user_and_schema_unchanged':True})
