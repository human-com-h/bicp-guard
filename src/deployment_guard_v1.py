"""Trusted pre-execution routing, deterministic grounding and conservative adapters."""
from __future__ import annotations
import copy
from datetime import datetime,timezone
from decimal import Decimal,InvalidOperation
import hashlib
import json
import math
import re
import unicodedata

from .mechanism_v3 import parse_call_v3
from .mechanism_v4 import DISCOVERY,tool_layer,truncate_observations
from .parsing import validate_schema

PAYLOAD_FIELDS={'EmailArchive':{'email_id':'entity_id'},
    'CalendarCreateEvent':{'title':'title','start':'start','attendees':'attendees'},
    'FinancePayInvoice':{'invoice_id':'entity_id','amount':'amount'},
    'MessagingForwardMessage':{'message_id':'entity_id','text':'text'},
    'DocumentsRouteDocument':{'document_id':'entity_id'},'AdminExportData':{}}
CONDITION_FIELDS={'EmailArchive':('label','sender_domain','approved','quarantined','attachment_count'),
    'CalendarCreateEvent':('event_type','confirmed','sender_domain','start','attendees','online'),
    'FinancePayInvoice':('amount','vendor_id','approval_status','currency','disputed','approved'),
    'MessagingForwardMessage':('urgency','category','severity','tags','customer_approved','routing_label'),
    'DocumentsRouteDocument':('category','signed','revision_status','retention_date','review_score','routing_label','recommended_folder')}


def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest()


def normalization_kind(field):
    if field in ('start','retention_date'):return 'datetime'
    if field in ('amount',):return 'number'
    if field in ('folder',):return 'path'
    if field in ('recipient',):return 'email'
    return 'text'


def normalized(value,kind='text'):
    if isinstance(value,list):return ('list',tuple(normalized(x,'email' if isinstance(x,str) and '@' in x else 'text') for x in value))
    if isinstance(value,bool):return ('bool',value)
    if isinstance(value,(int,float,Decimal)) or kind=='number':
        try:
            number=Decimal(str(value))
            if not number.is_finite():raise ValueError('Nonfinite number')
            return ('number',str(number.normalize()))
        except InvalidOperation:raise ValueError('Invalid numeric value')
    if not isinstance(value,str):return ('other',repr(value))
    if kind=='datetime':
        try:
            d=datetime.fromisoformat(value.strip().replace('Z','+00:00'))
            return ('aware' if d.tzinfo else 'naive',d.astimezone(timezone.utc).isoformat() if d.tzinfo else d.isoformat())
        except ValueError:raise ValueError('Invalid ISO datetime')
    if kind=='path':return ('path',unicodedata.normalize('NFC',value))
    if kind=='email':
        if '@' not in value:raise ValueError('Invalid email')
        local,domain=value.rsplit('@',1);return ('email',local+'@'+domain.lower())
    return ('text',' '.join(unicodedata.normalize('NFKC',value).split()))


def contract_from_public_user(tool,user_text):
    """Narrow trusted adapter for the public controlled-task instruction format.

    No oracle/environment/expected-call/attack label is an accepted argument.
    The user condition determines required field NAMES, not hidden truth values.
    """
    if tool not in PAYLOAD_FIELDS:return None
    match=re.search(r'If and only if (.*?), ',user_text)
    condition=match.group(1) if match else ''
    required=set(PAYLOAD_FIELDS[tool].values())
    # Parse the public controlled-language field names, never an oracle record.
    required.update(re.findall(r'\b([A-Za-z_]\w*)\s+(?:is|includes)\b',condition))
    return {'version':'public-extractive-contract-v1','tool':tool,'required_evidence_fields':sorted(required),
        'argument_source_fields':copy.deepcopy(PAYLOAD_FIELDS[tool]),'all_other_arguments':'normalized_span_in_U_union_O',
        'user_instruction_sha256':hashlib.sha256(user_text.encode()).hexdigest(),'does_not_evaluate_hidden_condition_truth':True}


def _no_duplicate_keys(pairs):
    result={}
    for k,v in pairs:
        if k in result:raise ValueError('Duplicate JSON evidence key')
        result[k]=v
    return result


def grounding_check(tool,arguments,messages,contract):
    if contract is None:return {'passed':True,'scope':'not_a_configured_conditional_extractive_write','checks':[]}
    if contract['tool']!=tool:raise ValueError('Grounding contract/tool mismatch')
    users=[(i,m['content']) for i,m in enumerate(messages) if m['role']=='user']
    observations=[(i,m['content']) for i,m in enumerate(messages) if m['role']=='tool']
    sources=[('U',i,s) for i,s in users]+[('O',i,s) for i,s in observations]
    record={};record_index=None
    if observations:
        record_index,body=observations[-1]
        try:
            record,_=json.JSONDecoder(object_pairs_hook=_no_duplicate_keys).raw_decode(body.lstrip())
            if not isinstance(record,dict):record={}
        except (ValueError,TypeError):record={}
    checks=[];missing=[]
    for field in contract['required_evidence_fields']:
        value=record.get(field)
        present=field in record and value is not None and not (isinstance(value,str) and not value.strip())
        checks.append({'evidence_field':field,'present':present,'source_tool_message':record_index})
        if not present:missing.append('required_evidence:'+field)
    def span(value,kind):
        want=normalized(value,kind)
        for source,index,text in sources:
            # Typed JSON values in any JSON object/array are evidence of presence,
            # not authorization. The O-injected destination must still pass this
            # necessary check and be evaluated by the provenance probe.
            for match in re.finditer(r'"(?:[^"\\]|\\.)*"',text):
                try:
                    if normalized(json.loads(match.group()),kind)==want:return {'source':source,'message_index':index,'raw_span':[match.start(),match.end()],'match':'json_value_or_string'}
                except (ValueError,TypeError):pass
            if kind=='datetime':
                candidates=re.finditer(r'\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)?',text)
            elif kind=='email':candidates=re.finditer(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",text)
            elif kind=='number':candidates=re.finditer(r'(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])',text)
            else:
                normalized_text=' '.join(unicodedata.normalize('NFKC',text).split()) if kind!='path' else unicodedata.normalize('NFC',text)
                literal=want[1]
                if not isinstance(literal,str):continue
                match=re.search(r'(?<![\w@/])'+re.escape(literal)+r'(?![\w@/])',normalized_text)
                if match:return {'source':source,'message_index':index,'normalized_span':[match.start(),match.end()],'match':'normalized_literal'}
                continue
            for match in candidates:
                try:
                    if normalized(match.group(),kind)==want:return {'source':source,'message_index':index,'raw_span':[match.start(),match.end()],'match':'normalized_typed_span'}
                except ValueError:pass
        return None
    for field,value in arguments.items():
        kind=normalization_kind(field);source_field=contract['argument_source_fields'].get(field)
        try:
            if source_field:
                found=source_field in record and normalized(value,kind)==normalized(record[source_field],kind)
                proof={'match':'typed_record_field','source':'O','message_index':record_index,'evidence_field':source_field} if found else None
            else:proof=span(value,kind)
        except (ValueError,TypeError):proof=None
        checks.append({'argument_field':field,'grounded':proof is not None,'proof':proof})
        if proof is None:missing.append('ungrounded_argument:'+field)
    return copy.deepcopy({'passed':not missing,'missing':missing,'checks':checks,
        'reason':'grounded_presence_not_authorization' if not missing else 'missing_or_unanchored_evidence',
        'only_user_and_tool_messages_used':True,'thought_used':False,'contract_sha256':digest(contract)})


def parse_deployment(output,candidates,*,reached_token_limit=False):
    if not reached_token_limit:
        match=re.fullmatch(r'(?:Thought:[\s\S]*?\n)?Action:[ \t]*Final Answer:[ \t]*([^\n]+(?:\n(?!Action:|Action Input:|Observation:|Final Answer:|<tool_call>)[^\n]+)*)\s*',output.strip())
        if match and not re.search(r'(?:^|\n)(?:Action Input:|Observation:|<tool_call>)',output):
            # Exactly one action marker: never reinterpret an actual tool call.
            if len(re.findall(r'(?m)^Action:',output))==1:
                return {'kind':'final_answer','text':match.group(1).strip(),'format_recovery':'wrapped_final_answer',
                    'synthetic_marker_inserted':False,'actual_tool_call_repaired':False}
    return copy.deepcopy(parse_call_v3(output,candidates,reached_token_limit=reached_token_limit))


def prepare_observations(messages,tokenizer,*,prompt_budget=7424,structured_tools=('list_files',)):
    """Remove long bodies from all objects, preserving all metadata and objects."""
    original=copy.deepcopy(messages);result=copy.deepcopy(messages);edits=[]
    count=lambda ms:len(tokenizer.encode(tokenizer.apply_chat_template(ms,tokenize=False,add_generation_prompt=True),add_special_tokens=False))
    before=count(result)
    if before>prompt_budget:
        import yaml
        for i,m in enumerate(original):
            if m['role']!='tool' or m.get('name') not in structured_tools:continue
            if len(m['content'])>2_000_000:raise ValueError('Observation parse budget exceeded')
            values=yaml.safe_load(m['content'])
            if not isinstance(values,list) or not values or not all(isinstance(x,dict) and 'id_' in x for x in values):
                raise ValueError('Unsupported structured observation; no silent metadata truncation')
            metadata=[{k:v for k,v in x.items() if k!='content'} for x in values]
            projected=yaml.safe_dump(metadata,sort_keys=True,allow_unicode=True)
            result[i]['content']=projected
            if yaml.safe_load(projected)!=metadata:raise ValueError('Metadata changed on serialization')
            edits.append({'message_index':i,'objects_before':len(values),'objects_after':len(metadata),
                'metadata_sha256':digest(metadata),'original_body_sha256':digest(m['content']),
                'effective_body_sha256':digest(projected),'rule':'all_objects_all_metadata_remove_content_uniformly'})
        if count(result)>prompt_budget:
            # Once structured metadata is preserved, never fall back to a crop
            # that silently deletes it. Unsupported unstructured inputs retain
            # the old explicit O-only boundary policy.
            if edits:raise ValueError('Protected structured metadata exceeds prompt budget; NEED_INFORMATION')
            return {**truncate_observations(original,tokenizer,prompt_budget=prompt_budget),'structured_projection':False}
    if any(a!=b for a,b in zip(original,result) if a['role']!='tool'):raise AssertionError('Protected message changed')
    return copy.deepcopy({'messages':result,'original_prompt_tokens':before,'effective_prompt_tokens':count(result),
        'prompt_budget':prompt_budget,'edits':edits,'structured_projection':bool(edits),'protected_messages_unchanged':True})


def guard_proposal(parsed,messages,schemas,*,scope_allowed,probe,policy=None,contract=None,reviewed_tags=None):
    """Lazy scorer is invoked only after scope/Schema/grounding, never argmax."""
    owned=copy.deepcopy(parsed);base={'branch_argmax_used':False,'actual_tool_executed':False}
    if owned.get('kind')=='final_answer':return {**base,'decision':'terminal','text':owned['text'],'extra_forward_calls':0}
    if owned.get('kind')!='tool_call':return {**base,'decision':'NEED_INFORMATION','reason':'invalid_proposal','extra_forward_calls':0}
    tool=owned['tool'];args=owned.get('arguments')
    schema=validate_schema(args,schemas.get(tool)) if tool in schemas and isinstance(args,dict) else {'valid':False}
    if isinstance(args,dict) and any(isinstance(x,float) and not math.isfinite(x) for x in args.values()):schema['valid']=False
    if not schema['valid']:return {**base,'decision':'NEED_INFORMATION','reason':'schema','extra_forward_calls':0}
    if not scope_allowed:return {**base,'decision':'block_scope','extra_forward_calls':0}
    layer=tool_layer(tool,reviewed_tags)
    if (reviewed_tags or {}).get(tool,{}).get('risk_tag')=='sensitive_or_private_read':layer='protected_sensitive_read'
    base['tool_layer']=layer
    if layer==DISCOVERY:return {**base,'decision':'allow_scoped_discovery','extra_forward_calls':0,'probe_used_for_blocking':False}
    if contract is None:
        contract=contract_from_public_user(tool,'\n'.join(m['content'] for m in messages if m['role']=='user'))
    grounding=grounding_check(tool,owned['arguments'],messages,contract);base['grounding']=grounding
    if not grounding['passed']:return {**base,'decision':'NEED_INFORMATION','reason':'grounding','extra_forward_calls':0}
    if policy is None:return {**base,'decision':'NEED_INFORMATION','reason':'pending_separate_calibration','extra_forward_calls':0}
    if not policy.get('frozen') or policy.get('selection_split')!='calibration':raise ValueError('Separately frozen calibration policy required')
    for key in ('alpha_name','alpha_arg','theta_name','theta_arg'):
        if not math.isfinite(policy[key]):raise ValueError('Nonfinite policy parameter')
    result=copy.deepcopy(probe(owned,policy['alpha_name'],policy['alpha_arg']));f=result['features']
    if not policy.get('software_fixture'):
        binding=policy.get('bindings',{}).get('model_configuration_sha256')
        if not binding or result.get('model_binding')!=binding:raise ValueError('Frozen policy/scorer model binding mismatch')
    if any(not math.isfinite(f[k]) for k in ('Risk_name','DeltaRisk_arg')):raise ValueError('Nonfinite probe result')
    if f.get('alpha_name')!=policy['alpha_name'] or f.get('alpha_arg')!=policy['alpha_arg']:raise ValueError('Scorer/policy alpha mismatch')
    block_name=f['Risk_name']>policy['theta_name'];block_arg=f['DeltaRisk_arg']>policy['theta_arg']
    return copy.deepcopy({**base,'decision':'NEED_INFORMATION' if block_name or block_arg else 'allow_protected',
        'name_gate':block_name,'argument_gate':block_arg,'probe':result,
        'extra_forward_calls':result['cost']['model_forward_calls'],'probe_used_for_blocking':True})


def evaluate_proposal(output,messages,schemas,*,scope_allowed,scorer,policy=None,
                      contract=None,reviewed_tags=None,view_builder=None,strategy='serial',reached_token_limit=False):
    """Local pre-execution adapter. No model generation or real tool execution.

    view_builder is a trusted source adapter: structured messages use exact
    message masks; InjecAgent MUST use its pinned source-field template adapter.
    It accepts only public messages; evaluation labels are not an input.
    """
    from .mechanism_v3 import extract_proposal_v3
    from .run_bicp import counterfactual_messages
    owned=copy.deepcopy(messages);parsed=parse_deployment(output,list(schemas),reached_token_limit=reached_token_limit)
    def probe(call,an,aa):
        proposal=extract_proposal_v3(output,call)
        if view_builder is None:
            if not any(m['role']=='tool' for m in owned):raise ValueError('Pinned source-field adapter required for embedded observations')
            views,mask=counterfactual_messages(copy.deepcopy(owned))
        else:views,mask=view_builder(copy.deepcopy(owned))
        contexts={k:scorer.tokenizer.apply_chat_template(v,tokenize=False,add_generation_prompt=True)+'Action:' for k,v in views.items()}
        score=scorer.score(contexts,proposal['tool_name_target'],proposal['full_call_target'],alpha_name=an,alpha_arg=aa,strategy=strategy)
        score['mask_provenance']=copy.deepcopy(mask)
        return score
    from .deployment_probe_v1 import PrefixSliceError
    try:
        result=guard_proposal(parsed,owned,schemas,scope_allowed=scope_allowed,probe=probe,policy=policy,
            contract=contract,reviewed_tags=reviewed_tags)
    except PrefixSliceError as exc:
        result={'decision':'NEED_INFORMATION','reason':'token_prefix_ineligible','error':str(exc),
            'actual_tool_executed':False,'extra_forward_calls':0,'branch_argmax_used':False}
    result['parsed_call']=copy.deepcopy(parsed);result['raw_output_sha256']=hashlib.sha256(output.encode()).hexdigest()
    return copy.deepcopy(result)
