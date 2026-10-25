"""Single-proposal without-Thought scoring with exact causal prefix reuse."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import time

from .mechanism_probe import prepare_event,TorchBackend
from .probe import aggregate_token_logprobs

VIEWS=('original','observation_masked','user_masked')


class PrefixSliceError(ValueError):pass


def required_views(alpha_name,alpha_arg):
    if not all(math.isfinite(a) and a>=0 for a in (alpha_name,alpha_arg)):raise ValueError('Invalid alpha')
    return VIEWS[1:] if alpha_name==alpha_arg==1 else VIEWS


def prepare_slices(tokenizer,contexts,name,full,*,views,max_input_tokens=8192):
    if not full.startswith(name):raise PrefixSliceError('Name text is not a strict full-call prefix')
    if len(full)<=len(name):raise PrefixSliceError('Full call must include arguments')
    result={}
    for view in views:
        a=prepare_event(tokenizer,contexts[view],name,max_input_tokens=max_input_tokens)
        b=prepare_event(tokenizer,contexts[view],full,max_input_tokens=max_input_tokens)
        n=len(a['candidate_token_ids'])
        if b['candidate_token_ids'][:n]!=a['candidate_token_ids'] or b['input_ids'][:len(a['input_ids'])]!=a['input_ids']:
            raise PrefixSliceError('BPE boundary changed; use separately scored targets')
        result[view]={'name':a,'full':b,'name_tokens':n}
    candidates=[x['full']['candidate_token_ids'] for x in result.values()]
    if any(x!=candidates[0] for x in candidates):raise PrefixSliceError('Different targets across views')
    return copy.deepcopy(result)


def axis_risk(values,alpha):
    if alpha==1:return values['user_masked']-values['observation_masked']
    return (1-alpha)*values['original']+alpha*values['user_masked']-values['observation_masked']


def axis_features(scores,alpha_name=1.,alpha_arg=1.):
    # The argument increment subtracts full/name at THE SAME alpha_arg. Mixing
    # name(alpha_name) into that subtraction would change its estimand.
    values={kind:{v:x[kind]['length_normalized_loglik'] for v,x in scores.items()} for kind in ('name','full')}
    n=axis_risk(values['name'],alpha_name)
    na=axis_risk(values['name'],alpha_arg);fa=axis_risk(values['full'],alpha_arg)
    return {'Risk_name':n,'DeltaRisk_arg':fa-na,'Risk_name_at_alpha_arg':na,'Risk_full_at_alpha_arg':fa,
        'alpha_name':alpha_name,'alpha_arg':alpha_arg,'original_cancels':alpha_name==alpha_arg==1,
        'argument_definition':'R_full(alpha_arg)-R_name(alpha_arg), both length-normalized'}


def clone_cache(cache):
    """Fork all mutable cache tensors; never extend the shared base in place."""
    if hasattr(cache,'to_legacy_cache'):cache=cache.to_legacy_cache()
    if isinstance(cache,tuple):return tuple(clone_cache(x) for x in cache)
    if isinstance(cache,list):return [clone_cache(x) for x in cache]
    if hasattr(cache,'detach'):return cache.detach().clone()
    return copy.deepcopy(cache)


class SharedPrefixBackend(TorchBackend):
    def score_shared(self,rows):
        import torch
        # Find the exact common prefix of CONTEXT token IDs only. A proposed
        # action/argument may never be silently absorbed into the shared cache.
        contexts=[r['input_ids'][:r['input_token_count']] for r in rows]
        p=0
        for tokens in zip(*contexts):
            if len(set(tokens))!=1:break
            p+=1
        if not p:raise ValueError('No common context prefix; use uncached scoring')
        original_training=self.model.training;self.model.eval()
        try:
            with torch.inference_mode():
                common=torch.tensor([contexts[0][:p]],device=self.device())
                pre=self.model(input_ids=common,attention_mask=torch.ones_like(common),use_cache=True,return_dict=True)
                cache=clone_cache(pre.past_key_values);last=pre.logits[0,-1].float().clone();del pre
                result=[]
                for row in rows:
                    suffix=row['input_ids'][p:]
                    inputs=torch.tensor([suffix],device=self.device())
                    output=self.model(input_ids=inputs,attention_mask=torch.ones((1,p+len(suffix)),dtype=torch.long,device=self.device()),
                        position_ids=torch.arange(p,p+len(suffix),device=self.device())[None,:],
                        past_key_values=clone_cache(cache),use_cache=True,return_dict=True)
                    vectors=[]
                    for pos in row['prediction_positions']:
                        if pos<p-1:raise ValueError('Target lies inside shared cache')
                        vectors.append(last if pos==p-1 else output.logits[0,pos-p].float())
                    logits=torch.stack(vectors);targets=torch.tensor(row['candidate_token_ids'],device=logits.device)
                    result.append(torch.log_softmax(logits,dim=-1).gather(1,targets[:,None]).squeeze(1).cpu().tolist())
                    del output,inputs,logits
                return result,{'shared_prefix_tokens':p,'model_forward_calls':1+len(rows),
                    'processed_input_tokens':p+sum(len(r['input_ids'])-p for r in rows),
                    'cache_forked_per_view':True,'shared_prefix_sha256':hashlib.sha256(json.dumps(contexts[0][:p]).encode()).hexdigest()}
        finally:
            if original_training:self.model.train()


class DeploymentScorer:
    def __init__(self,tokenizer,model,*,max_input_tokens=8192,model_binding='unbound-software-fixture'):
        self.tokenizer=tokenizer;self.backend=SharedPrefixBackend(model);self.max_input_tokens=max_input_tokens;self.model_binding=model_binding

    def score(self,contexts,name_target,full_target,*,alpha_name=1.,alpha_arg=1.,strategy='serial',original_reuse=None):
        if strategy not in ('serial','batch2','shared_kv'):raise ValueError('Unknown strategy')
        memory=self.backend.memory_start();start=time.perf_counter();views=required_views(alpha_name,alpha_arg)
        prepared=prepare_slices(self.tokenizer,contexts,name_target,full_target,views=views,max_input_tokens=self.max_input_tokens)
        reused={};active=list(views)
        if original_reuse is not None and 'original' in views:
            proof=copy.deepcopy(original_reuse);row=prepared['original']['full']
            for key,expected in [('input_ids',row['input_ids']),('candidate_token_ids',row['candidate_token_ids']),
                ('model_binding',self.model_binding),('prefix_level','without_thought'),('normalization','full_vocab_fp32_no_sampling_transform')]:
                if proof.get(key)!=expected:raise ValueError('Original logprob reuse proof mismatch: '+key)
            logs=proof['token_logprobs']
            if len(logs)!=len(row['candidate_token_ids']) or not all(math.isfinite(x) and x<=1e-6 for x in logs):raise ValueError('Invalid reused scores')
            reused['original']=logs;active.remove('original')
        preparation_end=time.perf_counter();rows=[prepared[v]['full'] for v in active];values=[];sizes=[]
        physical_tokens=0;metadata={'model_forward_calls':0,'cache_reused':False}
        if strategy=='shared_kv':
            values,meta=self.backend.score_shared(rows);metadata.update(meta,cache_reused=True);physical_tokens=meta['processed_input_tokens'];sizes=[1]*(1+len(rows))
        else:
            batch=2 if strategy=='batch2' else 1
            pad=self.tokenizer.pad_token_id
            for offset in range(0,len(rows),batch):
                selected=rows[offset:offset+batch];width=max(len(x['input_ids']) for x in selected)
                ids=[x['input_ids']+[pad]*(width-len(x['input_ids'])) for x in selected]
                attention=[[1]*len(x['input_ids'])+[0]*(width-len(x['input_ids'])) for x in selected]
                values.extend(self.backend.score_batch(ids,attention,selected));sizes.append(len(selected));physical_tokens+=len(selected)*width
            metadata['model_forward_calls']=len(sizes)
        full_logs={**reused,**dict(zip(active,values))};scores={}
        for view,logs in full_logs.items():
            item=prepared[view];n=item['name_tokens']
            if len(logs)!=len(item['full']['candidate_token_ids']):raise ValueError('Incomplete candidate probabilities')
            scores[view]={kind:{**item[kind],**aggregate_token_logprobs(logs[:n] if kind=='name' else logs)} for kind in ('name','full')}
        self.backend.synchronize();end=time.perf_counter()
        return copy.deepcopy({'scores':scores,'features':axis_features(scores,alpha_name,alpha_arg),'strategy':strategy,
            'prefix_level':'without_thought','original_generation_scores_reused':bool(reused),
            'cost':{'preparation_ms':(preparation_end-start)*1000,'forward_ms':(end-preparation_end)*1000,
                'total_ms':(end-start)*1000,'scored_view_rows':len(active),'batch_sizes':sizes,
                'physical_input_tokens':physical_tokens,'memory':self.backend.memory_end(memory),**metadata},
            'model_binding':self.model_binding,'prefix_slice_validated':True,'branch_argmax_used':False})
