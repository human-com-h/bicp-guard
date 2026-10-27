"""Recompute v2.2 scores and tables from accepted records, without a model.

The default path uses only the standard library. --bootstrap additionally uses
NumPy to reproduce the already accepted paired family-cluster intervals.
"""
import argparse,csv,hashlib,json,math,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/source/supplement_v2_2'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(name):return list(csv.DictReader((SOURCE/name).open(encoding='utf-8-sig')))
def auc(g,b):return sum((y>x)+.5*(y==x) for x in g for y in b)/(len(g)*len(b))
def reconstruct(score):
    m={k:{} for k in ('name','full','suffix')}
    for view,pair in score['scores'].items():
        a,b=pair['name'],pair['full'];n=len(a['token_logprobs']);f=len(b['token_logprobs'])
        assert 0<n<f and a['candidate_token_ids']==b['candidate_token_ids'][:n]
        assert a['token_logprobs']==b['token_logprobs'][:n]
        for key,event in [('name',a),('full',b)]:
            logs=event['token_logprobs'];assert len(logs)==len(event['candidate_token_ids'])
            assert all(math.isfinite(v) and v<=1e-6 for v in logs)
            total=math.fsum(logs);assert math.isclose(total,event['joint_loglik'],abs_tol=1e-6)
            m[key][view]=total/len(logs)
            assert math.isclose(m[key][view],event['length_normalized_loglik'],abs_tol=1e-7)
        m['suffix'][view]=math.fsum(b['token_logprobs'][n:])/(f-n)
    def risk(x,a):return (1-a)*x['original']-x['observation_masked']+a*x['user_masked']
    out={}
    for a in (1,2):
        out.update({f'{k}_{a}':risk(v,a) for k,v in m.items()})
        out[f'arg_{a}']=out[f'full_{a}']-out[f'name_{a}']
    assert math.isclose(out['name_1'],score['features']['Risk_name'],abs_tol=1e-7)
    assert math.isclose(out['arg_2'],score['features']['DeltaRisk_arg'],abs_tol=1e-7)
    return out
def main(output,bootstrap=False):
    manifest=read(ROOT/'resources/supplement_v2_2_manifest.json')
    for name,h in {**manifest['artifacts'],**manifest['source_code']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    with zipfile.ZipFile(ROOT/'resources/supplement_v2_2_inputs.zip') as z:
        for name,h in manifest['frozen_input_files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
        schedule=json.loads(z.read('data/supplement/schedule.json'))
        policy=json.loads(z.read('configs/frozen_policy.json'));suffix=json.loads(z.read('configs/suffix_policy.json'))
    with zipfile.ZipFile(ROOT/'results/raw/supplement_v2_2.zip') as z:
        raw=z.read('records.jsonl');assert hashlib.sha256(raw).hexdigest()==manifest['raw_records_sha256']
        records=[json.loads(l) for l in raw.splitlines() if l.strip()]
        labels=json.loads(z.read('contract_labels.json'))
    assert len(records)==156 and len({(r['phase'],r['block']) for r in records})==156
    inputs={(phase,b['block']):b for phase in ('qwen','mistral') for b in schedule[phase]}
    expected_contract={(r['unit_id'],r['contract'],r['proposal']):r for r in rows('contract_scores.csv')}
    expected_thought={(r['unit_id'],r['prefix']):r for r in rows('thought_scores.csv')}
    expected_factor={(r['family'],r['context'],r['candidate']):r for r in rows('factorial_scores.csv')}
    contract=[];thought=[];factor=[];native=[];cells=0;native_scores=0
    for r in records:
        b=inputs[r['phase'],r['block']]
        if r['kind']=='native_trajectory':
            native.append(r)
            for e in r['events']:
                if e.get('decision',{}).get('probe'):reconstruct(e['decision']['probe']);native_scores+=1
            continue
        for c in r['cells']:
            x=reconstruct(c['score']);cells+=1
            if r['kind']=='contract_pair':
                key=(r['unit_id'],c['contract'],c['proposal']);expected=expected_contract[key]
                rd=not(x['name_1']>policy['theta_name'] or x['arg_2']>policy['theta_arg'])
                zd=not(x['name_1']>policy['theta_name'] or x['suffix_2']>suffix['theta_suffix'])
                assert rd==c['residual_probe_only_allow'] and zd==c['suffix_probe_only_allow']
                assert rd==(expected['residual_probe_only_allow']=='True') and zd==(expected['suffix_probe_only_allow']=='True')
                if c['proposal']=='paired':
                    present=not labels[r['unit_id']]['condition_true']
                    assert c['preprobe_passed']==present and c['gate_replay_calls']==int(present)
                    assert not c['residual_full_pipeline_allow'] and not c['suffix_full_pipeline_allow']
                    if present:assert not rd and not zd
                contract.append(dict(expected,**x))
            elif r['kind']=='thought_replay':
                key=(r['unit_id'],c['prefix_level']);expected=expected_thought[key]
                thought.append(dict(expected,**x))
            else:
                key=(r['unit_id'],c['context_id'],c['candidate_id']);expected=expected_factor[key]
                factor.append(dict(expected,**x))
            for k,v in x.items():assert math.isclose(v,float(expected[k]),abs_tol=1e-7),(key,k)
    assert cells==435 and len(native)==30 and len(thought)==207 and len(factor)==40
    contract_summary=[]
    for template in ('v1','v3'):
        for gate in ('residual_probe_only_allow','suffix_probe_only_allow'):
            good=[r for r in contract if r['contract']==template and r['proposal']=='historical']
            contract_summary.append({'contract':template,'gate':gate,'retained':sum(r[gate]=='True' for r in good),'n':len(good)})
    prefix=[]
    for level in ('with_actual_thought','with_neutral_thought','without_thought'):
        for metric in ('name_1','full_1'):
            groups=[]
            for group in sorted({r['group'] for r in thought}):
                g=[r[metric] for r in thought if r['prefix']==level and r['group']==group and r['legal']=='True']
                bad=[r[metric] for r in thought if r['prefix']==level and r['group']==group and r['legal']=='False']
                groups.append(auc(g,bad))
            value=sum(groups)/len(groups)
            expected=next(r for r in rows('thought_macro.csv') if r['prefix']==level and r['metric']==metric)
            assert math.isclose(value,float(expected['macro_auc']),abs_tol=1e-12)
            prefix.append({'prefix':level,'metric':metric,'macro_auc':value})
    effects=[]
    for family in sorted({r['family'] for r in factor}):
        x={(r['authorization_source'],r['appearance']):r['full_1'] for r in factor if r['family']==family}
        effects.append({'family':family,'domain':family.split('.')[0],'D':x['O_only','neutral']-x['U','suspicious'],'C':x['O_only','neutral']-x['U','neutral']})
    assert all(e[k]>0 for e in effects for k in ('D','C'))
    domains=sorted({e['domain'] for e in effects});assert len(domains)==5
    native_labels={(r['task'],r['arm'],r['reserve_final_step']=='True'):r for r in rows('native_utility.csv')}
    for r in native:
        expected=native_labels[r['suite']+':'+r['task_id'],r['arm'],r['reserve_final_step']]
        assert len(r['events'])==int(expected['decision_steps']) and r['terminal']==expected['terminal']
        assert len(r['executed_calls'])==int(expected['tool_calls'])
        if r['reserve_final_step']:assert not any(e.get('reserved_final_step') for e in r['events'])
    result={'passed':True,'records':156,'fixed_scored_cells':cells,'native_probe_scores':native_scores,'native_trajectories':30,
        'contract_retention':contract_summary,'mistral_prefix':prefix,'mistral_source_contrasts':effects,'domain_IUT_p':2**-len(domains),
        'native_utility':'Accepted independent official evaluator labels retained; this entry point checks trajectory bindings, not a fresh official evaluator run.',
        'raw_records_sha256':manifest['raw_records_sha256'],'new_inference':False}
    if bootstrap:
        import numpy as np
        rng=np.random.default_rng(42);groups=sorted({r['group'] for r in contract});lookup={(r['unit_id'],r['contract']):r for r in contract if r['proposal']=='historical'}
        intervals=[];accepted=read(SOURCE/'analysis.json')['paired_family_bootstrap']
        for name,ga,gb in [('contract_effect_residual','residual_probe_only_allow','residual_probe_only_allow'),('contract_effect_suffix','suffix_probe_only_allow','suffix_probe_only_allow'),('suffix_effect_v3','residual_probe_only_allow','suffix_probe_only_allow')]:
            ca='v3' if name=='suffix_effect_v3' else 'v1';a=[]
            for group in groups:
                ids=[u for u,c in lookup if c=='v1' and lookup[u,c]['group']==group]
                a.append((sum(int(lookup[u,'v3'][gb]=='True')-int(lookup[u,ca][ga]=='True') for u in ids),len(ids)))
            a=np.array(a);draw=a[rng.integers(0,len(a),size=(10000,len(a)))].sum(axis=1);v=draw[:,0]/draw[:,1];ci=np.quantile(v,[.025,.975]).tolist()
            old=next(r for r in accepted if r['comparison']==name)
            assert np.allclose(ci,old['cluster_percentile_95'],atol=1e-12,rtol=0)
            intervals.append({'comparison':name,'cluster_percentile_95':ci})
        result['paired_cluster_bootstrap']=intervals
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':True,'fixed_scored_cells':cells,'native_probe_scores':native_scores,'bootstrap_recomputed':bootstrap,'new_inference':False}))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/supplement');p.add_argument('--bootstrap',action='store_true');a=p.parse_args();main(a.output,a.bootstrap)
