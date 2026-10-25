"""Reconstruct empirical tables from accepted observation-level source data."""
import argparse,csv,json,math,statistics,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.deployment_probe_v1 import axis_features
from src.mechanism_v4 import factorial_contrasts,exact_positive_sign_test

def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def yes(value):return value is True or str(value).lower()=='true'
def quantile(x,p):
    z=sorted(x);q=(len(z)-1)*p;i=int(q);return z[i]+(z[min(i+1,len(z)-1)]-z[i])*(q-i)
def auc(selected,field):
    positive=[float(r[field]) for r in selected if not yes(r['legal_call'])]
    negative=[float(r[field]) for r in selected if yes(r['legal_call'])]
    if not positive or not negative:raise ValueError('AUC group requires both labels')
    return sum((a>b)+.5*(a==b) for a in positive for b in negative)/(len(positive)*len(negative))
def write_csv(path,rs):
    fields=list(dict.fromkeys(k for r in rs for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rs)

def build(longitudinal_source=None):
    src=ROOT/'results/source';tables={}
    router=read(ROOT/'configs/tool_risk_layers.json')['suites']
    tables['table_01_router']=[{'suite':s,**dict(Counter(v['layer'] for v in tools.values()))} for s,tools in sorted(router.items())]
    policy=read(ROOT/'configs/frozen_policy.json')
    tables['table_02_policy']=[{'parameter':k,'value':policy[k]} for k in ('alpha_name','theta_name','alpha_arg','theta_arg')]
    outcomes=rows(src/'v1/outcomes.csv');v1=[]
    for method in dict.fromkeys(r['method'] for r in outcomes):
        out={'method':method}
        for field,subset,label in [('attack_success','injecagent/total','injecagent'),('task_success','conditional/all','conditional_utility'),('unsafe_action_executed','conditional/all','unsafe')]:
            r=next(r for r in outcomes if (r['method'],r['field'],r['subset'])==(method,field,subset))
            counts=json.loads(r['family_counts']);out[label+'_count']=sum(v[0] for v in counts.values());out[label+'_n']=int(r['records']);out[label+'_unknown']=int(r['unknown'])
        v1.append(out)
    tables['table_04_v1']=v1
    thought=rows(src/'mechanism/thought_scores.csv');groups=sorted({r['group_id'] for r in thought})
    tables['table_05_prefix']=[{'target':metric,'prefix':prefix,'macro_auc':statistics.mean(auc([r for r in thought if r['group_id']==g and r['prefix']==prefix],metric) for g in groups),'groups':len(groups),'proposals':sum(r['prefix']==prefix for r in thought)} for metric in ('Risk_name','Risk_full') for prefix in ('with_actual_thought','with_neutral_thought','without_thought')]
    cells=rows(src/'mechanism/factorial_cells.csv');families=defaultdict(list)
    for r in cells:families[r['unit_id']].append(r)
    contrasts=[]
    for family,rr in sorted(families.items()):
        contrasts.append({'family':family,'domain':rr[0]['domain'],**factorial_contrasts({(r['authorization_source'],r['appearance']):float(r['Risk_full']) for r in rr})})
    tables['table_06_reversal_families']=contrasts
    domains=defaultdict(list)
    for r in contrasts:domains[r['domain']].append(r)
    tables['table_06_reversal_tests']=[]
    for metric in ('D_neutral_bad','C_neutral_source'):
        means=[statistics.mean(r[metric] for r in rr) for rr in domains.values()]
        tables['table_06_reversal_tests'].append({'endpoint':metric,'positive_families':sum(r[metric]>0 for r in contrasts),'families':len(contrasts),**exact_positive_sign_test(means)})
    calibration=read(src/'calibration/accepted_score_table.json')['rows'];cal=[];gmail=[]
    for label,rr in [('legal',[r for r in calibration if r['legal_call']]),('cross_tool',[r for r in calibration if not r['legal_call'] and r['attack_type']=='cross_tool']),('same_tool_argument',[r for r in calibration if not r['legal_call'] and r['attack_type']=='same_tool_argument'])]:
        rejected=name=argument=0
        for r in rr:
            f=axis_features(r['scores'],policy['alpha_name'],policy['alpha_arg']);n=f['Risk_name']>policy['theta_name'];a=f['DeltaRisk_arg']>policy['theta_arg']
            rejected+=bool(r['preprobe_rejected'] or n or a);name+=n;argument+=a
            if r['tool_layer']=='protected_sensitive_read' and r['legal_call']:gmail.append({'proposal_id':r['proposal_id'],**f,'rejected':bool(n or a)})
        cal.append({'label':label,'n':len(rr),'rejected':rejected,'name_reject':name,'argument_reject':argument})
    tables['calibration_events']=cal;tables['table_11_gmail']=gmail
    pilot=rows(src/'pilot/events.csv');pilot_table=[]
    for a in 'ABC':
        ev=[e for e in pilot if e['arm']==a];conditional=[e for e in ev if e['source']=='conditional']
        legal=[e for e in conditional if yes(e['legal_call']) and yes(e['protected'])]
        adopted=[e for e in conditional if yes(e['target_adopted'])]
        historical=[e for e in ev if e['source']=='injecagent_development_replay' and e['legal_call']=='False']
        discovery=[e for e in ev if e['source']=='native_discovery_development_replay']
        pilot_table.append({'arm':a,'conditional_n':len(conditional),'unsafe':sum(yes(e['unsafe_dispatch']) for e in conditional),'utility':sum(yes(e['task_success']) for e in conditional),
            'adopted_malicious_n':len(adopted),'adopted_malicious_executed':sum(yes(e['executed']) for e in adopted),'legal_protected_n':len(legal),'legal_retained':sum(yes(e['executed']) for e in legal),
            'historical_unauthorized_n':len(historical),'historical_unauthorized_dispatched':sum(yes(e['executed']) for e in historical),'discovery_n':len(discovery),'discovery_retained':sum(yes(e['executed']) for e in discovery)})
    tables['table_08_pilot']=pilot_table
    source=Path(longitudinal_source) if longitudinal_source else src/'longitudinal'
    units=rows(source/'units.csv');events=rows(source/'events.csv');standard=[];variants=[]
    for a in 'ABC':
        u=[r for r in units if r['arm']==a];inj=[r for r in u if r['source']=='injecagent'];cond=[r for r in u if r['source']=='conditional']
        ev=[e for e in events if e['scope']=='L' and e['arm']==a and e['stage']=='stage1'];ce=[e for e in ev if e['source']=='conditional']
        legal=[e for e in ce if yes(e['legal_call']) and e['layer']!='read_only_discovery'];adopted=[e for e in ce if yes(e['target_destination_adopted'])]
        missing=[e for e in ce if e['variant']=='missing_evidence'];benign=[r for r in cond if not yes(r['is_attack'])]
        r={'arm':a,'injecagent_n':len(inj),'confirmed':sum(yes(x['attack_success_confirmed']) for x in inj),'unknown':sum(yes(x['attack_outcome_unknown']) for x in inj),
           'conditional_n':len(cond),'unsafe':sum(yes(x['unsafe_action_executed']) for x in cond),'utility':sum(yes(x['task_success']) for x in cond),'benign_n':len(benign),'benign_utility':sum(yes(x['task_success']) for x in benign),
           'adopted_malicious_n':len(adopted),'adopted_malicious_executed':sum(yes(e['executed']) for e in adopted),'missing_evidence_n':len(missing),'missing_evidence_writes':sum(yes(e['executed']) for e in missing),
           'legal_protected_n':len(legal),'legal_retained':sum(yes(e['executed']) for e in legal)}
        for kind in ('dh','ds'):
            subset=[x for x in inj if x['attack_type']==kind];r[kind+'_n']=len(subset);r[kind+'_confirmed']=sum(yes(x['attack_success_confirmed']) for x in subset);r[kind+'_unknown']=sum(yes(x['attack_outcome_unknown']) for x in subset)
        standard.append(r)
        for variant in sorted({r['variant'] for r in cond}):
            rr=[r for r in cond if r['variant']==variant];variants.append({'arm':a,'variant':variant,'n':len(rr),'utility':sum(yes(x['task_success']) for x in rr),'unsafe':sum(yes(x['unsafe_action_executed']) for x in rr)})
    tables['table_09_standard']=standard;tables['table_13_variants']=variants
    costs=[]
    for source_name in sorted({e['source'] for e in events if yes(e['probe_used'])}):
        rr=[float(e['probe_total_ms'])/1000 for e in events if e['source']==source_name and yes(e['probe_used'])]
        costs.append({'source':source_name,'probes':len(rr),'median_seconds':statistics.median(rr),'p95_seconds':quantile(rr,.95)})
    tables['table_10_cost']=costs
    paired=rows(src/'calibration/preflight_paired.csv');checks=rows(src/'longitudinal/healthchecks.csv')
    tables['slicing_speedup']=[{'measurement':'calibration_paired','median_ratio':statistics.median(float(r['paired_speedup_vs_6']) for r in paired if r['method']=='sliced3')},
       {'measurement':'longitudinal_healthcheck','median_ratio':statistics.median(float(r['ratio']) for r in checks)}]
    return tables

def main(output,longitudinal_source=None):
    output=Path(output)
    if output.exists():raise ValueError('Preserve previous reconstruction; choose a NEW output directory')
    tables=build(longitudinal_source);output.mkdir(parents=True)
    for name,rs in tables.items():write_csv(output/(name+'.csv'),rs)
    (output/'tables.json').write_text(json.dumps(tables,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'tables':len(tables),'output':str(output),'source':'accepted measurement records; no fitted parameters or hardcoded result cells'}))
    return tables
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--longitudinal-source',type=Path);a=p.parse_args();main(a.output,a.longitudinal_source)
