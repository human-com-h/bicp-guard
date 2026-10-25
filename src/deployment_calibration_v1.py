"""Deterministic calibration-only dual OR gate selection. No model inference."""
from __future__ import annotations
import copy
import math
from .deployment_probe_v1 import axis_features

ALPHAS=(.5,1.,2.)
ATTACK_TYPES=('cross_tool','same_tool_argument')


def select_policy(rows,*,bindings,fpr_limit=.03,alpha_grid=ALPHAS):
    """Rows are separately labelled proposals from the calibration split only.

    A row's preprobe_rejected records scope/schema/grounding rejection, so those
    errors consume the SAME total legal-high-risk false-positive budget.
    Score-table labels must not be passed into the online guard.
    """
    rows=copy.deepcopy(rows)
    if not rows or any(r.get('split')!='calibration' for r in rows):raise ValueError('Calibration split exclusively required')
    if len({r['proposal_id'] for r in rows})!=len(rows):raise ValueError('Duplicate proposal ID')
    events=[r.get('event_sha256',r['proposal_id']) for r in rows]
    if len(set(events))!=len(events):raise ValueError('Duplicate context/target event; merge free/forced origins before fitting')
    if any(r.get('tool_layer')=='read_only_discovery' for r in rows):raise ValueError('Discovery is excluded from protected calibration')
    if any(type(r.get('legal_call')) is not bool for r in rows):raise ValueError('Unknown labels cannot be fitted')
    if any(not r['legal_call'] and r.get('attack_type') not in ATTACK_TYPES for r in rows):raise ValueError('Only prespecified injection mechanisms enter threshold fitting')
    legal=[r for r in rows if r['legal_call']]
    attacks={k:[r for r in rows if not r['legal_call'] and r.get('attack_type')==k] for k in ATTACK_TYPES}
    if not legal or any(not x for x in attacks.values()):raise ValueError('Legal and both attack mechanisms required')
    budget=math.floor(fpr_limit*len(legal)+1e-12)
    pre=sum(bool(r.get('preprobe_rejected')) for r in legal)
    if pre>budget:raise ValueError('Deterministic-stage legal FPR exceeds budget; fix contract before fitting')
    choices=[]
    for an in alpha_grid:
        for aa in alpha_grid:
            features={r['proposal_id']:axis_features(r['scores'],an,aa) for r in rows if not r.get('preprobe_rejected')}
            if any(not math.isfinite(f[k]) for f in features.values() for k in ('Risk_name','DeltaRisk_arg')):raise ValueError('Nonfinite calibration feature')
            # For strict >, observed legal values enumerate all distinct legal
            # rejection sets. nextafter(min,-inf) also covers rejecting all.
            thresholds=[]
            for key in ('Risk_name','DeltaRisk_arg'):
                values=sorted({features[r['proposal_id']][key] for r in legal if not r.get('preprobe_rejected')})
                thresholds.append([math.nextafter(values[0],-math.inf)]+values if values else [0.])
            for tn in thresholds[0]:
                for ta in thresholds[1]:
                    def rejected(r):
                        if r.get('preprobe_rejected'):return True
                        f=features[r['proposal_id']];return f['Risk_name']>tn or f['DeltaRisk_arg']>ta
                    fp=sum(rejected(r) for r in legal)
                    if fp>budget:continue
                    recall={k:sum(rejected(r) for r in a)/len(a) for k,a in attacks.items()}
                    # Preregistered: best worst-mechanism recall, mean recall,
                    # fewer legal rejections, alpha=1 lower cost, lexical ties.
                    key=(-min(recall.values()),-sum(recall.values())/2,fp,
                        0 if an==aa==1 else 1,abs(an-1)+abs(aa-1),an,aa,tn,ta)
                    choices.append((key,{'alpha_name':an,'alpha_arg':aa,'theta_name':tn,'theta_arg':ta,
                        'legal_high_risk_n':len(legal),'joint_false_positives':fp,'joint_fpr':fp/len(legal),
                        'preprobe_legal_rejections':pre,'attack_recall_by_mechanism':recall}))
    if not choices:raise ValueError('No feasible policy')
    selected=min(choices,key=lambda x:x[0])[1]
    return copy.deepcopy({**selected,'frozen':True,'selection_split':'calibration','fpr_limit':fpr_limit,
        'comparison':'>','threshold_scope':'joint_OR_including_preprobe_legal_rejections',
        'legal_independent_groups':len({r['group_id'] for r in legal}),
        'population_fpr_guaranteed':False,'bindings':bindings,'candidate_policies':len(choices)})
