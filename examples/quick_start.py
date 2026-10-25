"""CPU-only functional example using the actual frozen guard and controlled data."""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.conditional_tasks import TOOLS, oracle_for
from src.mechanism_v3 import conditional_messages_v3
from src.deployment_guard_v1 import guard_proposal
from src.deployment_probe_v1 import axis_features

def main():
    cases=[json.loads(line) for line in (ROOT/'data/conditional/cases.jsonl').read_text(encoding='utf-8').splitlines()]
    c=next(c for c in cases if c['variant']=='missing_evidence' and c['domain']=='calendar')
    messages,schemas=conditional_messages_v3(c)
    parsed={'kind':'tool_call','tool':'CalendarCreateEvent','arguments':{'calendar_id':'office-calendar-1','title':'Project 1 planning','start':'2026-11-15T14:00:00','attendees':['owner1@example.invalid','team@example.invalid']},'format':'react','argument_format':'json'}
    def no_probe(*args):raise AssertionError('Missing evidence must stop before inference')
    decision=guard_proposal(parsed,messages,schemas,scope_allowed=True,probe=no_probe)
    assert decision['reason']=='grounding' and decision['extra_forward_calls']==0
    print(json.dumps({'example':'missing-evidence guard','decision':decision['decision'],'reason':decision['reason'],'extra_forward_calls':0,'model_loaded':False,'scientific_evaluation':False}))
    return decision
if __name__=='__main__':main()
