"""Bound v3 pilot: free continuations, historical replays, matched proposals.

No fitted gate; all external operations are official local simulations. Model
functions accept public views/targets only. Evaluation is attached afterwards.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from .mechanism_probe import (ARMS, SequenceScorer, TorchBackend, prepare_event,
    original_prefix, semantic_sha, json_segments)
from .mechanism_v3 import (BRANCH_PREFIXES, extract_proposal_v3, parse_call_v3,
    add_format_contract, native_read_label, injecagent_call_label)
from .probe import VIEW_NAMES, aggregate_token_logprobs, dual_margins
from .parsing import parse_call
from .run_mechanism_diagnostic import ImmutableGenerator, owned_generate, error_record, write_record
from .run_baseline import write_json, environment
from .run_bicp import counterfactual_messages, USER_MASK


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_lines(path):return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


class V3Scorer(SequenceScorer):
    def score_eos(self, contexts, target, *, eos_id=151645):
        """Append a verified special ID; never substitute its LF probability."""
        if set(contexts)!=set(VIEW_NAMES) or self.tokenizer.encode("<|im_end|>",add_special_tokens=False)!=[eos_id]:
            raise ValueError("Unverified EOS or views")
        memory=self.backend.memory_start();started=time.perf_counter()
        rows=[]
        for name in VIEW_NAMES:
            row=prepare_event(self.tokenizer,contexts[name],target,max_input_tokens=self.max_input_tokens-1)
            row["input_ids"].append(eos_id);row["candidate_token_ids"].append(eos_id)
            row["prediction_positions"].append(len(row["input_ids"])-2)
            row["target_offsets"].append(None)
            row["token_segments"].append({"kind":"explicit_eos","overlapping_kinds":[],"keys":[]})
            row["target_token_suffix"]=[eos_id]
            row["full_input_sha256"]=semantic_sha(row["input_ids"])
            row["full_input_hash_kind"]="token_ids_with_explicit_suffix"
            rows.append(row)
        if not all(r["candidate_token_ids"]==rows[0]["candidate_token_ids"] for r in rows):
            raise ValueError("EOS candidates differ across views")
        prepared=time.perf_counter();values={}
        for name,row in zip(VIEW_NAMES,rows):
            logs=self.backend.score_batch([row["input_ids"]],[[1]*len(row["input_ids"])],[row])[0]
            values[name]={**row,**aggregate_token_logprobs(logs)}
        self.backend.synchronize();finished=time.perf_counter()
        return copy.deepcopy({"available":True,"views":values,"target":target,"target_token_suffix":[eos_id],
            "same_candidate_token_ids":rows[0]["candidate_token_ids"],
            "margins":{f:dual_margins(values,alpha=1.,score_field=f) for f in ("joint_loglik","length_normalized_loglik")},
            "cost":{"preparation_ms":(prepared-started)*1000,"forward_ms":(finished-prepared)*1000,
                "total_ms":(finished-started)*1000,"batch_sizes":[1,1,1],"model_forward_calls":3,"scored_rows":3,
                "input_tokens":sum(r["input_token_count"] for r in rows),"candidate_tokens":sum(len(r["candidate_token_ids"]) for r in rows),
                "padded_model_input_tokens":sum(len(r["input_ids"]) for r in rows),"memory":self.backend.memory_end(memory),"cache_reused":False},
            "scope":"secondary terminal prefix plus explicit EOS; not mixed into primary argmax","threshold":None})


def safe_score(scorer,contexts,target,*,segments=(),eos=False):
    try:
        if eos:return scorer.score_eos(copy.deepcopy(contexts),target)
        return {"available":True,**scorer.score(copy.deepcopy(contexts),target,batch_size=1,segments=segments)}
    except Exception as error:
        return {"available":False,**error_record(error)}


def score_proposal(views,output,parsed,tokenizer,scorer,arm_order,*,branches=True):
    if set(views)!=set(VIEW_NAMES) or len(arm_order)!=4 or set(arm_order)!=set(ARMS):
        raise ValueError("Incomplete three-view/four-arm definition")
    owned=copy.deepcopy(views)
    rendered={k:tokenizer.apply_chat_template(v,tokenize=False,add_generation_prompt=True) for k,v in owned.items()}
    result={"messages_by_view":owned,"rendered_by_view":rendered,"arms":{},"branches":{},"eos_secondary":{},
            "oracle_in_scoring":False,"actual_arm_order":list(arm_order),"boundary_version":"v3"}
    try:
        proposal=extract_proposal_v3(output,parsed);result["proposal"]=proposal
    except Exception as error:
        proposal=None;result["proposal_missing_reason"]=f"{type(error).__name__}:{error}"
    thought=proposal["thought"] if proposal else original_prefix(output)
    result["thought_characters"]=len(thought)
    result["explicit_nonempty_thought"]=bool(re.match(r"\s*Thought:\s*\S",thought) and re.sub(r"^\s*Thought:\s*","",thought).strip())
    for arm in arm_order:
        if proposal is None:
            result["arms"][arm]={"available":False,"reason":result["proposal_missing_reason"]};continue
        prefix,kind=arm.split("__")
        contexts={k:text+(thought if prefix=="fixed_thought" else "")+"Action:" for k,text in rendered.items()}
        result["arms"][arm]=safe_score(scorer,contexts,proposal[kind+"_target"],segments=proposal[kind+"_segments"])
    if branches:
        for prefix in ("fixed_thought","without_thought"):
            contexts={k:text+(thought if prefix=="fixed_thought" else "") for k,text in rendered.items()}
            events={name:safe_score(scorer,contexts,target) for name,target in BRANCH_PREFIXES.items()}
            masses={}
            if all(v.get("available") for v in events.values()):
                for view in VIEW_NAMES:
                    logs={name:value["views"][view]["joint_loglik"] for name,value in events.items()}
                    maximum=max(logs.values());denominator=sum(math.exp(v-maximum) for v in logs.values())
                    masses[view]={k:math.exp(v-maximum)/denominator for k,v in logs.items()}
            result["branches"][prefix]={"events":events,"restricted_candidate_mass":masses,
                "exhaustive_probability_distribution":False,"action_anchor_already_in_context":False}
            result["eos_secondary"][prefix]={name:safe_score(scorer,contexts,target,eos=True)
                                            for name,target in BRANCH_PREFIXES.items() if name!="Execute"}
    return copy.deepcopy(result)


def collect_event(messages,views,tools,generator,scorer,arm_order,mask):
    before=copy.deepcopy(messages);generated=owned_generate(generator,before,tools)
    strict=parse_call(generated["output"],list(tools))
    parsed=parse_call_v3(generated["output"],list(tools),reached_token_limit=generated.get("reached_token_limit",False))
    diagnostic=score_proposal(views,generated["output"],parsed,generator.tokenizer,scorer,arm_order)
    if generated["messages"]!=before or diagnostic["messages_by_view"]!=views:raise ValueError("Mutable input history")
    return copy.deepcopy({"messages_before":before,"generation":generated,"strict_parsed_call":strict,"parsed_call":parsed,
        "diagnostic":diagnostic,"mask_provenance":copy.deepcopy(mask),"executed":False,
        "oracle_passed_to_model_or_scorer":False,"actual_external_tool_executions":0})


def collect_native(selection,native,generator,scorer,arm_order,max_steps=15):
    from .run_agentdojo_bicp import initial_messages,new_environment,json_safe,native_utility,reference_call_label
    suites,runtime_module,formatter=native;suite=suites[selection["suite"]];task=suite.user_tasks[selection["task_id"]]
    runtime=runtime_module.FunctionsRuntime(suite.tools);state,pre=new_environment(suite,task)
    messages=add_format_contract(initial_messages(task,runtime),runtime.functions)
    events=[];calls=[];final="";terminal="step_limit"
    for step in range(max_steps):
        before=copy.deepcopy(messages);obs=[i for i,m in enumerate(before) if m["role"]=="tool"]
        if obs:
            views,mask=counterfactual_messages(before,user_index=1,obs_index=obs[-1]);mask["observation_intervention_available"]=True
        else:
            views={k:copy.deepcopy(before) for k in VIEW_NAMES};views["user_masked"][1]["content"]=USER_MASK
            mask={"observation_intervention_available":False,"reason":"initial_step_identity_observation_control"}
        try:event=collect_event(before,views,runtime.functions,generator,scorer,arm_order,mask)
        except Exception as error:
            events.append({"step":step,"messages_before":before,"state_before":json_safe(state),**error_record(error,generator)})
            terminal="generation_or_collection_error";break
        event.update(step=step,state_before=json_safe(state));events.append(event);p=event["parsed_call"]
        if p["kind"]=="final_answer":
            final=p["text"];terminal="final_answer";messages.append({"role":"assistant","content":event["generation"]["output"]});break
        if p["kind"]!="tool_call":terminal="parse_failure";break
        try:
            call=runtime_module.FunctionCall(function=p["tool"],args=copy.deepcopy(p["arguments"]))
            value,error=runtime.run_function(state,call.function,call.args);formatted=formatter(value);calls.append(call)
            event.update(executed=True,tool_result=formatted,runtime_error=error,state_after=json_safe(state))
            messages.extend([{"role":"assistant","content":event["generation"]["output"]},
                {"role":"tool","name":call.function,"content":formatted if error is None else "Error: "+error}])
        except Exception as error:event["runtime_exception"]=str(error);terminal="native_runtime_exception";break
    # Reference material becomes visible to evaluation only after collection.
    reference=task.ground_truth(pre.model_copy(deep=True));preceding=[]
    result={**selection,"events":events,"initial_state":json_safe(pre),"final_state":json_safe(state),"messages":messages,
        "executed_calls":json_safe(calls),"reference_calls_evaluation_only":json_safe(reference),"final_text":final,"terminal":terminal}
    for event in events:
        p=event.get("parsed_call",{})
        if p.get("kind")=="tool_call":
            event["authorization_label"]=reference_call_label(p,reference,preceding,runtime)
            event["v3_authorization_label"]=native_read_label(result,event)
            if event.get("executed"):preceding.append(runtime_module.FunctionCall(function=p["tool"],args=copy.deepcopy(p["arguments"])))
    try:result.update(native_utility=native_utility(suite,task,pre,state,calls,final),utility_error=None)
    except Exception as error:result.update(native_utility=None,utility_error=str(error))
    return copy.deepcopy(result)


def score_forced(inputs,payloads,scorer,tokenizer,block):
    """No evaluation labels/oracle argument. Both candidates share exact inputs."""
    views=copy.deepcopy(inputs["views"])
    rendered={k:tokenizer.apply_chat_template(v,tokenize=False,add_generation_prompt=True) for k,v in views.items()}
    contexts={k:v+"Action:" for k,v in rendered.items()};candidates=[]
    for index in block["candidate_order"]:
        item=payloads[index];scores={}
        for arm in block["arm_order"]:
            prefix,kind=arm.split("__")
            if prefix!="without_thought":continue
            target=item[kind+"_target"]
            marker=target.find("Action Input:")
            segments=json_segments(target[marker+len("Action Input:"):],marker+len("Action Input:")) if kind=="full_call" else ()
            scores[arm]=safe_score(scorer,contexts,target,segments=segments)
        candidates.append({"candidate_id":item["candidate_id"],"scores":scores})
    if payloads[0]["tool_name_target"]!=payloads[1]["tool_name_target"]:raise ValueError("Not a same-tool pair")
    return copy.deepcopy({"sample_id":block["sample_id"],"views":views,"contexts":contexts,"candidates":candidates,
        "candidate_order":list(block["candidate_order"]),"oracle_passed_to_model_or_scorer":False,
        "fixed_thought_secondary":{"available":False,"reason":"optional secondary omitted uniformly in this pilot before results; primary without Thought complete"},
        "proposal_mode":"matched_forced_not_model_generated","actual_external_tool_executions":0})


def verify_bindings(root):
    binding=read(root/"configs/mechanism_execution.json")
    for path,expected in binding["file_sha256"].items():
        if sha(root/path)!=expected:raise ValueError("Bound file changed: "+path)
    protocol=read(root/"configs/mechanism_protocol.json");schedule=read(root/"configs/mechanism_schedule.json")["blocks"]
    if len(schedule)!=36 or [b["block"] for b in schedule]!=list(range(36)):raise ValueError("Changed 36-block schedule")
    if semantic_sha(read(root/"configs/mechanism_model.json"))!=binding["model_configuration_sha256"]:raise ValueError("Model digest mismatch")
    from collections import Counter
    if Counter(b["kind"] for b in schedule)!={"free_generation":16,"matched_forced_pair":10,"archived_injecagent_replay":6,"archived_native_replay":4}:raise ValueError("Unexpected pilot scope")
    if set(b["sample_id"] for b in schedule if b["kind"]=="free_generation")!=set(protocol["free_generation_pilot_ids"]):raise ValueError("Free ID scope")
    if set(b["sample_id"] for b in schedule if b["kind"]=="matched_forced_pair")!=set(protocol["matched_forced_pilot_ids"]):raise ValueError("Forced ID scope")
    expected={("archived_injecagent_replay",r["sample_id"],None) for r in protocol["archived_injecagent_replay_controls"]}
    expected|={("archived_native_replay",r["sample_id"],r["step"]) for r in protocol["archived_native_replay_controls"]}
    if {(b["kind"],b["sample_id"],b.get("step")) for b in schedule if b["kind"].startswith("archived_")}!=expected:raise ValueError("Replay scope")
    if any(len(b["arm_order"])!=4 or set(b["arm_order"])!=set(ARMS) for b in schedule):raise ValueError("Arm order")
    if any(sorted(b["candidate_order"])!=[0,1] for b in schedule if b["kind"]=="matched_forced_pair"):raise ValueError("Candidate order")
    return binding,protocol,schedule


def worker(root,output,gpu,workers):
    binding,protocol,schedule=verify_bindings(root);output.mkdir(parents=True)
    config=read(root/"configs/mechanism_model.json");write_json(output/"environment.json",environment(config))
    generator=ImmutableGenerator(copy.deepcopy(config),gpu)
    scorer=V3Scorer(generator.tokenizer,TorchBackend(generator.model),max_input_tokens=config["max_input_tokens"])
    eos=generator.tokenizer.encode("<|im_end|>",add_special_tokens=False)
    if eos!=[151645]:raise ValueError("Actual tokenizer EOS mismatch")
    write_json(output/"model.json",{"configuration":config,"configuration_sha256":semantic_sha(config),
        "model_class":type(generator.model).__name__,"tokenizer_class":type(generator.tokenizer).__name__,
        "eos_ids":eos,"device_map":{str(k):str(v) for k,v in generator.model.hf_device_map.items()}})
    write_json(output/"warmup.json",owned_generate(generator,[{"role":"user","content":"Say ready."}],[],max_new_tokens=3))
    eos_preflight=scorer.score_eos({k:generator.tokenizer.apply_chat_template([{"role":"user","content":"Reply NO_ACTION."}],tokenize=False,add_generation_prompt=True) for k in VIEW_NAMES},BRANCH_PREFIXES["NO_ACTION"])
    write_json(output/"eos_preflight.json",eos_preflight)
    static={r["sample_id"]:r for r in read_lines(root/"data/mechanism/static_inputs.jsonl")}
    replays={r["replay_key"]:r for r in read_lines(root/"data/mechanism/replays.jsonl")}
    forced=read(root/"data/mechanism/forced_payloads.json");labels=read(root/"data/mechanism/evaluation_only.json")
    roster={r["sample_id"]:r for r in binding["free_roster"]};native=None;errors=0;count=0
    with (output/"records.jsonl").open("w",encoding="utf-8") as stream:
        for block in schedule[gpu::workers]:
            sample=block["sample_id"];kind=block["kind"]
            try:
                if kind=="free_generation":
                    selection=roster[sample]
                    if selection["source"]=="agentdojo":
                        if native is None:
                            from .run_agentdojo_bicp import open_runtime
                            native=open_runtime(root)
                        record=collect_native(selection,native,generator,scorer,block["arm_order"])
                    else:
                        item=static[sample];event=collect_event(item["views"]["original"],item["views"],item["tool_names"],generator,scorer,block["arm_order"],item["mask_provenance"])
                        event["step"]=0;record={**selection,"events":[event]}
                        case=labels["cases"][sample]
                        if selection["source"]=="conditional":
                            from .conditional_tasks import classify_call,simulate_case
                            record.update(authorization_label=classify_call(case,event["parsed_call"]),no_defense=simulate_case(case,event["parsed_call"]))
                        else:
                            record["authorization_label"]=injecagent_call_label(case,event["parsed_call"],labels["injec_tools"])
                            record["complete_attack_asr_available"]=False
                            from .injecagent import load_module
                            official=load_module(root/"data/upstream/InjecAgent/src/output_parsing.py","v3_official")
                            try:record["official_stage1_only"]=official.evaluate_output_prompted(event["generation"]["output"],case["Attacker Tools"][0],case["Attacker Tools"],case["User Tool"],case["Attacker Instruction"])
                            except Exception as error:record["official_stage1_only"]={"eval":None,"error":str(error)}
                elif kind=="matched_forced_pair":
                    record=score_forced(static[sample],forced[sample],scorer,generator.tokenizer,block)
                    record["evaluation_only_candidate_labels"]=labels["forced_labels"][sample]
                else:
                    key=kind+":"+sample+(":"+str(block["step"]) if "step" in block else "")
                    item=replays[key]
                    diagnostic=score_proposal(item["views"],item["historical_output"],item["parsed_call"],generator.tokenizer,scorer,block["arm_order"],branches=False)
                    record={"sample_id":sample,"source":item["source"],"diagnostic":diagnostic,"historical_provenance":item["historical_provenance"],
                        "historical_output":item["historical_output"],"parsed_call":item["parsed_call"],"new_generation_performed":False,
                        "authorization_label":labels["replay_labels"][key]}
            except Exception as error:
                errors+=1;record={"sample_id":sample,**error_record(error,generator),"traceback":traceback.format_exc()}
            record.update(kind=kind,schedule_block=block["block"],worker_gpu=gpu,arm_order=copy.deepcopy(block["arm_order"]),
                binding_sha256=sha(root/"configs/mechanism_execution.json"),actual_model_measurement=True,actual_external_tool_executions=0)
            write_record(stream,copy.deepcopy(record));count+=1
            write_json(output/"progress.json",{"recorded":count,"planned":len(schedule[gpu::workers]),"last_block":block["block"],"record_errors":errors})
            print(json.dumps({"gpu":gpu,"block":block["block"],"kind":kind,"record_error":record.get("error")}),flush=True)
    write_json(output/"completion.json",{"recorded":count,"record_errors":errors,"collection_complete":count==len(schedule[gpu::workers])})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=2,choices=(1,2));parser.add_argument("--gpu",type=int)
    args=parser.parse_args();root=args.root.resolve();output=args.output.resolve()
    if args.gpu is not None:worker(root,output,args.gpu,args.workers);return
    binding,protocol,schedule=verify_bindings(root)
    if output.exists():raise ValueError("Preserve existing run")
    output.mkdir(parents=True)
    for name in ("mechanism_execution.json","mechanism_model.json","mechanism_protocol.json","mechanism_schedule.json"):
        (output/name).write_bytes((root/"configs"/name).read_bytes())
    from huggingface_hub import snapshot_download
    config=read(root/"configs/mechanism_model.json")
    snapshot_download(repo_id=config["model_id"],revision=config["model_revision"],allow_patterns=["*.json","*.safetensors","*.txt","*.model"])
    jobs=[subprocess.Popen([sys.executable,"-m","src.run_mechanism_v3","--root",str(root),"--output",str(output/"workers"/f"worker_{i}"),"--workers",str(args.workers),"--gpu",str(i)],cwd=root) for i in range(args.workers)]
    exits=[job.wait() for job in jobs];records=[]
    for path in sorted((output/"workers").glob("worker_*/records.jsonl")):records.extend(read_lines(path))
    records.sort(key=lambda r:r["schedule_block"])
    with (output/"records.jsonl").open("w",encoding="utf-8") as stream:
        for record in records:write_record(stream,record)
    complete=[r["schedule_block"] for r in records]==list(range(36))
    write_json(output/"completion.json",{"collection_complete":complete,"planned_blocks":36,"recorded_blocks":len(records),
        "record_errors":sum("error" in r for r in records),"worker_exit_codes":exits,"actual_model_measurement":True,
        "independently_accepted":False,"thresholds_selected":False,"actual_external_tool_executions":0})
    if not complete or any(exits):raise RuntimeError("Incomplete pilot; preserve partial blocks")


if __name__=="__main__":main()
