"""Immutable, development-only mechanism collection; never fits a gate.

84 static continuations and four bounded native tasks share one original
proposal across four score cells. The shadow is recorded/replayed, not used to
change that proposal's history. Each GPU is an isolated model process.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import random
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .mechanism_probe import ARMS, SequenceScorer, TorchBackend, batch_consistency, score_diagnostic, semantic_sha
from .parsing import parse_call
from .run_baseline import Generator, environment, write_json
from .run_bicp import USER_MASK, OBS_MASK, counterfactual_messages


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owned_generate(generator, messages, tools, **kwargs):
    """Also protects runner logs when an injected test/provider returns aliases."""
    snapshot = copy.deepcopy(messages)
    generated = copy.deepcopy(generator.generate(copy.deepcopy(snapshot), copy.deepcopy(list(tools)), **kwargs))
    if generated.get("messages") != snapshot:
        raise ValueError("Generator returned a history different from the requested snapshot")
    generated["messages"] = copy.deepcopy(snapshot)
    return generated


class ImmutableGenerator(Generator):
    """Pinned v1 loading/greedy hooks, with owned logs and explicit timing blocks."""
    def generate(self, messages, candidates, max_new_tokens=None):
        from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
        torch, tokenizer = self.torch, self.tokenizer
        snapshot = copy.deepcopy(messages)
        candidates = copy.deepcopy(list(candidates))
        backend = TorchBackend(self.model)
        memory = backend.memory_start()
        started = time.perf_counter()
        prompt = tokenizer.apply_chat_template(snapshot, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        n_prompt = inputs.input_ids.shape[1]
        if n_prompt > self.config["max_input_tokens"]:
            raise ValueError(f"context_overflow:{n_prompt}; no silent truncation permitted")
        diagnostics = []

        class ToolLogits(LogitsProcessor):
            def __call__(self, input_ids, scores):
                tail = tokenizer.decode(input_ids[0, max(n_prompt, input_ids.shape[1]-48):], skip_special_tokens=True)
                if len(diagnostics) < 4 and re.search(r"(?:^|\n)Action:[ \t]*$", tail):
                    forms = {tool: {"bare": tokenizer.encode(tool, add_special_tokens=False),
                        "space": tokenizer.encode(" " + tool, add_special_tokens=False)} for tool in candidates}
                    ids = sorted({seq[0] for form in forms.values() for seq in form.values() if seq})
                    values = scores[0, ids].float().detach().cpu().tolist()
                    normalizer = torch.logsumexp(scores[0].float(), dim=-1).item()
                    diagnostics.append({"generated_token_position": input_ids.shape[1]-n_prompt,
                        "prefix_tail": tail, "tool_token_ids": forms, "first_token_logits": dict(zip(map(str, ids), values)),
                        "vocabulary_log_normalizer": normalizer, "interpretation": "diagnostic only, never reused for scoring"})
                return scores

        class StopObservation(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                text = tokenizer.decode(input_ids[0, n_prompt:], skip_special_tokens=True)
                return "\nObservation:" in text or "\nObservation :" in text

        backend.synchronize()
        prepared = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False,
                max_new_tokens=max_new_tokens or self.config["max_new_tokens"],
                pad_token_id=tokenizer.eos_token_id, logits_processor=LogitsProcessorList([ToolLogits()]),
                stopping_criteria=StoppingCriteriaList([StopObservation()]))
        backend.synchronize()
        decoded_started = time.perf_counter()
        new_tokens = output[0, n_prompt:].detach().cpu().tolist()
        input_ids = inputs.input_ids[0].detach().cpu().tolist()
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = re.split(r"\nObservation\s*:", raw, 1)[0].strip()
        finished = time.perf_counter()
        result = {"messages": copy.deepcopy(snapshot), "rendered_prompt": prompt, "input_token_ids": input_ids,
            "input_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "output": text, "raw_output": raw,
            "generated_token_ids": new_tokens, "input_tokens": n_prompt, "output_tokens": len(new_tokens),
            "generation_seconds": decoded_started-prepared,
            "reached_token_limit": len(new_tokens) >= (max_new_tokens or self.config["max_new_tokens"]),
            "first_token_diagnostics": diagnostics, "truncated": False,
            "timing": {"preparation_ms": (prepared-started)*1000,
                "forward_decode_ms": (decoded_started-prepared)*1000,
                "postprocessing_ms": (finished-decoded_started)*1000, "total_ms": (finished-started)*1000,
                "memory": backend.memory_end(memory),
                "scope": "full prompt preparation, synchronized generate (prefill plus decode), output decoding; no cross-call KV reuse"}}
        del output, inputs
        return copy.deepcopy(result)


def error_record(error, generator=None):
    value = {"error": f"{type(error).__name__}:{error}", "is_oom": "out of memory" in str(error).lower()}
    if generator is not None and hasattr(generator, "torch"):
        torch = generator.torch
        if torch.cuda.is_available():
            value["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(generator.device)
            value["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(generator.device)
            gc.collect()
            torch.cuda.empty_cache()
    return value


def collect_event(messages, views, tools, generator, scorer, arm_order, *, mask_info=None):
    before = copy.deepcopy(messages)
    generated = owned_generate(generator, before, tools)
    parsed = parse_call(generated["output"], list(tools))
    # This function accepts model inputs only. Authorization and task utility
    # are evaluated by separate code after the scores have been collected.
    diagnostic = score_diagnostic(copy.deepcopy(views), generated["output"], parsed,
                                  generator.tokenizer, scorer, arm_order)
    event = {"messages_before": before, "generation": generated, "parsed_call": parsed,
        "diagnostic": diagnostic, "mask_provenance": copy.deepcopy(mask_info), "executed": False,
        "actual_external_tool_executions": 0, "oracle_passed_to_model_or_scorer": False}
    observation_available = mask_info is None or mask_info.get("observation_intervention_available", True)
    if parsed["kind"] == "tool_call" and observation_available:
        try:
            shadow = owned_generate(generator, views["user_masked"], tools)
            shadow_parsed = parse_call(shadow["output"], list(tools))
            same = shadow_parsed.get("kind") == "tool_call" and shadow_parsed.get("tool") == parsed["tool"] and shadow_parsed.get("arguments") == parsed["arguments"]
            event["shadow"] = {"generation": shadow, "parsed_call": shadow_parsed, "blocked": same,
                "comparison": "exact_function_and_arguments", "view": "user_masked", "diagnostic_only": True}
        except Exception as error:
            event["shadow"] = {**error_record(error, generator), "blocked": True,
                "reason": "generation_failure_safe_stop", "diagnostic_only": True}
    else:
        event["shadow"] = {"blocked": False, "reason": "no_valid_proposal_or_no_external_observation"}
    if generated["messages"] != before or diagnostic["messages_by_view"] != views:
        raise ValueError("History changed during collection")
    return copy.deepcopy(event)


def measure_performance(event, tools, generator, scorer, settings, *, sample_id):
    """The 12 preselected scenarios; the first step is never replaced on failure."""
    diagnostic = event["diagnostic"]
    arm = settings["performance_arm"]
    base = {"sample_id": sample_id, "step": event.get("step", 0), "arm": arm,
        "no_outcome_based_reselection": True, "repetitions": [], "batch_consistency": []}
    if not diagnostic.get("proposal") or not diagnostic["arms"][arm].get("available"):
        return {**base, "available": False, "reason": "first_step_has_no_scoreable_raw_json_proposal"}
    proposal = diagnostic["proposal"]
    contexts = {name: text + proposal["thought"] + "Action:" for name, text in diagnostic["rendered_by_view"].items()}
    target = proposal["full_call_target"]
    views = diagnostic["messages_by_view"]
    base.update({"available": True, "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
        "context_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in contexts.items()},
        "generation_max_new_tokens": generator.config["max_new_tokens"],
        "timing_scope": "fixed original event/target, isolated repeated components; no early-stop trajectory saving"})
    # Seed varies by predeclared sample ID, never by date, timing or outcome.
    order = random.Random(f"{settings['timing_seed']}:{sample_id}")
    retained = {}
    for phase, count in (("warmup", settings["warmup_repetitions"]), ("measured", settings["timed_repetitions"])):
        for repetition in range(count):
            methods = ["probe_batch1", "probe_batch3", "user_masked_regeneration"]
            order.shuffle(methods)
            for position, method in enumerate(methods):
                row = {"phase": phase, "repetition": repetition, "method": method, "within_repetition_order": position}
                try:
                    if method == "user_masked_regeneration":
                        generated = owned_generate(generator, views["user_masked"], tools,
                            max_new_tokens=generator.config["max_new_tokens"])
                        row.update(generation=generated, cost=generated["timing"])
                    else:
                        score = scorer.score(contexts, target, batch_size=1 if method == "probe_batch1" else 3,
                                             segments=proposal["full_call_segments"])
                        row.update(score=score, cost=score["cost"])
                        if phase == "measured":
                            retained[(repetition, method)] = score
                    row["success"] = True
                except Exception as error:
                    row.update(success=False, **error_record(error, generator), fallback_batch_size=None)
                base["repetitions"].append(copy.deepcopy(row))
            if phase == "measured":
                left, right = retained.get((repetition, "probe_batch1")), retained.get((repetition, "probe_batch3"))
                if left and right:
                    base["batch_consistency"].append({"repetition": repetition,
                        **batch_consistency(left, right, absolute_token_tolerance=settings["absolute_token_tolerance"],
                            mean_tolerance=settings["mean_loglik_tolerance"])})
                else:
                    base["batch_consistency"].append({"repetition": repetition, "passed": None, "reason": "missing_paired_measurement"})
    return copy.deepcopy(base)


def collect_native(selection, suites, runtime_module, formatter, generator, scorer, arm_order, max_steps):
    from .run_agentdojo_bicp import initial_messages, new_environment, json_safe, native_utility, reference_call_label
    suite = suites[selection["suite"]]
    task = suite.user_tasks[selection["task_id"]]
    runtime = runtime_module.FunctionsRuntime(suite.tools)
    state, pre = new_environment(suite, task)
    messages = initial_messages(task, runtime)
    calls, events, final_text, terminal = [], [], "", "step_limit"
    # Reference calls are evaluated only after collection. Never pass them to
    # generator/scorer or use them to fix a failed action.
    for step in range(max_steps):
        before = copy.deepcopy(messages)
        state_before = json_safe(state)
        observations = [i for i, m in enumerate(before) if m["role"] == "tool"]
        if observations:
            views, mask = counterfactual_messages(before, user_index=1, obs_index=observations[-1])
            views = copy.deepcopy(views)
            mask["observation_intervention_available"] = True
        else:
            views = {k: copy.deepcopy(before) for k in ("original", "observation_masked", "user_masked")}
            views["user_masked"][1]["content"] = USER_MASK
            mask = {"observation_intervention_available": False,
                "reason": "initial_call_no_external_observation; observation view is explicit identity control"}
        try:
            event = collect_event(before, views, runtime.functions, generator, scorer, arm_order, mask_info=mask)
        except Exception as error:
            events.append({"step": step, "messages_before": before, "state_before": state_before,
                **error_record(error, generator), "executed": False, "actual_external_tool_executions": 0})
            terminal = "generation_or_collection_error"
            break
        event.update(step=step, state_before=state_before)
        events.append(event)
        parsed = event["parsed_call"]
        if parsed["kind"] == "final_answer":
            final_text, terminal = parsed["text"], "final_answer"
            messages.append({"role": "assistant", "content": event["generation"]["output"]})
            break
        if parsed["kind"] != "tool_call":
            terminal = "parse_failure"
            break
        try:
            call = runtime_module.FunctionCall(function=parsed["tool"], args=copy.deepcopy(parsed["arguments"]))
            returned, error = runtime.run_function(state, call.function, call.args)
            formatted = formatter(returned)
            calls.append(call)
            event.update(executed=True, tool_result=formatted, runtime_error=error, state_after=json_safe(state))
            messages.extend([{"role": "assistant", "content": event["generation"]["output"]},
                {"role": "tool", "name": call.function, "content": formatted if error is None else "Error: " + error}])
        except Exception as error:
            event.update(runtime_exception=f"{type(error).__name__}:{error}")
            terminal = "native_runtime_exception"
            break
    reference = task.ground_truth(pre.model_copy(deep=True))
    preceding = []
    for event in events:
        parsed = event.get("parsed_call", {})
        if parsed.get("kind") == "tool_call":
            event["authorization_label"] = reference_call_label(parsed, reference, preceding, runtime)
            if event.get("executed"):
                preceding.append(runtime_module.FunctionCall(function=parsed["tool"], args=copy.deepcopy(parsed["arguments"])))
    try:
        utility = native_utility(suite, task, pre, state, calls, final_text)
        utility_error = None
    except Exception as error:
        utility, utility_error = None, f"{type(error).__name__}:{error}"
    # Separately replay the same-prefix shadow stop control in a fresh state.
    shadow_state, shadow_pre = new_environment(suite, task)
    shadow_calls, blocked, shadow_text = [], False, ""
    for event in events:
        if event.get("shadow", {}).get("blocked"):
            blocked = True
            break
        parsed = event.get("parsed_call", {})
        if parsed.get("kind") == "final_answer":
            shadow_text = parsed["text"]
            break
        if parsed.get("kind") == "tool_call" and event.get("executed"):
            call = runtime_module.FunctionCall(function=parsed["tool"], args=copy.deepcopy(parsed["arguments"]))
            runtime.run_function(shadow_state, call.function, call.args)
            shadow_calls.append(call)
    try:
        shadow_utility = native_utility(suite, task, shadow_pre, shadow_state, shadow_calls, shadow_text)
        shadow_error = None
    except Exception as error:
        shadow_utility, shadow_error = None, f"{type(error).__name__}:{error}"
    result = {**selection, "events": events, "messages": messages, "initial_state": json_safe(pre),
        "final_state": json_safe(state), "executed_calls": json_safe(calls), "reference_calls_evaluation_only": json_safe(reference),
        "terminal": terminal, "final_text": final_text, "native_utility": utility, "utility_error": utility_error,
        "shadow_control": {"blocked": blocked, "native_utility": shadow_utility, "error": shadow_error,
            "executed_calls": json_safe(shadow_calls), "final_state": json_safe(shadow_state)},
        "actual_external_tool_executions": 0, "actual_model_measurement": True}
    for event in events:
        if event.get("generation") and event["generation"]["messages"] != event["messages_before"]:
            raise ValueError("Mutable generation history detected")
    return copy.deepcopy(result)


def verify_bindings(root):
    binding = read_json(root / "configs/mechanism_execution.json")
    for name, expected in binding["file_sha256"].items():
        if sha_file(root / name) != expected:
            raise ValueError("Execution input/source mismatch: " + name)
    protocol = read_json(root / "configs/mechanism_protocol.json")
    schedule = read_json(root / "configs/mechanism_schedule.json")
    if len(protocol["roster"]) != 88 or any(r["split"] != "development" for r in protocol["roster"]):
        raise ValueError("Only the approved 88 development scenarios may run")
    if [r["block"] for r in schedule] != list(range(88)) or {r["sample_id"] for r in schedule} != {r["sample_id"] for r in protocol["roster"]}:
        raise ValueError("Missing/changed schedule IDs")
    if semantic_sha(read_json(root / "configs/mechanism_model.json")) != binding["model_configuration_sha256"]:
        raise ValueError("New model configuration not bound")
    return binding, protocol, schedule


def write_record(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def worker(root, output, gpu, workers):
    binding, protocol, schedule = verify_bindings(root)
    output.mkdir(parents=True)
    config = read_json(root / "configs/mechanism_model.json")
    write_json(output / "environment.json", environment(copy.deepcopy(config)))
    generator = ImmutableGenerator(copy.deepcopy(config), gpu)
    scorer = SequenceScorer(generator.tokenizer, TorchBackend(generator.model), max_input_tokens=config["max_input_tokens"])
    write_json(output / "model.json", {"configuration": config, "configuration_sha256": semantic_sha(config),
        "device_map": {str(k): str(v) for k, v in generator.model.hf_device_map.items()},
        "model_class": type(generator.model).__name__, "tokenizer_class": type(generator.tokenizer).__name__,
        "actual_model_measurement": True})
    write_json(output / "warmup.json", owned_generate(generator, [{"role": "user", "content": "Say ready."}], [], max_new_tokens=3))
    static = {r["sample_id"]: r for r in (json.loads(line) for line in (root / "data/mechanism/static_inputs.jsonl").read_text(encoding="utf-8").splitlines())}
    labels = read_json(root / "data/mechanism/evaluation_only.json")
    roster = {r["sample_id"]: r for r in protocol["roster"]}
    native = None
    selected = schedule[gpu::workers]
    records, perf_pending = [], []
    with (output / "records.jsonl").open("w", encoding="utf-8") as stream:
        for index, item in enumerate(selected):
            sample_id = item["sample_id"]
            selection = roster[sample_id]
            try:
                if selection["source"] == "agentdojo":
                    if native is None:
                        from .run_agentdojo_bicp import open_runtime
                        native = open_runtime(root)
                    record = collect_native(selection, *native, generator, scorer, item["arm_order"], protocol["max_steps_agentdojo"])
                else:
                    inputs = static[sample_id]
                    event = collect_event(inputs["views"]["original"], inputs["views"], inputs["tool_names"],
                        generator, scorer, item["arm_order"], mask_info=inputs["mask_provenance"])
                    event["step"] = 0
                    record = {**selection, "events": [event], "actual_external_tool_executions": 0, "actual_model_measurement": True}
                    # Model/scoring collection above has no label object argument.
                    if selection["source"] == "conditional":
                        from .conditional_tasks import classify_call, simulate_case
                        case = labels[sample_id]["case"]
                        record["authorization_label"] = classify_call(case, event["parsed_call"])
                        record["no_defense"] = simulate_case(case, event["parsed_call"])
                        record["shadow_control"] = simulate_case(case, event["parsed_call"], blocked=event["shadow"]["blocked"])
                    else:
                        from .injecagent import load_module
                        official = load_module(root / "data/upstream/InjecAgent/src/output_parsing.py", "diagnostic_official")
                        case = labels[sample_id]["case"]
                        try:
                            record["official_stage1_only"] = official.evaluate_output_prompted(event["generation"]["output"],
                                case["Attacker Tools"][0], case["Attacker Tools"], case["User Tool"], case["Attacker Instruction"])
                        except Exception as error:
                            # An evaluator error cannot erase the already
                            # collected generation or any of its score cells.
                            record["official_stage1_only"] = {"eval": None,
                                "error": f"{type(error).__name__}:{error}", "unresolved": True}
                        record["authorization_label"] = {"legal_call": False if event["parsed_call"].get("tool") in case["Attacker Tools"] else None,
                            "scope": "attacker tool negatives only; no fabricated benign positives"}
                        record["complete_attack_asr_available"] = False
                record.update(schedule_block=item["block"], arm_order=copy.deepcopy(item["arm_order"]), worker_gpu=gpu,
                    binding_sha256=sha_file(root / "configs/mechanism_execution.json"))
                if sample_id in binding["performance_sample_ids"]:
                    event = record.get("events", [{}])[0]
                    if event.get("diagnostic"):
                        perf_pending.append((sample_id, copy.deepcopy(event), static[sample_id]["tool_names"]))
                    else:
                        perf_pending.append((sample_id, None, []))
            except Exception as error:
                record = {**selection, **error_record(error, generator), "traceback": traceback.format_exc(),
                    "schedule_block": item["block"], "worker_gpu": gpu, "actual_external_tool_executions": 0,
                    "actual_model_measurement": True}
                if sample_id in binding["performance_sample_ids"]:
                    perf_pending.append((sample_id, None, []))
            write_record(stream, record)
            records.append({"sample_id": sample_id, "source": selection["source"], "error": record.get("error")})
            write_json(output / "progress.json", {"completed": index+1, "planned": len(selected), "last_sample_id": sample_id,
                "performance_pending": len(perf_pending), "stage": "mechanism"})
            print(json.dumps({"gpu": gpu, "completed": index+1, "planned": len(selected), "sample_id": sample_id,
                "error": record.get("error")}), flush=True)
    # Main rows are durable before the separate, potentially expensive timing study.
    with (output / "performance.jsonl").open("w", encoding="utf-8") as stream:
        for index, (sample_id, event, tools) in enumerate(perf_pending):
            result = measure_performance(event, tools, generator, scorer, binding["performance"], sample_id=sample_id) if event else {
                "sample_id": sample_id, "available": False, "reason": "first_event_generation_or_collection_failed"}
            write_record(stream, result)
            write_json(output / "performance_progress.json", {"completed": index+1, "planned": len(perf_pending)})
            print(json.dumps({"gpu": gpu, "performance_completed": index+1, "planned": len(perf_pending), "sample_id": sample_id}), flush=True)
    write_json(output / "completion.json", {"planned": len(selected), "recorded": len(records), "collection_complete": len(records) == len(selected),
        "record_errors": sum(bool(r["error"]) for r in records), "performance_planned": len(perf_pending),
        "actual_model_measurement": True, "independently_accepted": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if args.gpu is not None:
        worker(root, output, args.gpu, args.workers)
        return
    binding, protocol, schedule = verify_bindings(root)
    if output.exists():
        raise ValueError("Refuse to overwrite existing run")
    output.mkdir(parents=True)
    for name in ("mechanism_execution.json", "mechanism_model.json", "mechanism_protocol.json", "mechanism_schedule.json"):
        (output / name).write_bytes((root / "configs" / name).read_bytes())
    from huggingface_hub import snapshot_download
    config = read_json(root / "configs/mechanism_model.json")
    snapshot_download(repo_id=config["model_id"], revision=config["model_revision"],
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    jobs = [subprocess.Popen([sys.executable, "-m", "src.run_mechanism_diagnostic", "--root", str(root),
        "--output", str(output / "workers" / f"worker_{i}"), "--workers", str(args.workers), "--gpu", str(i)], cwd=root)
        for i in range(args.workers)]
    exits = [job.wait() for job in jobs]
    ids, perf_ids = [], []
    with (output / "records.jsonl").open("w", encoding="utf-8") as merged:
        for p in sorted((output / "workers").glob("worker_*/records.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                ids.append(record["sample_id"])
                merged.write(line+"\n")
    with (output / "performance.jsonl").open("w", encoding="utf-8") as merged:
        for p in sorted((output / "workers").glob("worker_*/performance.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    perf_ids.append(json.loads(line)["sample_id"])
                    merged.write(line+"\n")
    complete = len(ids) == len(set(ids)) == 88 and set(ids) == {r["sample_id"] for r in protocol["roster"]}
    perf_complete = len(perf_ids) == len(set(perf_ids)) == 12 and set(perf_ids) == set(binding["performance_sample_ids"])
    write_json(output / "completion.json", {"collection_complete": complete, "performance_collection_complete": perf_complete,
        "planned": 88, "recorded": len(ids), "performance_recorded": len(perf_ids), "worker_exit_codes": exits,
        "actual_model_measurement": True, "independently_accepted": False,
        "actual_external_tool_executions": 0, "thresholds_selected": False})
    if not complete or not perf_complete or any(exits):
        raise RuntimeError("Collection incomplete; preserve per-worker records and resume missing IDs only")


if __name__ == "__main__":
    main()
