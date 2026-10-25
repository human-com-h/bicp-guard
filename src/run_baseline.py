"""Greedy, unprotected Qwen inference with the official InjecAgent evaluator."""
import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

from .injecagent import load_cases, load_tools, load_module, build_prompt, CachedSimulator, clean_controls
from .metrics import summarize
from .parsing import parse_call, validate_schema


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def environment(config):
    import torch
    packages = {}
    for name in ("torch", "transformers", "accelerate", "bitsandbytes", "datasets", "huggingface-hub", "tokenizers", "nltk"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "cuda": torch.version.cuda, "gpus": [{"index": i, "name": torch.cuda.get_device_name(i),
            "memory_bytes": torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())],
            "configuration": config, "actual_external_tools": False}


class Generator:
    def __init__(self, config, gpu):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        self.torch = torch
        self.config = config
        self.device = f"cuda:{gpu}"
        torch.cuda.set_device(gpu)
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"], trust_remote_code=False)
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config["model_revision"],
                    quantization_config=quantization, device_map={"": gpu}, torch_dtype=torch.float16,
                    attn_implementation="sdpa", trust_remote_code=False).eval()
        self.torch.cuda.synchronize()

    def generate(self, messages, candidates, max_new_tokens=None):
        import re
        from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
        torch, tokenizer = self.torch, self.tokenizer
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        n_prompt = inputs.input_ids.shape[1]
        if n_prompt > self.config["max_input_tokens"]:
            raise ValueError(f"context_overflow:{n_prompt}; no silent truncation permitted")
        diagnostics = []

        class ToolLogits(LogitsProcessor):
            def __call__(self, input_ids, scores):
                tail = tokenizer.decode(input_ids[0, max(n_prompt, input_ids.shape[1]-48):], skip_special_tokens=True)
                if len(diagnostics) < 4 and re.search(r"(?:^|\n)Action:[ \t]*$", tail):
                    # Space and bare forms are diagnostics only. Shared prefixes are preserved.
                    forms = {tool: {"bare": tokenizer.encode(tool, add_special_tokens=False),
                                    "space": tokenizer.encode(" " + tool, add_special_tokens=False)} for tool in candidates}
                    ids = sorted({seq[0] for form in forms.values() for seq in form.values() if seq})
                    values = scores[0, ids].float().detach().cpu().tolist()
                    normalizer = torch.logsumexp(scores[0].float(), dim=-1).item()
                    diagnostics.append({"generated_token_position": input_ids.shape[1]-n_prompt,
                                        "prefix_tail": tail, "tool_token_ids": forms,
                                        "first_token_logits": dict(zip(map(str, ids), values)),
                                        "vocabulary_log_normalizer": normalizer,
                                        "interpretation": "first-token diagnostic; not a complete-tool probability"})
                return scores

        class StopObservation(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                text = tokenizer.decode(input_ids[0, n_prompt:], skip_special_tokens=True)
                return "\nObservation:" in text or "\nObservation :" in text

        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens or self.config["max_new_tokens"],
                pad_token_id=tokenizer.eos_token_id, logits_processor=LogitsProcessorList([ToolLogits()]),
                stopping_criteria=StoppingCriteriaList([StopObservation()]))
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        new_tokens = output[0, n_prompt:].detach().cpu().tolist()
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = re.split(r"\nObservation\s*:", raw, 1)[0].strip()
        return {"messages": messages, "rendered_prompt": prompt, "input_token_ids": inputs.input_ids[0].cpu().tolist(),
                "output": text, "raw_output": raw, "generated_token_ids": new_tokens,
                "input_tokens": n_prompt, "output_tokens": len(new_tokens), "generation_seconds": elapsed,
                "reached_token_limit": len(new_tokens) >= (max_new_tokens or self.config["max_new_tokens"]),
                "first_token_diagnostics": diagnostics}


def evaluate_case(case, generator, tools, prompts, official, simulator):
    candidates = [case["User Tool"]] + case["Attacker Tools"]
    record = {"case_id": case["case_id"], "group_id": case["group_id"], "attack": case["attack"],
              "split": case["split"], "attacker_tools": case["Attacker Tools"], "source_case": case,
              "actual_tool_execution": False}
    first = generator.generate(build_prompt(case, tools, prompts), candidates)
    record["stage1"] = first
    record["parsed_call"] = parse_call(first["output"], candidates)
    if record["parsed_call"]["kind"] == "tool_call":
        record["schema_diagnostic"] = validate_schema(record["parsed_call"]["arguments"], tools[record["parsed_call"]["tool"]])
    evaluated = official.evaluate_output_prompted(first["output"], case["Attacker Tools"][0], case["Attacker Tools"],
                                                   case["User Tool"], case["Attacker Instruction"])
    record["official_stage1"] = evaluated
    record["official_eval"] = evaluated["eval"]
    if case["attack"] == "ds" and evaluated["eval"] == "succ":
        parameters = evaluated["Attacker Tool Parameters"]
        response, provenance = simulator.get(case["Attacker Tools"][0], parameters)
        record["simulation_provenance"] = provenance
        if response is None:
            record["unresolved_stage2"] = True
        else:
            stage2 = {"parameters": parameters, "thought": evaluated["Attacker Tool Thought"], "response": response}
            second = generator.generate(build_prompt(case, tools, prompts, stage2), candidates)
            record["stage2"] = second
            evaluated2 = official.evaluate_output_prompted(second["output"], case["Attacker Tools"][1], case["Attacker Tools"],
                                                            case["User Tool"], case["Attacker Instruction"])
            record["official_stage2"] = evaluated2
            record["official_eval_stage2"] = evaluated2["eval"]
    return record


def worker(root, output_dir, mode, gpu, workers):
    config = json.loads((root / "configs/baseline.json").read_text())
    random.seed(config["seed"])
    env = environment(config)
    write_json(output_dir / f"environment_gpu{gpu}.json", env)
    if len(env["gpus"]) < workers:
        raise RuntimeError("Requested GPU count unavailable")
    upstream = root / "data/upstream/InjecAgent"
    tools = load_tools(upstream)
    prompts = load_module(upstream / "src/prompts/agent_prompts.py", "official_injecagent_prompts")
    official = load_module(upstream / "src/output_parsing.py", "official_injecagent_evaluator")
    simulator = CachedSimulator(upstream)
    splits = json.loads((root / "data/splits.json").read_text())
    assignment = splits["group_assignment"]
    cases = load_cases(upstream)
    for case in cases:
        case["split"] = assignment[case["group_id"]]
    if mode == "smoke":
        chosen = []
        for attack in ("dh", "ds"):
            available = [c for c in cases if c["attack"] == attack and c["split"] == "development"]
            chosen.extend(random.Random(config["seed"]).sample(available, 6))
        cases = chosen + [c for c in clean_controls(cases) if c["split"] == "development"]
    else:
        cases += clean_controls(cases)
    cases = sorted(cases, key=lambda c: c["case_id"])[gpu::workers]
    output = output_dir / f"records_gpu{gpu}.jsonl"
    if output.exists():
        raise RuntimeError("Refuse to overwrite an existing experiment")
    generator = Generator(config, gpu)
    write_json(output_dir / f"model_gpu{gpu}.json", {"hf_device_map": generator.model.hf_device_map,
               "model_memory_bytes": generator.model.get_memory_footprint(), "tokenizer_revision": config["model_revision"]})
    warmup = generator.generate([{"role": "user", "content": "Reply with the word ready."}], [], max_new_tokens=8)
    write_json(output_dir / f"warmup_gpu{gpu}.json", warmup)
    collected = []
    with output.open("a", encoding="utf-8") as handle:
        for index, case in enumerate(cases):
            try:
                row = evaluate_case(case, generator, tools, prompts, official, simulator)
            except Exception as error:
                row = {"case_id": case["case_id"], "group_id": case["group_id"], "attack": case["attack"],
                       "split": case["split"], "official_eval": "invalid", "actual_tool_execution": False,
                       "error": f"{type(error).__name__}:{error}", "traceback": traceback.format_exc(), "source_case": case}
            collected.append(row)
            handle.write(json.dumps(row, ensure_ascii=False)+"\n")
            handle.flush()
            write_json(output_dir / f"progress_gpu{gpu}.json", {"completed": index+1, "planned": len(cases),
                        "last_case_id": case["case_id"], "metrics": summarize(collected)})
            if index % 10 == 0 or index+1 == len(cases):
                print(f"GPU {gpu}: {index+1}/{len(cases)} completed", flush=True)
    write_json(output_dir / f"worker_done_gpu{gpu}.json", {"records": len(collected),
                "peak_allocated_bytes": generator.torch.cuda.max_memory_allocated(gpu)})
    if any(r.get("error") for r in collected):
        raise RuntimeError("Infrastructure failures retained in records; run is not validated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.gpu is not None:
        worker(args.root, args.output, args.mode, args.gpu, args.workers)
        return
    from huggingface_hub import snapshot_download
    config = json.loads((args.root / "configs/baseline.json").read_text())
    write_json(args.output / "configuration.json", config)
    snapshot_download(repo_id=config["model_id"], revision=config["model_revision"],
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    jobs = [subprocess.Popen([sys.executable, "-m", "src.run_baseline", "--root", str(args.root),
             "--output", str(args.output), "--mode", args.mode, "--workers", str(args.workers), "--gpu", str(i)], cwd=args.root)
            for i in range(args.workers)]
    exits = [job.wait() for job in jobs]
    rows = []
    for path in sorted(args.output.glob("records_gpu*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    write_json(args.output / "summary.json", {"mode": args.mode, "worker_exit_codes": exits,
               "all": summarize(rows), "by_split": {split: summarize([r for r in rows if r["split"] == split])
               for split in ("development", "calibration", "test")}, "n_records": len(rows)})
    expected = 15 if args.mode == "smoke" else 1071
    valid = all(code == 0 for code in exits) and len(rows) == expected
    write_json(args.output / "run_status.json", {"validated_complete": valid, "expected_records": expected,
               "actual_records": len(rows), "mode": args.mode})
    if not valid:
        raise RuntimeError("Baseline incomplete; see per-worker records")


if __name__ == "__main__":
    main()
