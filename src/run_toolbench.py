"""Static benign next-action replay. This does not estimate task success."""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

from .injecagent import load_module
from .metrics import proportion
from .parsing import parse_call, validate_schema
from .run_baseline import Generator, environment, write_json
from .toolbench import build_replay_prompt


def worker(root, output, gpu, workers):
    config = json.loads((root / "configs/baseline.json").read_text())
    config["protocol_version"] = "stage1-toolbench-static-replay-v1"
    write_json(output / f"environment_gpu{gpu}.json", environment(config))
    prompts = load_module(root / "data/upstream/InjecAgent/src/prompts/agent_prompts.py", "official_prompts")
    samples = [json.loads(line) for line in (root / "data/toolbench_replay.jsonl").read_text().splitlines() if line.strip()][gpu::workers]
    generator = Generator(config, gpu)
    write_json(output / f"warmup_gpu{gpu}.json", generator.generate([{"role": "user", "content": "Reply with ready."}], [], max_new_tokens=8))
    path = output / f"records_gpu{gpu}.jsonl"
    if path.exists():
        raise RuntimeError("Refuse to overwrite an experiment")
    with path.open("a", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            row = {"sample_id": sample["sample_id"], "split": sample["split"], "source_sample": sample,
                   "actual_tool_execution": False, "reference_tool": sample["target_call"]["name"]}
            try:
                messages, tools = build_replay_prompt(sample, prompts)
                row["generation"] = generator.generate(messages, list(tools))
                row["parsed_call"] = parse_call(row["generation"]["output"], list(tools))
                row["next_tool_matches_reference"] = row["parsed_call"].get("tool") == row["reference_tool"]
                if row["parsed_call"]["kind"] == "tool_call":
                    row["schema_diagnostic"] = validate_schema(row["parsed_call"]["arguments"], tools[row["parsed_call"]["tool"]])
            except Exception as error:
                row["error"] = f"{type(error).__name__}:{error}"
                row["traceback"] = traceback.format_exc()
                row["next_tool_matches_reference"] = False
            handle.write(json.dumps(row, ensure_ascii=False)+"\n")
            handle.flush()
            write_json(output / f"progress_gpu{gpu}.json", {"completed": index+1, "planned": len(samples)})
            if index % 10 == 0 or index+1 == len(samples):
                print(f"GPU {gpu}: {index+1}/{len(samples)} benign replays completed", flush=True)
    write_json(output / f"worker_done_gpu{gpu}.json", {"records": len(samples), "peak_allocated_bytes": generator.torch.cuda.max_memory_allocated(gpu)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", default="toolbench")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.gpu is not None:
        worker(args.root, args.output, args.gpu, args.workers)
        return
    from huggingface_hub import snapshot_download
    config = json.loads((args.root / "configs/baseline.json").read_text())
    write_json(args.output / "configuration.json", dict(config, protocol_version="stage1-toolbench-static-replay-v1"))
    snapshot_download(repo_id=config["model_id"], revision=config["model_revision"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    jobs = [subprocess.Popen([sys.executable, "-m", "src.run_toolbench", "--root", str(args.root), "--output", str(args.output),
            "--workers", str(args.workers), "--gpu", str(i)], cwd=args.root) for i in range(args.workers)]
    exits = [job.wait() for job in jobs]
    rows = [json.loads(line) for path in args.output.glob("records_gpu*.jsonl") for line in path.read_text().splitlines() if line.strip()]
    errors = sum(bool(r.get("error")) for r in rows)
    summary = {"n": len(rows), "errors": errors, "next_action_agreement": proportion(sum(r["next_tool_matches_reference"] for r in rows), len(rows)),
               "task_success_rate": None, "actual_external_tool_executions": 0,
               "by_split": {split: proportion(sum(r["next_tool_matches_reference"] for r in rows if r["split"] == split), sum(r["split"] == split for r in rows))
                            for split in ("calibration", "test")},
               "interpretation": "Static next-tool agreement with recorded annotations; alternative correct actions may disagree. This is not end-to-end utility or defense FPR."}
    write_json(args.output / "summary.json", summary)
    complete = len(rows) == 500 and all(x == 0 for x in exits) and errors == 0
    write_json(args.output / "run_status.json", {"validated_complete": complete, "records": len(rows), "worker_exit_codes": exits, "errors": errors})
    if not complete:
        raise RuntimeError("Incomplete benign replay; errors are retained")


if __name__ == "__main__":
    main()
