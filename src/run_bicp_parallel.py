"""Shard the unchanged BiCP collector over two independent local GPU workers."""
import argparse
import copy
import json
import multiprocessing
from pathlib import Path
import time
import traceback

from . import run_bicp as runner
from .run_baseline import write_json


def shard_work(work, rank, workers):
    if workers not in (1, 2) or not 0 <= rank < workers:
        raise ValueError("Only one or two distinct GPU workers are supported")
    return work[rank::workers]


def merge_records(expected_ids, worker_rows, tag_key="runtime_worker"):
    if tag_key not in {"runtime_worker", "collection_chunk"}:
        raise ValueError("Unknown collection metadata field")
    index = {}
    for rank, rows in enumerate(worker_rows):
        for row in rows:
            identity = row["sample_id"]
            if identity not in expected_ids or identity in index:
                raise ValueError("Unexpected or duplicate worker sample ID")
            if tag_key in row and row[tag_key] != rank:
                raise ValueError("Collection merge would overwrite original runtime metadata")
            index[identity] = dict(row, **{tag_key: rank})
    missing = [identity for identity in expected_ids if identity not in index]
    return [index[identity] for identity in expected_ids if identity in index], missing


def check_resume_plan(plan, full_work, protocol):
    full_ids = [case.get("case_id", case.get("sample_id")) for _, case in full_work]
    if plan["expected_sample_ids"] != full_ids or plan["scoring_signature"] != runner.scoring_signature(protocol):
        raise ValueError("Resume plan has changed data selection or scoring provenance")
    completed, missing = plan["completed_sample_ids"], plan["missing_sample_ids"]
    if (len(completed) != len(set(completed)) or len(missing) != len(set(missing))
            or set(completed) & set(missing) or set(completed) | set(missing) != set(full_ids)):
        raise ValueError("Resume IDs do not form an exact disjoint partition")
    return [pair for pair in full_work if pair[1].get("case_id", pair[1].get("sample_id")) in set(missing)]


def worker(args_dict, protocol, rank, workers):
    args = argparse.Namespace(**args_dict)
    args.output = args.output / "workers" / f"worker_{rank}"
    args.output.mkdir(parents=True, exist_ok=True)
    args.gpu += rank
    original_load = runner.load_work
    def load_shard(*positional, **keywords):
        work = original_load(*positional, **keywords)
        if args.resume_plan:
            work = check_resume_plan(json.loads(args.resume_plan.read_text(encoding="utf-8")), work, protocol)
        return shard_work(work, rank, workers)
    runner.load_work = load_shard
    write_json(args.output/"protocol.json", protocol)
    try:
        runner.collect(args, protocol)
    except Exception as error:
        write_json(args.output/"worker_error.json", {"error": f"{type(error).__name__}:{error}", "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--phase", choices=["collect"], default="collect")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--splits", default="development")
    parser.add_argument("--sources", default="injecagent,conditional")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", choices=["gpu"], default="gpu")
    parser.add_argument("--smoke-per-source", type=int, default=3)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--resume-plan", type=Path)
    args = parser.parse_args()
    if (args.output/"records.jsonl").exists() or (args.output/"workers").exists():
        raise RuntimeError("Use a fresh output directory; worker records are immutable")
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.root/"configs/baseline.json").read_text(encoding="utf-8"))
    protocol = runner.normalize_protocol(json.loads(args.protocol.read_text(encoding="utf-8")))
    protocol["model_configuration_sha256"] = runner.digest(config)
    protocol["implementation_sha256"] = runner.implementation_digest(args.root)
    frozen = None
    if args.calibration:
        frozen = json.loads(args.calibration.read_text(encoding="utf-8"))
        runner.validate_calibration(frozen, protocol)
    if "test" in args.splits.split(",") and frozen is None:
        raise ValueError("Final test collection requires pre-frozen --calibration")
    work = runner.load_work(args.root, args.mode, args.splits.split(","), args.smoke_per_source,
                            dict(protocol, seed=config["seed"]), args.sources.split(","))
    full_planned = len(work)
    if args.resume_plan:
        work = check_resume_plan(json.loads(args.resume_plan.read_text(encoding="utf-8")), work, protocol)
    expected_ids = [case.get("case_id", case.get("sample_id")) for _, case in work]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Duplicate planned sample IDs")
    if len(work) < args.workers:
        raise ValueError("Not enough records for requested workers")
    write_json(args.output/"protocol.json", protocol)
    write_json(args.output/"selection.json", {"splits": args.splits.split(","), "mode": args.mode,
        "smoke_is_not_final_evaluation": args.mode == "smoke", "planned_records": len(work), "sample_ids": expected_ids,
        "test_threshold_frozen_before_collection": frozen is not None,
        "calibration_file_sha256": __import__("hashlib").sha256(args.calibration.read_bytes()).hexdigest() if args.calibration else None,
        "workers": args.workers, "sharding": "fixed ordered sample index modulo worker count; group splits preserved"})
    if args.resume_plan:
        selected = json.loads((args.output/"selection.json").read_text(encoding="utf-8"))
        selected.update(full_planned_records=full_planned, resume_plan_sha256=__import__("hashlib").sha256(args.resume_plan.read_bytes()).hexdigest(),
                        resume_only_missing_records=True)
        write_json(args.output/"selection.json", selected)
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=worker, args=(vars(args), protocol, rank, args.workers)) for rank in range(args.workers)]
    for process in processes:
        process.start()
    latest_count = -1
    while any(process.is_alive() for process in processes):
        progress = []
        for rank in range(args.workers):
            path = args.output/"workers"/f"worker_{rank}"/"progress.json"
            try:
                progress.append(json.loads(path.read_text(encoding="utf-8")))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        count = sum(item["completed"] for item in progress)
        if count != latest_count:
            latest_count = count
            write_json(args.output/"progress.json", {"completed": count, "planned": len(work),
                "errors": sum(item["errors"] for item in progress), "workers": args.workers})
            print(f"BiCP parallel: {count}/{len(work)}", flush=True)
        time.sleep(2)
    for process in processes:
        process.join()
    folders = [args.output/"workers"/f"worker_{rank}" for rank in range(args.workers)]
    worker_rows = [runner.read_jsonl(folder/"records.jsonl") if (folder/"records.jsonl").exists() else [] for folder in folders]
    records, missing = merge_records(expected_ids, worker_rows)
    (args.output/"records.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in records), encoding="utf-8")
    completed = not missing and not any(row.get("error") for row in records) and all(p.exitcode == 0 for p in processes)
    worker_statuses = [json.loads((folder/"run_status.json").read_text()) if (folder/"run_status.json").exists() else None for folder in folders]
    write_json(args.output/"run_status.json", {"collection_complete": completed, "full_evaluation_complete": False,
        "mode": args.mode, "records": len(records), "errors": sum(bool(r.get("error")) for r in records)+len(missing),
        "missing_ids": missing, "workers": args.workers, "worker_exit_codes": [p.exitcode for p in processes],
        "wall_seconds": time.perf_counter()-started, "worker_statuses": worker_statuses})
    if completed:
        models = [json.loads((folder/"model.json").read_text()) for folder in folders]
        model = copy.deepcopy(models[0])
        model["worker_models"] = models
        write_json(args.output/"model.json", model)
        environment = json.loads((folders[0]/"environment.json").read_text())
        environment["parallel_workers"] = args.workers
        write_json(args.output/"environment.json", environment)
        if frozen:
            from . import conditional_tasks
            write_json(args.output/"frozen_calibration.json", frozen)
            runner.write_applied(args.output, [runner.apply_record(row, frozen, protocol, conditional_tasks) for row in records])
    if not completed:
        raise RuntimeError("Parallel collection incomplete; all partial and error records retained")


if __name__ == "__main__":
    main()
