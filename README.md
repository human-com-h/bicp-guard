# BiCP v2

**Reasoning-Prefix Deconfounding for Dual-Axis Counterfactual Probing and Operation-Risk Routing**

Research software by Zida Hao and Yubo Li, School of Software, Xinjiang University.

BiCP v2 screens one proposed tool call before simulated execution. It removes the current-step visible reasoning prefix, compares teacher-forced tool-name and complete-call likelihoods under observation/user ablation, and routes operations by risk. Scoped discovery requires zero additional model forwards. Protected calls use three serial forwards; mutations first receive deterministic evidence anchoring. The method is training-free.

This repository contains the final research implementation, fixed protocols, necessary inputs, and accepted evaluation artifacts. It contains **no manuscript or submission files**. No publication DOI or public preprint identifier has been assigned to this software record. [CITATION.cff](CITATION.cff) provides verified authorship and software citation metadata.

## Release and validation status

[v0.9.0-rc1](https://github.com/human-com-h/bicp-guard/releases/tag/v0.9.0-rc1) is a **prerelease**. The archived studies were completed on two Tesla T4 GPUs. Release preparation verifies source identity, CPU functionality, recorded-run evaluation, table reconstruction, and local CPU/CUDA scoring arithmetic. It does not repeat full Qwen-7B GPU collection on a clean two-T4 machine. See [VALIDATION.md](VALIDATION.md) for executed checks and limitations.

## Level 1: Quick start (Python 3.10–3.13, no GPU)

From a fresh checkout:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python examples/quick_start.py
python -m unittest discover -s tests -v
python scripts/reproduce_tables.py --output outputs/tables
```

The example runs the real evidence-anchoring path on a controlled missing-evidence case, stops before a model forward, and prints `NEED_INFORMATION`. The tests import every research module, check the frozen policy, data completeness, routing, axes, archive safety, and recorded table values. These checks do not load a language model.

Optional real model loading/generation/scoring smoke test on a CUDA GPU:

```sh
python -m pip install -r requirements-gpu.txt
python examples/model_smoke.py --gpu 0
```

This downloads the official pinned Qwen weights if absent. It is a smoke test, not a paper result. The 7B model requires substantial disk space and GPU memory; the recorded NF4 footprint is about 6.91 GiB per GPU during the main collection. Package/CUDA compatibility must be checked against the recorded environments before a full rerun.

## Level 2: Main evaluation and paper table reconstruction

All empirical main tables can be reconstructed from the accepted source data already in Git:

```sh
python scripts/reproduce_tables.py --output outputs/tables-new
```

For an independent evaluation from **raw model records**, download the two checksummed Release assets and install CPU evaluator dependencies:

```sh
python scripts/fetch_artifacts.py --assets all
python -m pip install -r requirements-evaluation.txt
python scripts/evaluate_longitudinal.py --output outputs/raw-evaluation
python scripts/reproduce_tables.py --longitudinal-source outputs/raw-evaluation --output outputs/raw-tables
```

The evaluator reuses the pinned official InjecAgent evaluator, recomputes controlled state effects, checks token-level score/gate arithmetic, and replays the native trajectory with the official AgentDojo simulator and utility. Offline evaluation labels remain separate from all model-facing sealed bundles. There are no live payment, messaging, or other external tool calls.

The downloader validates archive and member SHA-256 values, rejects path traversal, and preserves modified local artifacts. No Kaggle account, private URL, original research directory, or author-specific path is needed. It writes under ignored `artifacts/`. Every command requires a new output directory to preserve old results.

### Recorded main results

| Endpoint | A: Schema/scope | B: + anchoring | C: + dual probe |
| --- | ---: | ---: | ---: |
| Confirmed InjecAgent attack successes / 682 | 83 (+3 unknown) | 83 (+3 unknown) | 0 (+0 unknown) |
| Unsafe controlled executions / 240 | 33 | 5 | 0 |
| Actually adopted malicious argument calls executed | 5/5 | 5/5 | 0/5 |
| Controlled task utility / 240 | 98 | 126 | 109 |
| Benign controlled utility / 120 | 66 | 94 | 78 |
| Verified legal protected calls retained | 47/47 | 47/47 | 30/47 |

The standard benchmark is a longitudinal re-gating of previously viewed prompts/proposals, not a new blind test. The controlled tasks are synthetic one-continuation scenarios, not complete business workflows. Native workspace utility is **0/1 in all three arms**. C rejects 17/47 verified legal protected proposals; the security/utility tradeoff is material. Zero observed attack successes is not a general safety guarantee.

Prefix-control macro AUC is 0.7164 → 1.0000 for the tool name across 69 proposals, with equal weight on three task groups. Both preregistered source-reversal endpoints are positive in 10/10 families across five domains (one-sided domain sign test p=0.03125). Full details, denominators, and limitations are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Level 3: Full reproduction

There is no training stage and no learned adapter/router/checkpoint to recover. The only fitted research state is the published four-parameter calibration policy. Preserve its original bytes and precision.

After downloading the sealed inputs, use the original runners on **two T4 GPUs**, with one independently quantized model replica per GPU:

```sh
python scripts/run_frozen.py --experiment mechanism --output /absolute/new-output/mechanism
python scripts/run_frozen.py --experiment calibration --output /absolute/new-output/calibration
python scripts/run_frozen.py --experiment pilot --output /absolute/new-output/pilot
python scripts/run_frozen.py --experiment longitudinal --output /absolute/new-output/longitudinal
python scripts/run_frozen.py --experiment v1 --output /absolute/new-output/v1
```

Use `--check-only` to validate every bound input and print the command without running inference. These runners preserve the exact experiment-specific code versions, source revisions, prompts, schedules, seeds, and policy. `src/` exposes the final implementation; the historical v1 control uses its separately frozen source. Re-running calibration does **not** replace the released policy. Evaluate a fresh longitudinal output with `scripts/evaluate_longitudinal.py --run /absolute/new-output/longitudinal --output outputs/new-evaluation`.

The v1 command runs the six implemented controls, including the scoped user-masked regeneration control. These are not full reproductions of other defense papers. Unexecuted cross-model, held-out AgentDojo-47, BIPIA, adaptive-attack, and complete strong-baseline studies remain future work. See the [experiment map](REPRODUCIBILITY.md) and [timing protocol](docs/PROTOCOL.md).

## Resources and repository layout

| Path | Purpose |
| --- | --- |
| `src/deployment_probe_v1.py` | Final full-call prefix slicing and dual-axis scoring |
| `src/deployment_guard_v1.py` | Scope, Schema, grounding, and strict-threshold OR guard |
| `src/native_adapter_v3.py`, `src/deployment_rl_v1.py` | Native adaptation and final L/R execution semantics |
| `configs/` | Full-precision policy, model configuration, and tool-risk catalog |
| `data/` | Frozen controlled cases and InjecAgent split IDs |
| `results/source/` | Accepted numeric source observations for table rebuilding |
| `resources/` | File/member checksums, original-source map, and publication policy |
| `scripts/`, `examples/`, `tests/` | Reproduction commands, smoke examples, and integrity tests |

Model: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28), revision `a09a35458c702b33eeacc393d103063234e8bc28`, Apache-2.0. We do not redistribute weights. Necessary public benchmark projections and pinned runtime sources are included in the small frozen-input Release asset, with their MIT notices; canonical raw observations are in the accepted-runs asset. [Resource and license inventory](docs/RESOURCES.md).

## Maintenance and citation

Push/PR tests, scheduled resource checks, and Dependabot run on GitHub. Tags identify immutable research versions; result-affecting fixes require new versions and explicit comparison with the released policy. [MAINTENANCE.md](MAINTENANCE.md) distinguishes platform automation from work requiring a future maintainer session. [CONTRIBUTING.md](CONTRIBUTING.md) describes the review and publication gates.

Code and project-generated controlled data/results are MIT licensed; upstream assets retain their own notices. Cite the software release using GitHub's “Cite this repository” entry or [CITATION.cff](CITATION.cff). No DOI is fabricated; Zenodo archival integration has not been configured.
