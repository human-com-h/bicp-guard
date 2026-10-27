# BiCP

**Reasoning-Prefix Deconfounding for Dual-Axis Counterfactual Probing and Operation-Risk Routing**

Zida Hao and Yubo Li  
School of Software, Xinjiang University

## Overview

BiCP is a training-free defense against indirect prompt injection in tool-using language agents. It evaluates a proposed tool call before execution through three components:

- **Reasoning-prefix deconfounding:** remove the current-step reasoning prefix before counterfactual scoring.
- **Dual-axis probing:** score the tool name and complete call to detect both tool hijacking and argument substitution.
- **Operation-risk routing:** allow scoped discovery without additional model forwards, while applying evidence anchoring and counterfactual checks to protected operations.

The implementation uses Qwen2.5-7B-Instruct with NF4 quantization. Complete-call prefix slicing evaluates the protected proposal with three serial forward passes.

## Main results

The historical 922-case results below compare three configurations: **A**, Schema and scope checks; **B**, A with evidence anchoring; and **C**, B with the BiCP dual-axis probe.

| Metric | A | B | C: BiCP |
| --- | ---: | ---: | ---: |
| Confirmed InjecAgent attack successes / 682 | 83 (+3 unknown) | 83 (+3 unknown) | 0 |
| Unsafe controlled executions / 240 | 33 | 5 | 0 |
| Adopted malicious argument calls executed | 5/5 | 5/5 | 0/5 |
| Controlled task success / 240 | 98 | 126 | 109 |
| Benign controlled task success / 120 | 66 | 94 | 78 |
| Legal protected calls retained | 47/47 | 47/47 | 30/47 |
| Native workspace task success | 0/1 | 0/1 | 0/1 |

This evaluation replays fixed historical prompts and proposals; the controlled tasks are synthetic single-continuation scenarios. Unknown outcomes reflect unavailable cached continuations. The prefix-control study improves tool-name macro AUC from 0.7164 to 1.0000 across 69 proposals, and the source-reversal study yields positive contrasts in all ten families across five domains.

Experiment definitions, additional results, and evaluation details are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

### Supplementary validation v2.2

With the same frozen policy and fixed proposals, aligning the prompt contract to calibration raises legal protected-call retention from **30/47 (63.83%) to 43/47 (91.49%)**. A calibration-only frozen suffix candidate retains **44/47 (93.62%)**. Of 47 paired malicious targets, anchoring stops 40 evidence-unsupported substitutions and both probes stop all seven observation-supported injected targets.

**Mistral-7B-Instruct-v0.3** reproduces prefix deconfounding on 69 historical proposals (name macro AUC **0.6509 → 1.0000**) and positive source reversals in ten families across five domains (joint **p=0.03125**). This is fixed-proposal scoring, with zero Mistral action generation.

Thirty native development trajectories across four suites show Travel's five-step answer success under advance answer planning, Banking safe stops, and remaining Workspace extraction/guard-utility errors. The eighth-step hard prefix activates zero times. These results do not replace the historical benchmark or establish broad native generalization.

See [supplement protocol, strata, and results](docs/SUPPLEMENT_V2_2.md). Reconstruct its scores and tables without a GPU:

```sh
python scripts/reproduce_supplement.py --output outputs/supplement
```

The existing `v0.9.0-rc1` tag and assets are unchanged; the supplement is an addition on the main branch.

## Installation

Use Python 3.10 or later:

```sh
git clone https://github.com/human-com-h/bicp-guard.git
cd bicp-guard
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv/Scripts/Activate.ps1`.

## Quick start

Run the evidence-anchoring example on CPU:

```sh
python examples/quick_start.py
```

The example processes a controlled case with missing evidence and returns `NEED_INFORMATION` before model inference.

To load the base model and run a small generation and scoring example on a CUDA GPU:

```sh
python -m pip install -r requirements-gpu.txt
python examples/model_smoke.py --gpu 0
```

The model is downloaded from the official [Qwen2.5-7B-Instruct repository](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28). The experiments use revision `a09a35458c702b33eeacc393d103063234e8bc28`.

## Evaluation

### Rebuild result tables

Generate the tables from the included evaluation source data:

```sh
python scripts/reproduce_tables.py --output outputs/tables
```

This step runs on CPU and produces CSV tables and a combined JSON file without model inference.

### Evaluate recorded model outputs

Download the experiment inputs and raw outputs, then run the official benchmark evaluators and controlled-task evaluation:

```sh
python scripts/fetch_artifacts.py --assets all
python -m pip install -r requirements-evaluation.txt
python scripts/evaluate_longitudinal.py --output outputs/evaluation
python scripts/reproduce_tables.py --longitudinal-source outputs/evaluation --output outputs/evaluated-tables
```

The downloaded [experiment resources](https://github.com/human-com-h/bicp-guard/releases/tag/v0.9.0-rc1) include fixed inputs, model outputs, token log probabilities, timing records, and the frozen calibration policy. Files are checked against their SHA-256 manifests and extracted into `artifacts/`. Use a new output directory for each run.

### Run the experiments

The original experiments use two Tesla T4 GPUs, with one quantized model replica per GPU. After installing `requirements-gpu.txt` and downloading the resources above, run:

```sh
# Reasoning-prefix and source-reversal studies
python scripts/run_frozen.py --experiment mechanism --output outputs/runs/mechanism

# Calibration and paired closed-loop evaluation
python scripts/run_frozen.py --experiment calibration --output outputs/runs/calibration
python scripts/run_frozen.py --experiment pilot --output outputs/runs/pilot

# Main longitudinal and native-task evaluation
python scripts/run_frozen.py --experiment longitudinal --output outputs/runs/longitudinal

# Historical six-arm control study
python scripts/run_frozen.py --experiment v1 --output outputs/runs/v1
```

Add `--check-only` to check the experiment inputs and inspect its command. Each runner uses the corresponding fixed code, prompts, splits, seeds, and configuration. The method requires no training; the released calibration policy is in `configs/frozen_policy.json`.

To evaluate a new longitudinal run:

```sh
python scripts/evaluate_longitudinal.py --run outputs/runs/longitudinal --output outputs/new-evaluation
```

## Repository structure

| Path | Contents |
| --- | --- |
| `src/deployment_probe_v1.py` | Dual-axis scoring and complete-call prefix slicing |
| `src/deployment_guard_v1.py` | Scope checks, evidence anchoring, and execution guard |
| `src/native_adapter_v3.py` | Native tool and observation adaptation |
| `src/deployment_rl_v1.py` | Longitudinal and native-task execution |
| `configs/` | Model settings, calibration policy, and tool-risk catalog |
| `data/` | Controlled cases and fixed split IDs |
| `results/source/` | Evaluation data for result-table reconstruction |
| `scripts/` | Experiment, evaluation, and result-processing commands |
| `examples/`, `tests/` | Minimal examples and automated tests |

## Documentation

- [Reproducibility guide and experiment map](REPRODUCIBILITY.md)
- [Scoring and timing protocol](docs/PROTOCOL.md)
- [Model, data, and resource inventory](docs/RESOURCES.md)
- [Environment and validation details](VALIDATION.md)

## Citation

If you use this code or the evaluation resources, please cite:

```bibtex
@software{hao2026bicp,
  author = {Hao, Zida and Li, Yubo},
  title = {BiCP: Reasoning-Prefix Deconfounding for Dual-Axis Counterfactual Probing and Operation-Risk Routing},
  year = {2026},
  url = {https://github.com/human-com-h/bicp-guard}
}
```

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

## License

This project is released under the [MIT License](LICENSE). Third-party resources retain their respective licenses; see [licenses/](licenses/) and the [resource inventory](docs/RESOURCES.md).
