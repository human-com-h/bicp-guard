# Reproducibility map

## Supplementary validation v2.2 (main-branch addition)

[The supplementary protocol and results](docs/SUPPLEMENT_V2_2.md) cover paired prompt contracts, Mistral fixed-proposal mechanisms, and native answer reservation. Run `python scripts/reproduce_supplement.py --output outputs/supplement` on CPU to verify the archives and reconstruct all 435 fixed scoring cells plus eight native probes; `--bootstrap` additionally uses NumPy 1.26.4 to reproduce the accepted intervals. `python scripts/run_supplement.py --check-only` verifies all 67 original collector inputs without loading a model. Utility labels retain the independent official evaluation from acceptance; the default CPU command does not rerun that evaluator.

`resources/supplement_v2_2_manifest.json` binds the original input archive, raw-output archive, source tables, model configurations, and collector code. Collection inputs exclude evaluator answers; evaluation labels are held separately with the recorded outputs. The original release and its immutable assets remain unchanged. A fresh complete GPU reproduction has not been performed.

The code and evaluation artifacts support **BiCP: Reasoning-Prefix Deconfounding for Dual-Axis Counterfactual Probing and Operation-Risk Routing**. Original experiment identifiers, release tags, and archived source bytes retain their historical versions. Manuscripts are excluded. The map below identifies internal table numbers solely to associate code/data with research claims.

## Evidence to command mapping

| Internal table / figure | Fixed input and execution | Accepted output → reconstruction |
| --- | --- | --- |
| Table 1: operation routing | `configs/tool_risk_layers.json`; official AgentDojo revision below | Tool registrations → `table_01_router.csv` |
| Table 2: policy | `configs/frozen_policy.json`; 390-block calibration | Original parameter bytes → `table_02_policy.csv` |
| Table 3: execution procedure | `src/deployment_guard_v1.py`, `src/deployment_pilot_v1.py` | Algorithm description; not an empirical numeric table |
| Table 4: v1 six-arm diagnostic | Sealed `v1/`; `--experiment v1` | `artifacts/runs/v1/records.jsonl`; `results/source/v1/outcomes.csv` → `table_04_v1.csv` |
| Table 5 / prefix figure | Sealed `mechanism/`; 69 historical proposals × three prefix treatments | Raw `mechanism/records.jsonl`; `thought_scores.csv` → `table_05_prefix.csv` |
| Table 6 / source-reversal figure | Same mechanism run; 10 families × four cells in five domains | `factorial_cells.csv` → family contrasts and domain sign tests |
| Table 7: evidence chain | Bundle/member manifests in `resources/` | Provenance inventory, not a new statistical endpoint; archived historical audit counts are not sample sizes |
| Calibration figure and group sensitivity | Sealed `calibration/`; `--experiment calibration` | `accepted_score_table.json`, `all_proposals.csv`, `leave_group_out.csv`; token-log scores → `calibration_events.csv`; no refitting in reconstruction |
| Table 8 / pilot axes | Sealed `pilot/`; `--experiment pilot`; 28 paired units | Raw `pilot/records.jsonl`; source `events.csv`/`unit_outcomes.csv` → `table_08_pilot.csv` |
| Table 9 / standard benchmark | Sealed `longitudinal/`; `--experiment longitudinal`; 922 L cases | `longitudinal_records.jsonl` + official/controlled evaluator → unit/event rows → `table_09_standard.csv` |
| Table 10 / T4 cost figure | Same L/R run; 430 protected probe events | Measured per-event `probe_total_ms` → `table_10_cost.csv`; empirical linear-interpolated P95 |
| Table 11: Gmail errors | Four explicit-user sensitive-read calibration controls | Original token logs and view means → `table_11_gmail.csv` |
| Table 12: artifact directory map | This file and `resources/source_manifest.json` | Provenance listing; no numerical result |
| Table 13: six controlled variants | Same 240 L cases, all A/B/C arms | Unit-level task success → `table_13_variants.csv` |
| Native workspace trajectory | Same L/R run, one clean task with three independent arm trajectories | `workspace_record.json`; official task/runtime replay → `workspace.csv`; utility 0/1 in every arm |

Run `python scripts/reproduce_tables.py --output outputs/tables` for all implemented reconstruction outputs. Table identifiers 3, 7, and 12 describe algorithms/provenance and are not reconstructed as fabricated measurement tables. Figures' underlying numeric observations are included; this release does not redistribute manuscript figure files or a full figure-rendering pipeline.

## Fixed protocol

- Model: Qwen/Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28`; no finetuning.
- Quantization: bitsandbytes NF4, double quantization, float16 compute; Transformers 4.51.3, bitsandbytes 0.45.5, SDPA. Exact per-run Python, torch, CUDA, GPU and package reports are inside the accepted-runs archive under each run's `workers/*/environment.json` (and calibration `preflight/environment.json`).
- Native simulator: AgentDojo benchmark v1.2.2, revision `089ed468cf3ed0322acc66b0211f26d9d90dbf60`; Pydantic 2.10.3. The scoped runtime compatibility loader changes import initialization; official tools, tasks, and utility logic are retained.
- InjecAgent: `f19c9f2c79a41046eb13c03c51a24c567a8ffa07`; 1,054 base cases grouped by user task; development/calibration/test = 186/186/682. Original task-group IDs and fixed 682-case schedule are supplied. Missing cache outcomes remain unknown.
- Controlled data: 100 entity instantiations of 25 families × six variants = 600 scenarios. Group splits and IDs are frozen in the provided manifest/bundles; the main longitudinal portion has 240 cases from ten families. They are not 600 independent real workflows.
- Greedy generation: seed 42 for final mechanism/deployment experiments, max context 8,192 tokens, max new tokens 512, prompt budget 7,680. Counterfactual scoring checks context plus target separately.
- Policy: `alpha_name=1`, `theta_name=0.02706692410243683`, `alpha_arg=2`, `theta_arg=1.2999669161113965`. Strict `>` comparisons; joint OR including pre-probe rejection. Policy SHA-256: `95a3f6afcd8fe47aeade2a5fa6499b2ccaadc4d0683ea8968bea02d7dce62e99`.
- Scoring: without current-step reasoning, serial three-view complete-call prefix slicing, full-vocabulary float32 log-softmax. The argument feature subtracts full/name risk at the **same** `alpha_arg`; it does not subtract a name score at a different alpha.
- Generation, scoring and offline evaluation stay separate. Model-facing sealed deployment bundles contain no utility/answer labels. The mechanism diagnostic includes offline labels for controlled contrasts; these are public benchmark/synthetic annotations, not private human data.

## Seeds and immutable versions

Final generation seed 42, paired calibration timing seed 2718, pilot allocation seed 137, existing group resampling seed 31415 (10,000 draws), and v1 final resampling seed 42 are preserved. The older conditional-data builder and v1 configurations contain historical date-shaped seeds. They are retained only to reproduce their already-completed version; do not use these defaults for a new experiment or rewrite them to pretend the old experiment used seed 42. New experiments must declare a non-date seed and a new protocol version.

## Evaluation and variation

`evaluate_longitudinal.py` checks 922 unique cases, exact historical proposal texts, unchanged policy, real-measurement flags, block-before-dispatch state, gate decisions, and score arithmetic from token logs. It uses the official InjecAgent output evaluator for attacks and the existing controlled oracle for task effects. It optionally replays actual native calls using official utility (enabled by default). `--skip-native` explicitly omits only native utility and needs just NLTK beyond the standard library.

The prerelease has not repeated full model collection in a new two-T4 environment. There is no empirically established across-run tolerance for ASR or utility; do not invent one. Recorded-data reconstruction is deterministic (normal floating-point rounding aside). Frozen runtime health checks compare six-forward and sliced scoring with maximum token-logprob tolerance 0.1 and feature tolerance 0.05; observed calibration discrepancies were substantially smaller. These are numerical screening tolerances, not permission to change a threshold or claim invariant discrete decisions.

The 922 benchmark had already been viewed during v1 development. Calibration uses a separate split, but v2 longitudinal results are not blind confirmation. Prefix AUCs are macro-averages over three groups; source-reversal tests use five domains as units. The 2/75 calibration false rejection rate is fitted-sample evidence, not a population bound. On transferred benchmark calls it rises to 17/47. Complete strong-defense reproductions, cross-model performance, held-out native-47, BIPIA and adaptive attacks have not been run.

## Full collection commands and resource expectations

See README Level 3. The same wrapper supports the mechanism study, 390-block calibration, 28-unit pilot, 922+one-task final run, and historical v1 baselines. It first checks every sealed file against original hashes and prints/runs the original command in that bundle's directory. Use fresh output paths outside the bundle. No reconstruction script creates new model measurements, chooses new hyperparameters, silently drops invalid rows, or refits the frozen release policy.

Collectors enforce their original hardware contract (two Tesla T4 replicas for the final run). Single-GPU and CPU quick starts do not replace that experiment. Model download/cache location is handled by Hugging Face; do not place the cache inside a sealed input root. CPU table reconstruction requires little memory; raw-record evaluation temporarily reads hundreds of megabytes of JSON. Decompressed raw/input artifacts occupy several hundred megabytes, and base-model weights require additional disk capacity.
