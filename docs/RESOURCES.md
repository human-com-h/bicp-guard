# Resource, access and license inventory

| Resource | Required role | Access / identifier | Redistribution decision |
| --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | Base generator and likelihood scorer | Official Hugging Face repository; revision `a09a35458c702b33eeacc393d103063234e8bc28` | Apache-2.0; downloaded from official source, weights not reuploaded |
| Four-parameter policy | Final deployed decision state | `configs/frozen_policy.json`, SHA in REPRODUCIBILITY | Included verbatim; no additional learned weights |
| Tool-risk catalog | Runtime operation semantics | `configs/tool_risk_layers.json`; implementation/schema hashes and upstream source paths included | Project-authored catalog, MIT |
| InjecAgent base cases, tools, prompts, cache and evaluator | Attack evaluation and fixed inputs | [Official repository](https://github.com/uiuc-kang-lab/InjecAgent/tree/f19c9f2c79a41046eb13c03c51a24c567a8ffa07) | Necessary pinned subset in frozen-input archive; original MIT notice in archive and `licenses/` |
| AgentDojo simulator, benchmark tasks and initial state | Native/discovery mechanism controls and utility evaluation | [Official repository](https://github.com/ethz-spylab/agentdojo/tree/089ed468cf3ed0322acc66b0211f26d9d90dbf60), benchmark v1.2.2 | Pinned runtime/task files in mechanism/calibration archive; strictly projected runtime in final model-facing bundle; MIT notice preserved |
| Controlled conditional scenarios | Fixed synthetic oracle and proposals | `data/conditional/` and sealed experiment input versions | Project-generated, MIT; deterministic original 600 rows retained |
| Historical proposals / cached continuations | Viewed-benchmark L re-gating | Sealed longitudinal input; 994 public stage inputs (922 stage1 + 72 available stage2) | Project model outputs derived from public/synthetic benchmarks, included; reachability remains conditional |
| Calibration scores/labels | Inspect fitted policy and false rejections | Accepted score table in Git; full 390 raw blocks in accepted-runs archive | Included; offline labels separated from collector inputs |
| Accepted raw runs | Audit model outputs, token logs, timing and state effects | Two immutable Release assets in `resources/artifacts.json` | Included canonical merged streams; redundant workers/checkpoints and account receipts omitted |
| ToolBench mirror / BIPIA | Not required for the final reported method's main tables | Historical adaptation only | No mirror dataset redistributed; neither is presented as a completed generalization/full-utility result |

There is no finetuned model, adapter, learned router, vector index or private dataset required by the final method. Official benchmark records use synthetic evaluation personas and simulated services. Runtime files that mention secrets/addresses as attack targets are benchmark text, never local credentials. No manuscript, literature PDFs, reviewer material, internal account receipts or credentials are in the public delivery.

All archive members have SHA-256 and byte counts. `fetch_artifacts.py` downloads a pinned Release URL and verifies both levels of integrity. The original source-to-public-file map records provenance without publishing the research directory. Archived collector output is optional for fresh model collection but required for offline raw-record verification. The main frozen-input asset is sufficient to regenerate the model measurements with official base weights; no request to the authors is needed.

The project MIT license applies to the authors' code, synthetic scenarios, annotations and generated outputs. It does not replace upstream notices or license external base-model weights. Upstream data/task files are preserved with their MIT notices. No known unresolved redistribution restriction was found for the selected material; other historical files are excluded.
