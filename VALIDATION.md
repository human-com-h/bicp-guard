# Release validation

Release candidate: **v0.9.0-rc1**, 2026-10-26. The final method, frozen policy, model revision and accepted experiments are unchanged.

## Actually executed during release preparation

| Check | Result / scope |
| --- | --- |
| Fresh isolated Python environment | Created independently; CPU evaluator dependencies installed from PyPI; no global environment changed |
| Research module imports | Every shipped `src/*.py` module imports without loading a model |
| CPU quick start and tests | 12 unittest cases pass; actual missing-evidence guard stops with zero extra forwards |
| Formal source identity | All 33 active research source files match the final experiment's original SHA-256 bindings; policy bytes match `95a3f6af…ce62e99` |
| Frozen-input reconstruction | All 440 archive members match original experiment package manifests; all five runner commands pass `--check-only` |
| Accepted raw archive | All 86 selected merged-record/environment/evaluation members and archive SHA verified; redundant worker streams/checkpoints omitted |
| Independent raw L/R evaluation | 922 unique cases, 2,766 arms, 61,803 token-logprob arithmetic checks; original official InjecAgent evaluator and controlled effects used |
| Native task/state replay | Actual call lists replay through official AgentDojo; final states match; utility remains 0/1 in A/B/C |
| Numeric tables | 13 reconstruction output files generated from observations/configuration; main, six-variant and cost tables from independent raw evaluation match the shipped source-data reconstruction exactly |
| CPU/CUDA scoring arithmetic | Tiny deterministic causal toy model on Torch 2.2.2 / CUDA 11.8; 123 token comparisons per device; maximum absolute error 1.67e-6 (CPU), 0 (CUDA); three-forward slice matches separate targets |
| Official model source | Pinned Qwen configuration URL resolves and reports model_type qwen2; full weights not downloaded for this check |
| Package availability | Published PyPI versions of all declared major GPU pins checked; complete GPU environment installation not performed |
| Publication scan | Project files and both archive contents pass path, known-manuscript fingerprint, document-content and credential-pattern checks; source-data scan found no author-specific machine paths |

## Not claimed

The optional real 7B model-loading smoke test and complete clean two-T4/NF4 recollection have not been run during packaging. The toy model test supplies software arithmetic evidence only. We do not claim new benchmark measurements, hardware-independent latency, population error bounds, cross-model generalization, held-out AgentDojo-47, BIPIA, complete other-defense reproduction or adaptive-attack coverage.

Historical accepted model runs and their environment reports are included. Raw-record replay and deterministic reconstruction are completed validation levels; fresh full model collection remains the RC gate. GPU requirement files combine observed experiment pins with required simulator dependencies; their end-to-end installation/runtime compatibility remains to be validated on the target hardware.

GitHub CI and public download verification are recorded separately in the final release notes once the corresponding remote jobs have actually completed. A configured workflow is not itself a passed run.
