# Maintenance and handover

This repository uses author-supervised maintenance. The GitHub account owner remains the final authority for scientific changes and exceptional destructive operations. A maintainer agent executes authorized routine work using the CLI/API and reads this file and REPRODUCIBILITY first.

## What runs without an active agent

- GitHub Actions: push/PR tests on Python 3.10 and 3.12; module imports, functional example, deterministic table reconstruction, compile checks, and publication audit of all fetched history.
- Weekly/workflow-dispatch resource checks: Release assets are downloaded and SHA-verified; official base-model configuration URL is checked; raw L/R evaluation, official native utility and table reconstruction run on CPU.
- Dependabot: weekly GitHub Actions and pip dependency update proposals. Updates must pass checks and review; no automatic merge or alteration of frozen experiment requirements is configured.

These are GitHub platform jobs. No persistent agent watcher or scheduled agent automation is configured. Issue investigation, PR review, compatibility fixes, scientific impact assessment and new Releases require a future maintainer session or an explicitly configured agent schedule. Ending a chat does not leave an agent permanently monitoring this repository.

## Next-session procedure

1. Read README, this file, CHANGELOG, VALIDATION, and the fixed artifact manifests. Inspect `git status`, remotes, branches/tags, latest Release, Actions runs, open issues and PRs for `human-com-h/bicp-guard` only.
2. Use the existing account's CLI/API authentication without printing credentials. Do not modify global Agent settings or unrelated repositories.
3. Diagnose the issue; preserve original raw runs and policy. Make ordinary fixes in a branch, update tests/docs/CHANGELOG, reconstruct relevant tables, and compare with the released artifacts.
4. Run `python scripts/audit_publication.py --history` plus checks for all intended Release assets. The reviewed file inventory must match. Update `resources/release_files.json` only after inspecting each new/changed file. Review benchmark text separately from actual secrets; do not blanket-ignore PDF/TEX formats.
5. Create/review/merge a normal PR. Do not force-push shared history, overwrite another person's changes, delete a repository or retained Release, or redistribute materials with unclear ownership. Escalate only those exceptions or genuine account authentication/scientific decisions.
6. A patch release may fix packaging/compatibility without changing scientific values. A result-affecting change requires a new protocol/version, fresh outputs in new directories, comparison with this release, and author review. Never move an existing paper-version tag or silently replace its assets.

## Release gate

Use semantic versions; current release is an RC because clean-machine full two-T4 model collection has not been repeated. Stable v1.0.0 requires verifying the GPU installation/runner end to end, reviewing any changed results, and closing documented reproduction gaps. Create a commit, immutable tag, Release notes and checksummed resources together. Reference the exact policy/source SHA and list precisely which checks ran. Preserve old asset URLs and tags.

If unauthorized manuscript material is found, preserve a local backup and stop further publication. Safely remove it from planned content immediately. Inspect every branch, tag, commit, Release asset, workflow artifact and other generated public copy. Latest-tree deletion alone is insufficient; assess history rewriting and obtain confirmation for destructive/shared-history actions. External clones/caches may remain recoverable.

Zenodo DOI registration is not configured. A future persistent DOI must be obtained from the actual repository/record and verified before insertion into README or CITATION.cff. Do not manufacture a paper DOI, arXiv identifier, or acceptance status.
