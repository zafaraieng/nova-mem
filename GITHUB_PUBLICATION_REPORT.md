# NOVA-Mem — Post-Publication Presentation and Repository Audit

**Audit date:** 2026-11-20  
**Repository:** https://github.com/zafaraieng/nova-mem  
**Default branch:** `main`  
**Presentation commit:** `a2c20fb5ef54d0f66b8472d797cd683cff9dd1f9`  
**Final CI workflow restoration commit:** `3d299866808f7a294cba5468a9e94589a81da375`  
**Publication status:** PARTIAL — repository presentation and hosted software CI are verified; GitHub About metadata and project-level licensing still require owner action.

## 1. Repository audit

The existing public repository was inspected before editing. The audit covered:

- `README.md`, `docs/RESEARCH_STATUS.md`, and `GITHUB_PUBLICATION_REPORT.md`
- `CITATION.cff`, `THIRD_PARTY_NOTICES.md`, `pyproject.toml`, and the retained upstream LongMemEval license
- `docs/ARCHITECTURE.md`, `docs/reproducibility.md`, and `docs/data_provenance.md`
- Relevant implementation files for retrieval, reranking, memory systems, model clients, evaluation, scripts, tests and YAML profiles
- The current `main` branch, commit history, recursive Git tree, repository metadata and GitHub Actions runs

The presentation work did not rebuild the project or modify source-code modules, experiment ledgers, dataset checksums, or scientific evidence.

## 2. README redesign

The main README now opens with:

> **NOVA-Mem — A Resource-Aware Evaluation Framework for Conversational Memory Strategies**

Its introduction positions NOVA-Mem as research software and describes the experimental framework, evaluation pipelines, retrieval implementations, configuration management, and reproducibility documentation around cleaned LongMemEval.

The README has 13 sections:

1. Overview
2. Research motivation
3. Research objectives
4. Framework architecture
5. Experimental methodology
6. Implemented memory strategies
7. Installation and usage
8. Dataset and evaluation
9. Experimental findings
10. Current research snapshot
11. Research roadmap
12. Limitations and reproducibility
13. Citation and acknowledgements

The opening no longer leads with a negative research-status label. The software capabilities appear before the experimental snapshot. The README still makes clear that code availability is distinct from experiment execution and that hosted CI does not establish scientific completion.

### Capabilities verified against the source

The README describes only code paths confirmed in the current source tree:

- A no-memory, B full-context, and Oracle evidence-only context systems
- BM25 and dense retrieval
- Hybrid min-max fusion and reciprocal-rank fusion
- Cross-encoder reranking
- Structured-memory fact extraction, evidence-linked storage, and supersession handling
- Model clients/adapters, generation cache, QA/retrieval metrics, run ledger and analysis utilities
- YAML configurations, data fingerprints, scripts and tests

Hash/deterministic embedders and mock readers/rerankers are explicitly described as test/plumbing tools, not scientific evidence.

### README validation

The remote `README.md` and public GitHub blob path
[README.md](https://github.com/zafaraieng/nova-mem/blob/main/README.md)
were fetched after commit. All 11 relative README documentation/file links were checked against the remote Git tree and resolved. The Markdown source and link targets were verified. A browser screenshot of GitHub's rendered page was not available through the connected GitHub integration, so pixel-level visual rendering was not independently inspected.

## 3. Research-status documentation

`docs/RESEARCH_STATUS.md` is now titled and introduced as the **Current Research Status (S-36 snapshot)**. Its opening explains that the page records a dated research snapshot and separates implementation availability from experimental execution.

The numerical status remains unchanged:

| Condition | Generated | Judged | Accuracy | Status |
|---|---:|---:|---:|---|
| A baseline | 100/100 | 100/100 | 0.04 | scored dev-100 |
| Oracle aligned | 100/100 | 100/100 | 0.76 | scored dev-100 |
| C-BM25 | 100/100 | 100/100 | 0.33 | scored dev-100 |
| C-Dense | 6/100 | 0/100 | not available | partial; unjudged |
| C-Hybrid | 0/100 | 0/100 | not available | not started |
| C-RRF | 0/100 | 0/100 | not available | not started |
| D-BM25 rerank | 0/100 | 0/100 | not available | not started |
| E-v1 | 1/1 | 1/1 | not reported | one-example cold pilot only |
| E-v2 | 0/100 | 0/100 | not available | not started |
| Human review | — | 0/144 labeled | not available | no labels completed |
| Test-400 | — | — | — | locked; 2/11 gate criteria met |

The ZAI `glm-4-plus` deviation remains explicit. The scored dev-100 A, Oracle aligned, and C-BM25 runs used ZAI rather than the frozen Ollama/Qwen configuration. The partial C-Dense continuation also used ZAI. Those ZAI-backed results are not directly comparable to the frozen Ollama/Qwen protocol.

The page retains the local test record (96 passed, 8 failed, 5 skipped) and clearly distinguishes hosted software CI from scientific evidence.

## 4. Research roadmap

Created `docs/RESEARCH_ROADMAP.md`. It separates:

- Implemented software capabilities.
- Experiments recorded at S-36.
- Experiments requiring execution (C-Dense continuation/judging, C-Hybrid, C-RRF, D-BM25 rerank and E-v2).
- Validation requiring future work (human review, Test-400 gate, cross-backend comparability and model-path validation).

No deadlines, completed milestones, new results, or experimental claims were invented.

## 5. Citation and attribution

Updated `CITATION.cff` to use the professional project title and describe the actual software capabilities. The citation instructions point to the S-36 research-status page and its backend deviation.

No author, affiliation, ORCID, DOI, venue, or project license field was invented.

`THIRD_PARTY_NOTICES.md` and `third_party/longmemeval_official/LICENSE` were not modified. The retained upstream MIT license and Di Wu copyright notice remain specific to the separately attributed LongMemEval reference material; they do not license all NOVA-Mem code.

## 6. CI and test verification

The workflow at `.github/workflows/tests.yml` uses:

- `actions/checkout@v7.0.1`
- `actions/setup-python@v7.1.0`

It also sets `permissions: contents: read` and excludes report-only commits from test triggers. The v7 action lines are supported by their current official release records and use the Node 24 runtime. Sources: [checkout v7.0.1 release](https://github.com/actions/checkout/releases/tag/v7.0.1), [setup-python v7.1.0 release](https://github.com/actions/setup-python/releases/tag/v7.1.0), and [setup-python v7 documentation](https://github.com/actions/setup-python#readme).

### GitHub Actions runs

All seven runs returned by the repository's Actions runs endpoint were inspected. All completed successfully. Run #6 was an intermediate documentation commit that briefly carried the older v4/v5 workflow; its log shows the Node 20 deprecation warning. That workflow was restored to v7/v7 in the following commit and revalidated by run #7.

| Run | Commit | Result | Pytest summary | Action versions / warning |
|---|---|---|---|---|
| [#1 — 44718088502](https://github.com/zafaraieng/nova-mem/actions/runs/44718088502) | `ba923ed0994a` | success | 104 passed, 5 skipped (0.38s) | checkout v4 / setup-python v5; Node 20 warning |
| [#2 — 44718250779](https://github.com/zafaraieng/nova-mem/actions/runs/44718250779) | `ceedbcad2869` | success | 104 passed, 5 skipped (0.51s) | checkout v4 / setup-python v5; Node 20 warning |
| [#3 — 44718460506](https://github.com/zafaraieng/nova-mem/actions/runs/44718460506) | `f8cfa56c5ea3` | success | 104 passed, 5 skipped (0.66s) | checkout v4 / setup-python v5; Node 20 warning |
| [#4 — 44738350541](https://github.com/zafaraieng/nova-mem/actions/runs/44738350541) | `da1ff7c008e7` | success | 104 passed, 5 skipped (0.71s) | checkout v7.0.1 / setup-python v7.1.0; no Node 20 warning |
| [#5 — 44738968047](https://github.com/zafaraieng/nova-mem/actions/runs/44738968047) | `95ddc27b405e` | success | 104 passed, 5 skipped (0.39s) | checkout v7.0.1 / setup-python v7.1.0; no Node 20 warning |
| [#6 — 44739545340](https://github.com/zafaraieng/nova-mem/actions/runs/44739545340) | `a2c20fb5ef54` | success | 104 passed, 5 skipped (0.72s) | transient checkout v4 / setup-python v5; Node 20 warning |
| [#7 — 44739692403](https://github.com/zafaraieng/nova-mem/actions/runs/44739692403) | `3d299866808f` | success | 104 passed, 5 skipped (0.65s) | checkout v7.0.1 / setup-python v7.1.0; no Node 20 warning |

For run #7, the job log confirms successful installation of the declared dependencies, the explicit `rank-bm25` import check, and the pytest step. It ended with **104 passed, 5 skipped in 0.65s**. The log contains no Node 20 deprecation warning.

### Skipped tests

The five skipped tests are dataset-presence checks. They require locally acquired cleaned LongMemEval files and/or a generated split that are intentionally not bundled in the public repository. They are skipped software checks, not experimental conditions and not evidence that the benchmark evaluations were completed.

### Local test result

The latest recorded local invocation remains:

| Result | Count |
|---|---:|
| Passed | 96 |
| Failed | 8 |
| Skipped | 5 |

The eight local failures raised `ModuleNotFoundError: rank_bm25`, because that declared dependency was absent from the local interpreter. A clean local virtual environment could not install dependencies through the configured package index, starting with `numpy>=1.26`. These local results were not rerun during this presentation task and are not represented as passing.

Hosted CI and local results are reported separately. A successful hosted test suite verifies the tested software paths; it does not complete NOVA-Mem's broader experimental study or replace human review.

## 7. GitHub About metadata

The live repository API still reports:
- Description: empty (`null`).
- Topics: empty (`[]`).
- Visibility: public.
- Default branch: `main`.

The connected GitHub actions in this environment provide repository reads and file/commit operations but no About-settings write operation. The description/topics therefore remain **not applied**.

Recommended description:

> Research software for evaluating conversational memory strategies under resource constraints.

Recommended topics:

`artificial-intelligence`, `llm`, `conversational-memory`, `long-term-memory`, `retrieval-augmented-generation`, `longmemeval`, `research-software`, `python`.

The repository owner must enter these values in GitHub's About settings.

## 8. Project-level licensing

No root-level `LICENSE` was added. GitHub reports `license: null`, and the root `LICENSE` path returns 404. `CITATION.cff` and `pyproject.toml` contain no NOVA-Mem project-license declaration.

The visible Git history is not sufficient to establish ownership of every original project file or the authority to license it. The owner must confirm file provenance, any external/institutional contributions or assignments, and the license they intend to grant. Until then, no project-wide reuse or redistribution permission is asserted.

The LongMemEval MIT notice and upstream license remain unchanged and limited to the attributed third-party material.

## 9. Privacy and research-integrity verification

The remote recursive Git tree was inspected after the documentation/workflow changes. It contained 84 blobs (107 tree entries including directories), and the tree response was not truncated. Filename/path scanning found no:

- research-evidence ZIP or raw LongMemEval JSON dataset;
- unblinding key or private human-review mapping;
- internal recovery archive;
- `.env`, credentials, token, PEM/private key, or model cache.

No research experiment was launched. The S-36 condition counts, ZAI backend deviation, non-comparability warning, human-review count, and Test-400 gate state are preserved.

## 10. Files changed

| File | Change |
|---|---|
| `README.md` | Repositioned as professional research software documentation; added 13 sections, architecture, methodologies, available implementations, findings, status snapshot, roadmap, limits, citation and license note. |
| `docs/RESEARCH_STATUS.md` | Neutral S-36 snapshot framing; preserved every count, accuracy value, backend caveat, and software-test distinction. |
| `docs/RESEARCH_ROADMAP.md` | New document separating code capabilities, conducted experiments, future experiment work, and validation gates. |
| `CITATION.cff` | Updated project title, keywords, citation guidance and abstract; no invented authors or license. |
| `.github/workflows/tests.yml` | Restored and retained v7.0.1/v7.1.0 actions, read-only permission, and report-only path exclusions. |
| `GITHUB_PUBLICATION_REPORT.md` | This audit update. |
| `THIRD_PARTY_NOTICES.md` and upstream MIT `LICENSE` | Inspected; unchanged. |
| `pyproject.toml` | Inspected; unchanged and has no project license field. |
| Root `LICENSE` | Remains absent pending the owner's decision and rights confirmation. |

No source modules, tests, experiment data, ledgers, frozen configurations, or historical evidence were modified or deleted in this presentation pass.

## 11. Final state and remaining owner actions

The latest code/workflow commit verified by run #7 is:

`3d299866808f7a294cba5468a9e94589a81da375`

The publication report itself is updated on `main` after run #7; report-only commits are excluded from CI triggers by the workflow.

**Completed:** professional README and research-document structure, S-36 status preservation, research roadmap, citation metadata refinement, Node 24-compatible CI, seven successful hosted workflow runs, final remote privacy scan, and link validation.

**Still requiring owner action:**
1. Set the GitHub About description and eight recommended topics manually.
2. Confirm copyright ownership/contributor permissions and select the actual project license before adding a root `LICENSE`.
3. Acquire the authorized benchmark files locally when the five dataset-gated tests need to be exercised.

**Final assessment: PARTIAL.** The public repository's technical presentation and hosted software CI are verified. GitHub About metadata and project-level licensing remain unresolved. The research status remains exactly as recorded at S-36; this presentation work does not claim scientific completion, backend comparability, or completed human review.
