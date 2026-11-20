# NOVA-Mem — Post-Publication Audit Report

**Audit date:** 2026-11-20  
**Publication status:** PARTIAL — source publication and hosted CI are verified; project licensing and GitHub About metadata remain unresolved.  
**Repository:** https://github.com/zafaraieng/nova-mem  
**Default branch:** `main`  
**Latest code/workflow maintenance commit:** `da1ff7c008e7f4c733ca796f274777701c33fcc7`

## 1. Current remote state

GitHub confirms that the repository is public, the default branch is `main`, and the latest code/workflow maintenance commit is `da1ff7c008e7f4c733ca796f274777701c33fcc7`. The prior head was `f8cfa56c5ea3cf34d175f1fdba08adbc4d9c9515`; the maintenance update is a normal fast-forward and did not recreate the repository, overwrite history, or force-push.

The latest remote README, `CITATION.cff`, `pyproject.toml`, workflow, third-party notices, and licensing status were fetched from GitHub during this audit.

## 2. GitHub Actions — all current runs verified

The GitHub Actions runs endpoint returned four runs for this repository. Each run's job record and log were inspected. All four runs completed successfully:

| Run | Commit | Workflow result | Pytest output | Action runtime |
|---|---|---|---|---|
| [#1 — 44718088502](https://github.com/zafaraieng/nova-mem/actions/runs/44718088502) | `ba923ed0994a` source-tree publication | success | 104 passed, 5 skipped (0.38s) | checkout v4 / setup-python v5; Node 20 deprecation warning |
| [#2 — 44718250779](https://github.com/zafaraieng/nova-mem/actions/runs/44718250779) | `ceedbcad2869` report commit | success | 104 passed, 5 skipped (0.51s) | checkout v4 / setup-python v5; Node 20 deprecation warning |
| [#3 — 44718460506](https://github.com/zafaraieng/nova-mem/actions/runs/44718460506) | `f8cfa56c5ea3` report update | success | 104 passed, 5 skipped (0.66s) | checkout v4 / setup-python v5; Node 20 deprecation warning |
| [#4 — 44738350541](https://github.com/zafaraieng/nova-mem/actions/runs/44738350541) | `da1ff7c008e7` README and CI maintenance | success | 104 passed, 5 skipped (0.71s) | checkout v7.0.1 / setup-python v7.1.0; no Node 20 deprecation warning |

For all four runs, dependency installation, the explicit `rank-bm25` assertion, and the pytest step succeeded. Run #4 fetched and executed both upgraded actions successfully. Its log confirms `rank-bm25 0.2.2` was installed and importable.

### Skipped tests

The five skips are conditional dataset-presence checks, not passing experimental evaluations. The benchmark and split files are intentionally not bundled with this repository. The skipped tests are:

- `test_real_file_spot_checks` — cleaned S dataset absent.
- `test_oracle_file_consistency` — cleaned S and Oracle datasets absent.
- `test_committed_split_protocol` — local split and cleaned S dataset absent.
- `test_stats_real_file_regression` — cleaned S dataset absent.
- `test_set_flag_recorded_in_summary` — cleaned S dataset absent.

No benchmark dataset was added to the repository to eliminate these skips.

## 3. Local versus hosted tests

The last recorded local invocation remains:

| Result | Count |
|---|---:|
| Passed | 96 |
| Failed | 8 |
| Skipped | 5 |

The eight local failures reported `ModuleNotFoundError: rank_bm25`: that dependency was declared but absent from the local interpreter. The previous audit also recorded that a clean local virtual environment could not install dependencies because the configured package index returned no distributions, starting with `numpy>=1.26`.

These local results have **not** been represented as passing and were not rerun during this targeted maintenance task. They are distinct from the four successful hosted CI runs, in which dependencies installed and the full offline test suite completed.

The five local skips are the same dataset-gated tests listed above.

## 4. CI maintenance changes

The workflow now uses:

- `actions/checkout@v7.0.1`
- `actions/setup-python@v7.1.0`

The official action documentation identifies checkout v7 and setup-python v7 as the current release lines. Setup-python v7 is the post-Node-20 release line; the action's release notes describe the v7 ESM migration and no change to inputs, outputs, or behavior. See the [checkout changelog](https://github.com/actions/checkout/blob/main/CHANGELOG.md), [checkout releases](https://github.com/actions/checkout/releases), and [setup-python releases](https://github.com/actions/setup-python/releases).

GitHub removed Node 20 from Actions runners on **September 23, 2026**; runners now use Node 24 for JavaScript actions. See the [official GitHub Changelog notice](https://github.blog/changelog/2026-09-23-node-20-is-no-longer-available-in-github-actions/). The new run used the upgraded actions and did not emit the previous Node 20 deprecation warning.

Additional workflow maintenance:
- Added `permissions: contents: read` for least-privilege checkout/testing.
- Added `paths-ignore` for `GITHUB_PUBLICATION_REPORT.md` on push and pull request events. Report-only edits do not rerun the software suite; changes to source, tests, README, configuration, or workflow still do.
- No experiment was launched and no research result was generated.

## 5. README, citation, package metadata, and repository licensing

### README

The README was fetched from the remote default branch after the update. It retains all S-36 disclosures and now explains why the public Git history is insufficient to establish sole copyright ownership or licensing authority.

### Citation and package metadata

- `CITATION.cff` was inspected. It contains no unverified author, affiliation, ORCID, DOI, or license field.
- `pyproject.toml` was inspected. It contains no project-level license declaration.
- Neither file was changed in this maintenance pass.

### Root project license

GitHub's repository API reports `license: null`; fetching the root `LICENSE` path returns 404. No root-level NOVA-Mem license was added.

The visible Git history before this maintenance task contained four commits attributed to the repository owner's GitHub account. Git author/committer metadata does not prove copyright ownership, establish employment or contract IP rights, or rule out contributions made outside the visible history. The current source tree also contains known third-party LongMemEval material, which is separately attributed.

The available evidence is therefore insufficient to determine:
1. who owns every original NOVA-Mem file;
2. whether all original project files were written solely by the repository owner;
3. whether external or institutional contributions/assignments exist; or
4. which license the owner actually intends to grant.

The root license remains absent. The repository owner must confirm file provenance, third-party contribution permissions, and any institutional or contractual obligations, then choose the intended license. No license choice has been inferred.

The separate LongMemEval MIT notice and `THIRD_PARTY_NOTICES.md` were not modified. Their remote blob identifiers remain unchanged from the prior audit; the MIT license in `third_party/longmemeval_official/LICENSE` continues to carry Di Wu's upstream copyright notice. That license applies to the attributed upstream material only, not to all NOVA-Mem code.

## 6. GitHub About description and topics

The live GitHub repository metadata currently reports:
- Description: empty (`null`).
- Topics: empty (`[]`).
- Visibility: public.
- Default branch: `main`.

The connected GitHub actions available in this environment support repository reads and file/commit changes, but do not expose a repository-settings write operation for description/topics. These About values have therefore **not** been configured by this audit.

Recommended description:

> Resource-aware research software for evaluating conversational memory strategies using LongMemEval.

Recommended topics:

`artificial-intelligence`, `llm`, `conversational-memory`, `long-term-memory`, `retrieval-augmented-generation`, `longmemeval`, `research-software`, `python`.

The repository owner must add these values in the repository's About settings.

## 7. Remote file/privacy verification

The current recursive Git tree contains 83 blobs (106 tree entries including directories); the response was not truncated. A filename/path scan found no research-evidence ZIP, raw LongMemEval JSON, unblinding key, private human-review mapping, internal recovery archive, credential file, `.env`, PEM/private key, or model cache. No evidence files were introduced by this maintenance change.

No research experiment was launched. The status remains:
- A baseline: 100/100 judged, accuracy 0.04.
- Oracle: 100/100 judged, accuracy 0.76.
- C-BM25: 100/100 judged, accuracy 0.33.
- C-Dense: 6/100 generated, 0 judged.
- C-Hybrid and C-RRF: not started.
- D-BM25 rerank: not started.
- E-v1: one-example cold pilot only.
- E-v2: not started.
- Human review: 0/144.
- Test-400: locked, 2/11 gate criteria met.

The ZAI `glm-4-plus` backend deviation remains explicit. These results are not comparable to the frozen Ollama/Qwen protocol. Successful CI validates software tests only; it does not complete, validate, or replace the scientific experiments.

## 8. Exact changes in this maintenance pass

| File | Change |
|---|---|
| `README.md` | Expanded project ownership/license caveat; retained all research-status disclosures. |
| `.github/workflows/tests.yml` | Upgraded checkout/setup-python actions to Node-24-compatible releases; set read-only contents permission; excluded report-only commits from CI. |
| `GITHUB_PUBLICATION_REPORT.md` | Updated to this report with verified CI runs, local result distinction, license finding, and About metadata limitation. |
| `CITATION.cff` | Inspected; unchanged. |
| `pyproject.toml` | Inspected; unchanged and has no project license field. |
| `THIRD_PARTY_NOTICES.md` and upstream MIT `LICENSE` | Inspected; unchanged. |
| Root `LICENSE` | Remains absent; no license was invented or added. |

## 9. Remaining tasks

1. **Project licensing:** the owner must verify copyright ownership, any third-party contributions/assignments, and choose the actual project license before a root `LICENSE` can be added.
2. **GitHub About:** manually set the recommended description and topics; these settings could not be written through the connected GitHub actions.
3. **Dataset-gated tests:** download the authorized public benchmark data locally when needed to run the five skipped dataset-dependent tests. Do not commit the dataset to this source repository.

**Final assessment: PARTIAL.** The public source repository remains intact, all four visible hosted workflow runs are successful, the updated Node-24-compatible CI workflow is verified, and the local test failures are separately and accurately recorded. Project-level licensing and GitHub About metadata remain unresolved. NOVA-Mem remains incomplete research software, not a completed scientific study.
