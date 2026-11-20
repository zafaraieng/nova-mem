# NOVA-Mem GitHub publication report

**Audit date:** 2026-11-20  
**Publication status:** PARTIAL  
**Repository:** https://github.com/zafaraieng/nova-mem  
**Default branch:** `main`  
**Source-tree publication commit:** `ba923ed0994a4ba0dd15e2d0d727e8646f186ab2`

## 1. Source archive and SHA-256

The newer source archive, selected by its later upload timestamp, was
`nova-mem-source-release.zip`.

```text
SHA-256 313ef509083ad061ac4bf8f04ba0827b18759fb9302264bd8d061f046031524d
```

The other supplied archive, `nova-mem-source-release-published.zip`, had SHA-256:

```text
4325b7b055642ceea436c836e9bb96fb552510b6d1715a08faa59877f8898f04
```

Both ZIP archives passed ZIP integrity checks and were preserved unchanged.
The newer source ZIP supplied the source-code baseline. The older published
ZIP contained publication metadata but omitted nine Python modules under
`src/nova_mem/data/` and `src/nova_mem/models/`; these modules were restored
from the newer source archive without editing their contents.

## 2. Inputs inspected and unavailable

Inspected:
- `nova-mem-source-release.zip`
- `nova-mem-source-release-published.zip`
- Root project README, package configuration, requirements and ignore rules
- Source tree, scripts, tests, YAML configurations and vendored LongMemEval files
- `EXPERIMENT_STATUS.csv`
- `paper/final_manuscript.md` (historical draft; excluded)
- `RECONCILIATION_REPORT.md` (excluded from public release)

Not found among the supplied archives or accessible Library files:
- a separate file named `publicationreports36`
- `DELIVERY_REPORT_S36.md`
- `AUTHORITATIVE_CHECKPOINT.md`
- `RECONCILIATION_REPORT.md` as a separately supplied source file
- `DISCREPANCIES_AND_DECISIONS.md)

No content was inferred from those missing files. The S-36 status in this
release is limited to the supplied experiment-status record and values
specified for publication.

## 3. Files added, modified, restored and excluded

The GitHub source commit adds the 81 non-README files from the prepared,
privacy-filtered source tree. The preceding initial commit adds the project
README. The resulting remote tree has 82 files.

Key public additions/updates include:
- Project `README.md` with overview, research motivation, architecture,
  installation, use, tests, limitations, attribution and citation guidance.
- `docs/RESEARCH_STATUS.md` with the S-36 status and backend deviation.
- `CITATION.cff`, `THIRD_PARTY_NOTICES.md`, `.gitignore`, and
  `.github/workflows/tests.yml`.
- `docs/ARCHITECTURE.md`, `docs/data_provenance.md`,
  `docs/reproducibility.md`, and `paper/README.md`.
- `EXPERIMENT_STATUS.csv`, configurations, checksum sidecars, scripts,
  source modules, tests, and retained upstream reference files.
- Nine missing source modules restored from the newer source archive:
  `src/nova_mem/data/*` and `src/nova_mem/models/*`.

The historical `paper/final_manuscript.md` and the stale closure figure were
excluded. The reconciliation and internal audit reports were excluded.
No benchmark dataset, experiment-evidence ZIP, private human-review mapping,
unblinding material, credentials, private recovery archive, or model cache was
uploaded.

A repository-wide `LICENSE` file was **not** published. The supplied project
metadata asserted a blanket MIT license, but the project copyright holder and
authority to grant that license were not established by the inspected inputs.
The unverified project-level license claim was removed from `pyproject.toml`
and `CITATION.cff`; the README explains that NOVA-Mem project-code reuse is
not licensed by this release. The root license remains a publication blocker.

## 4. Final repository structure

The 82 tracked files are organized as follows:

```text
.github/workflows/tests.yml
.gitignore
CITATION.cff
EXPERIMENT_STATUS.csv
README.md
THIRD_PARTY_NOTICES.md
configs/
data-templates/
docs/
paper/README.md
prompts/v1/README.md
pyproject.toml
requirements.txt
requirements-dev.txt
scripts/
src/nova_mem/
  analysis/
  data/
  evaluation/
  memory/
  metrics/
  models/
  reranking/
  retrieval/
  utils/
tests/
third_party/longmemeval_official/
```

The remote Git tree was fetched recursively after publication: 82 blobs,
105 entries including directory entries, not truncated. `.gitignore` is
anchored to root-level downloaded-data/model directories so the Python
packages `src/nova_mem/data` and `src/nova_mem/models` remain tracked.

## 5. Dependency installation

Declared runtime and test dependencies are in `requirements.txt`,
`requirements-dev.txt`, and `pyproject.toml`; `rank-bm25>=0.2.2` is
explicitly declared.

A clean virtual environment was created locally, but dependency installation
could not complete: the configured package index returned no available
distributions, beginning with `numpy>=1.26`. An editable package build
succeeded only when dependency resolution was disabled; this is not a
successful clean installation.

GitHub Actions run
[44718088502](https://github.com/zafaraieng/nova-mem/actions/runs/44718088502)
was triggered on the source-tree commit. At the time this report was written,
the hosted job was still in the dependency-install step. No hosted CI pass is
claimed here.

## 6. Actual test results

The complete local pytest invocation in the available interpreter reported:

| Result | Count |
|---|---:|
| Passed | 96 |
| Failed | 8 |
| Skipped | 5 |

All eight failures raised `ModuleNotFoundError: rank_bm25`; that package was
declared but absent from the provided interpreter. Five dataset-dependent
tests skipped because benchmark data is not bundled. No tests were deleted or
silently skipped to improve the result.

Other checks completed successfully: Python source compilation, parsing of
the six YAML configurations, script help/import checks, CFF parsing, status
assertions, package metadata/build check (without dependency resolution),
README internal-link checks, and ignore-rule checks.

## 7. Secret and privacy scan

The complete prepared repository tree was scanned for common credential
patterns and restricted-file indicators. The scan found no high-confidence
credential-pattern matches and no restricted evidence filenames. The remote
file tree was checked after publication: no research-evidence ZIP, raw
benchmark dataset, unblinding key, human-review mapping, private recovery
archive, `.env`, private key, or credential file is present.

This is a pattern and file-tree scan, not a guarantee that every conceivable
secret or sensitive value is detectable. Do not commit credentials or private
research material in future changes.

## 8. Third-party licensing

The vendored LongMemEval reference files were compared with the upstream
`xiaowu0162/LongMemEval` GitHub source. Git blob identifiers for
`run_generation.py`, `evaluate_qa.py`, `print_qa_metrics.py`, and the
upstream MIT `LICENSE` matched the inspected upstream versions. The complete
MIT notice and Di Wu copyright statement remain under
`third_party/longmemeval_official/LICENSE`, with attribution in
`THIRD_PARTY_NOTICES.md`.

The LongMemEval upstream code license does not establish a project-wide
license for NOVA-Mem. A root-level NOVA-Mem license is withheld pending
confirmation of copyright ownership and redistribution authority.

## 9–10. GitHub URL, default branch and commit

- Repository URL: https://github.com/zafaraieng/nova-mem
- Visibility: public
- Default branch: `main`
- Source-tree commit:
  `ba923ed0994a4ba0dd15e2d0d727e8646f186ab2`
- Initial README commit:
  `eb85687a70295b3aa6f2be491f2829b23c77b07e`

The target repository was confirmed empty before the first commit. The source
tree was added as a fast-forward update; no force-push or overwrite of prior
remote history occurred. The README, workflow, CFF, project metadata,
research-status document and complete recursive tree were fetched back from
GitHub after publication.

## 11. Publication status

**PARTIAL** — the audited source tree and public documentation are on GitHub,
but a clean dependency installation and successful complete test run have not
been established, hosted CI was still running at report time, and the
repository-wide project license remains unresolved.

## 12. Remaining tasks

1. Confirm the NOVA-Mem copyright holder and the permission/license terms
   intended for project-owned code; only then add an authorized root
   `LICENSE` and matching metadata.
2. Resolve package-index access, install the declared dependencies including
   `rank-bm25`, and rerun the complete test suite. Review all failures if any
   remain.
3. Inspect the completed GitHub Actions run and address any installation or
   test failures; do not describe a pending run as passing.
4. Obtain the missing S-36 checkpoint/delivery/discrepancy records if a fuller
   provenance reconciliation is required. They were not available to this
   audit.
5. Continue experiments and human review only under their approved research
   protocol. C-Dense remains partial and unjudged; C-Hybrid, C-RRF,
   D-BM25 rerank, E-v2 remain not started; E-v1 remains a one-example pilot;
   human review remains 0/144; Test-400 remains locked at 2/11 gate criteria.

No claim is made that NOVA-Mem is a completed scientific study, that the
ZAI-backed results are comparable to the frozen Ollama/Qwen protocol, or that
human review has been completed.
