# NOVA-Mem — Current Research Status (S-36 snapshot)

**Snapshot date:** 2026-11-20  
**Scope:** recorded experimental execution and validation state at S-36.

This page is the status record for the S-36 research snapshot. NOVA-Mem's
software framework and the execution state of individual experiments are
separate: implemented or configured software paths do not imply that an
experiment has been run. The figures below describe the recorded S-36
snapshot and should be cited with the backend and protocol notes that follow.

## Experimental status

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

The table records execution status, not software availability. No accuracy
value is inferred for a condition without a recorded judged result.

## Backend and protocol provenance

The scored dev-100 results for A baseline, Oracle aligned, and C-BM25 were
produced using the **ZAI `glm-4-plus` backend**, rather than the frozen
Ollama/Qwen configuration. The partial C-Dense continuation also used the
ZAI backend.

These ZAI-backed measurements are **not results of the frozen Ollama/Qwen
protocol** and must **not** be described as directly comparable to that
protocol or to Ollama/Qwen runs. No backend-equivalence claim is made. Any
future result should identify its backend and configuration alongside its
metrics.

## Findings and interpretation boundaries

The S-36 dev-100 local-judge accuracy values are descriptive measurements
under the ZAI continuation configuration. They do not, by themselves,
establish an architecture ranking, causal superiority, or generalization to
other model backends. The Oracle condition is a reference condition; it should
not be framed as a deployable memory system or as a directly comparable
frozen Ollama/Qwen result.

C-Dense has six generated outputs and none judged. Its accuracy is therefore
not available. E-v1 is a one-example cold pilot, not a completed experimental
condition.

## Research and validation activity

The following status is preserved for S-36:

- **Retrieval experiments:** C-Hybrid and C-RRF had not started; D-BM25
  reranking had not started.
- **Structured memory:** E-v1 consisted of a one-example cold pilot; E-v2
  had not started.
- **Human validation:** 0 of 144 planned labels were completed. No claim of
  completed human review or human-validated accuracy is made.
- **Held-out evaluation:** Test-400 remained locked. Only 2 of 11 gate
  criteria were met; no claim is made that the locked test set was evaluated.

For the distinction between available code paths, conducted experiments,
and remaining validation work, see
[RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md).

## Software verification and scientific evaluation

Software tests check implementation behavior, interfaces, and selected
regression properties. A successful unit-test or CI run does not complete
the experimental matrix, establish cross-backend comparability, satisfy
human-review requirements, or unlock Test-400.

The 2026-11-20 publication audit recorded the following local test invocation
in the provided interpreter:

| Result | Count |
|---|---:|
| Passed | 96 |
| Failed | 8 |
| Skipped | 5 |

The eight local failures raised `ModuleNotFoundError: rank_bm25`, because
that declared dependency was absent from the local interpreter. A separate
clean virtual environment could not install dependencies through the
configured package index, which returned no available distributions starting
with `numpy>=1.26`. These local results are not represented as successful.

The hosted GitHub Actions runs installed declared dependencies, including
`rank-bm25`, and passed the offline test suite. At the latest S-36
publication-maintenance verification (run #5), CI reported **104 passed,
5 skipped**. The five skipped tests require locally downloaded LongMemEval
benchmark data or a locally generated split. They are dataset-gated software
checks; they are not research experiments and do not count as completed
scientific conditions. See `GITHUB_PUBLICATION_REPORT.md` for run IDs and
audit details.

## Data and privacy

The public source repository does not redistribute raw benchmark JSON,
private research evidence, human-review mappings, unblinding materials,
credentials, or internal recovery archives. Unit tests that require local
dataset files skip when those files are absent. Consult
[data_provenance.md](data_provenance.md) before acquiring or using benchmark
data.
