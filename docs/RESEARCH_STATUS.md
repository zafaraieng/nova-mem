# NOVA-Mem research status — S-36

**Status date:** 2026-11-20
**Overall status:** incomplete research / development-stage software. This is
not a completed scientific study, and the current measurements do not support
a ranking or causal conclusion about memory architectures.

## Experimental status

| Condition | Generated | Judged | Accuracy | Status |
|---|---:|---:|---:|---|
| A baseline | 100/100 | 100/100 | 0.04 | scored dev-100 |
| Oracle | 100/100 | 100/100 | 0.76 | scored dev-100 |
| C-BM25 | 100/100 | 100/100 | 0.33 | scored dev-100 |
| C-Dense | 6/100 | 0/100 | not available | partial; unjudged |
| C-Hybrid | 0/100 | 0/100 | not available | not started |
| C-RRF | 0/100 | 0/100 | not available | not started |
| D-BM25 rerank | 0/100 | 0/100 | not available | not started |
| E-v1 | 1/1 | 1/1 | not reported | one-example cold pilot only |
| E-v2 | 0/100 | 0/100 | not available | not started |
| Human review | — | 0/144 labeled | not available | not completed |
| Test-400 | — | — | — | locked; 2/11 gate criteria met |

The counts above describe the S-36 research status. An experiment's existence
in the codebase or configuration does not mean it has been executed.

## Mandatory backend deviation

The scored dev-100 results for A baseline, Oracle, and C-BM25 were produced
using the **ZAI `glm-4-plus` backend**, rather than the frozen Ollama/Qwen
configuration. The partial C-Dense continuation also used the ZAI backend.
This deviation must accompany any mention of these measurements.

These ZAI-backed results are **not results of the frozen Ollama/Qwen
protocol** and must **not be described as comparable** to that protocol or to
Ollama/Qwen runs. No backend-equivalence claim is made.

## Incomplete work and safeguards

- C-Dense has only 6 generated answers and none judged.
- C-Hybrid, C-RRF, D-BM25 rerank, and E-v2 have not started.
- E-v1 is a one-example cold pilot, not a completed condition.
- Human review is incomplete: 0 of 144 labels are complete. No human-review
  completion or human-validated accuracy claim is made.
- Test-400 remains locked. Only 2 of its 11 gate criteria are met; no claim is
  made that the test set was evaluated.

Accordingly, the project does not claim a completed scientific study,
state-of-the-art performance, experimental superiority, or a validated
comparison between the listed memory approaches.

## Software verification in the 2026-11-20 publication audit

The complete pytest suite was invoked in the provided interpreter: **96 passed,
8 failed, 5 skipped**. Every observed failure raised
`ModuleNotFoundError: rank_bm25`; `rank-bm25` is declared in
`requirements.txt`, but was not installed in that interpreter. A clean virtual
environment was also created, but dependency installation could not proceed
because the configured package index returned no available distributions
(starting with `numpy>=1.26`). Therefore the full suite has **not** been
independently reproduced in a clean environment during this audit. See
`GITHUB_PUBLICATION_REPORT.md` for exact verification details.

## Software versus scientific evidence

The repository distributes source code, configuration templates, tests, and
publicly shareable provenance documentation. It does not distribute raw
benchmark datasets, private research evidence, human-review mappings,
unblinding materials, credentials, or internal recovery archives. Unit tests
check software behavior; they do not complete or validate the research
experiments.
