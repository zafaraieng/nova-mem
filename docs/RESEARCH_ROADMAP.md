# NOVA-Mem — Research Roadmap

This roadmap separates software capabilities already present in the repository
from the execution of experiments and the validation activities recorded at
S-36. It is a status map, not a schedule: no deadlines or unverified milestones
are implied.

## 1. Implemented software capabilities

The following components are present in the source tree and covered by
software tests to varying degrees:

| Area | Available capability |
|---|---|
| Data | LongMemEval question/session loading, split handling, dataset statistics and fingerprints |
| Model clients | Common inference-client interface, Ollama and ZAI adapters, OpenAI-compatible path, generation cache |
| Memory strategies | No-memory, full-context, Oracle context, retrieval-augmented, retrieval-plus-reranking and structured-memory implementations |
| Retrieval | BM25, dense embeddings, hybrid score fusion and reciprocal-rank fusion |
| Reranking | Cross-encoder adapter and deterministic mock reranker for tests |
| Structured memory | Fact extraction pipeline, evidence-linked facts, temporal ordering and supersession handling |
| Evaluation | Answer judging, QA metrics, retrieval metrics and per-question retrieval traces |
| Analysis | Run ledger, paired bootstrap, exact McNemar test, Holm correction and report/aggregation scripts |
| Reproducibility | YAML configuration profiles, checksum references, provenance documentation and CI workflow |

Software availability should not be interpreted as evidence that the associated
research condition has been executed.

## 2. Experiments recorded at S-36

| Condition | S-36 record | Interpretation |
|---|---|---|
| A baseline | 100/100 generated and judged; accuracy 0.04 | scored dev-100 |
| Oracle aligned | 100/100 generated and judged; accuracy 0.76 | scored dev-100 reference condition |
| C-BM25 | 100/100 generated and judged; accuracy 0.33 | scored dev-100 |
| C-Dense | 6/100 generated; 0 judged | partial; accuracy not available |
| E-v1 | 1/1 generated and judged | one-example cold pilot only |

**Backend deviation:** the scored dev-100 A, Oracle aligned, and C-BM25 runs
used the ZAI `glm-4-plus` backend rather than the frozen Ollama/Qwen
configuration. The partial C-Dense continuation also used ZAI. The recorded
ZAI results are not directly comparable to the frozen Ollama/Qwen protocol.
This caveat must accompany references to the affected results.

## 3. Experiment work requiring execution

At the S-36 snapshot, the following runs were not started:

- C-Hybrid retrieval.
- C-RRF retrieval.
- D-BM25 retrieval with cross-encoder reranking.
- E-v2 structured-memory condition.

C-Dense had only 6/100 generated answers and 0 judged; completing its
generation and judging remains separate work. E-v1 remains a one-example
pilot, not a completed condition.

Any further execution should retain protocol versioning, backend identity,
model/configuration provenance, data fingerprints, judge configuration, and
per-question outputs. A change in backend or frozen setting must be reported
as a protocol deviation, not silently pooled with another configuration.

## 4. Validation activities requiring future work

| Validation activity | S-36 state | Gate/interpretation |
|---|---|---|
| Human review | 0/144 labeled | no completed human review is claimed |
| Test-400 held-out evaluation | locked | 2/11 gate criteria met; do not evaluate until gate requirements are satisfied |
| Cross-backend comparability | not established | ZAI glm-4-plus results are not comparable to frozen Ollama/Qwen |
| Dense/reranker model validation | not established by the six C-Dense generations | real model paths require their configured weights and a documented execution protocol |
| End-to-end reproduction | hosted offline tests pass; dataset-gated tests skip without data | software CI does not replace benchmark reproduction |

## 5. Reporting principles

For every future result, report at least:

- condition and configuration/profile identifier;
- generated and judged counts;
- metric and judge identity;
- model backend and model identifier;
- data/split fingerprint;
- deviations from the frozen configuration;
- missing, skipped, failed, or gated work.

Do not infer values for unjudged runs, label a pilot as a completed condition,
or claim human validation when labels are not complete. Test-400 remains
locked until the recorded gate criteria are met.
