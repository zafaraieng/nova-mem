# NOVA-Mem — A Resource-Aware Evaluation Framework for Conversational Memory Strategies

[![Tests](https://github.com/zafaraieng/nova-mem/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/zafaraieng/nova-mem/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

NOVA-Mem is a research software project investigating conversational memory
strategies under resource constraints. It provides an experimental framework,
evaluation pipelines, retrieval implementations, configuration management,
and reproducibility documentation built around the cleaned LongMemEval
benchmark.

The software is organized to make memory-system choices, model backends,
retrieval settings, evaluation outputs, and experiment provenance explicit.
Its research snapshot and protocol deviations are documented separately from
the framework implementation.

## 1. Overview

Long conversations create a practical systems challenge: an assistant must
identify useful information from a growing history while balancing answer
quality, inference cost, storage, and retrieval latency. NOVA-Mem provides a
common software framework for examining several memory and context strategies
under a configurable evaluation pipeline.

The repository includes:

- A Python `src/` package for dataset handling, memory systems, model clients,
  retrieval, reranking, evaluation, metrics, caching, and experiment analysis.
- Reusable YAML profiles for local development, evaluation, frozen settings,
  and a separately identified ZAI continuation backend.
- Scripts for dataset preparation, experiment execution, retrieval evaluation,
  smoke checks, result aggregation, and report generation.
- Unit and regression tests, prompt-parity checks against attributed upstream
  LongMemEval code, and a GitHub Actions test workflow.
- Dataset checksums, provenance notes, research-status documentation, and a
  reproducibility guide.

## 2. Research motivation

Conversational memory is not a single retrieval operation. Systems can retain
all context, retrieve evidence when answering, rerank candidate memories, or
maintain structured facts across sessions. Each design introduces different
trade-offs in computation, storage, latency, context length, and the handling
of changing information.

NOVA-Mem is designed to make these choices measurable and inspectable. The
project emphasizes explicit configurations, consistent data handling,
traceable retrieval outputs, backend identification, and clear separation
between software verification and scientific findings.

## 3. Research objectives

The framework supports investigation of:

1. **Context and memory strategies:** compare no-memory, full-context,
   evidence-only, retrieval-augmented, reranked, and structured-memory paths.
2. **Retrieval design:** examine lexical, dense, and fused candidate retrieval
   at session granularity.
3. **Resource use:** record generation, retrieval, reranking, caching, token,
   runtime, and structured-storage measurements where available.
4. **Evaluation discipline:** use explicit splits, dataset fingerprints,
   judge prompts, per-question records, and reproducible configuration files.
5. **Research provenance:** identify model backends, deviations, data sources,
   and the status of each condition rather than treating configured code as
   completed experimental evidence.

## 4. Framework architecture

```text
src/nova_mem/
  data/         LongMemEval loading, question/session models, splits and fingerprints
  models/       inference clients, backend adapters and generation cache
  memory/       memory-system interfaces, prompts and structured memory
  retrieval/    BM25, dense, hybrid and reciprocal-rank fusion
  reranking/    cross-encoder reranking adapters
  evaluation/   answer judging and QA evaluation
  metrics/      retrieval metrics and aggregation
  analysis/     experiment ledger, paired statistics and summaries
  utils/        shared utilities

configs/        local, evaluation, frozen and backend-specific YAML profiles
scripts/        data preparation, experiments, retrieval evaluation and reports
tests/          unit, regression, integration and prompt-parity checks
third_party/    attributed LongMemEval upstream reference code
docs/           architecture, provenance, research status and reproducibility
```

The package separates dataset and model interfaces from memory strategies and
retrieval components. Experiment scripts assemble these components from YAML
configuration and record run metadata in the experiment ledger. See
[Framework Architecture](docs/ARCHITECTURE.md) for component responsibilities.

## 5. Experimental methodology

The repository's evaluation pipeline is built around cleaned LongMemEval
inputs and explicit local/evaluation configuration profiles.

- **Data preparation:** retrieve the selected upstream benchmark files,
  validate SHA-256 checksums, load question/session records, and use the
  configured dev/test split.
- **Configuration:** keep retrieval parameters, generation settings, reader
  models, judge settings, prompt version, and backend selection in YAML.
  Frozen configuration files are versioned rather than silently edited.
- **Execution:** run the selected memory system over the requested split and
  persist answers, retrieval traces, timing/token information, and ledger
  metadata.
- **Evaluation:** compute answer and retrieval metrics, with analysis utilities
  for paired bootstrap summaries, exact McNemar tests, and Holm adjustment
  where compatible per-question runs are available.
- **Reproducibility:** retain data fingerprints and run metadata, distinguish
  mock plumbing checks from real model runs, and document configuration
  deviations alongside the measurements.

The existence of an implementation or configuration does not by itself mean
that its experiment has been executed. Current evidence is summarized in
[Research Status](docs/RESEARCH_STATUS.md).

## 6. Implemented memory strategies

The codebase contains the following strategy paths:

| Strategy | Software implementation |
|---|---|
| A — No memory | Question-only answer path |
| B — Full context | Date-ordered conversation history with configurable session/context limits |
| Oracle | Evidence-only context reference |
| C — Retrieval augmented | Session retrieval with BM25 or dense embeddings |
| C — Hybrid fusion | Equal-weight min-max score fusion of BM25 and dense candidates |
| C — Reciprocal rank fusion | Rank-based BM25/dense fusion |
| D — Retrieval plus reranking | Candidate retrieval followed by cross-encoder reranking |
| E — Structured memory | Fact extraction, evidence-linked structured storage, and supersession handling |

Real dense and reranking runs require the configured model dependencies and
model weights. Deterministic/hash embedders and mock readers/rerankers are
provided for software tests and plumbing checks; their outputs are not
scientific results.

## 7. Installation and usage

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e .
```

On Windows, activate the environment with
`.venv\\Scripts\\activate`.

Run the offline suite:

```bash
python -m pytest tests/
```

Run a mock plumbing check in a disposable working copy:

```bash
python scripts/run_smoke_test.py
```

Run a small mock experiment (software plumbing only, not research evidence):

```bash
python scripts/run_experiment.py --system A --split dev --limit 3 --backend mock
```

Prepare locally obtained benchmark data and inspect script options:

```bash
python scripts/prepare_data.py
python scripts/run_experiment.py --help
python scripts/eval_retrieval.py --help
```

The BM25 path requires `rank-bm25`, which is included in
`requirements.txt`. Dense retrieval and cross-encoder reranking use
Sentence Transformers and may download model weights when first invoked.

## 8. Dataset and evaluation

NOVA-Mem uses the cleaned LongMemEval benchmark. Dataset JSON files are not
redistributed in this repository. Obtain them from the
[LongMemEval cleaned dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
and review the upstream dataset terms before use.

The repository keeps checksum references under `data-templates/` and the
data-preparation procedure in `scripts/prepare_data.py`. Downloaded data,
derived splits, caches, model assets, and experiment outputs remain local.
See [Dataset Provenance](docs/data_provenance.md).

The test workflow deliberately does not download the dataset. Dataset-gated
tests therefore skip when the required local benchmark files are absent.

## 9. Experimental findings

The S-36 snapshot contains three scored dev-100 conditions:

| Condition | Judged | Local-judge accuracy |
|---|---:|---:|
| A baseline | 100/100 | 0.04 |
| Oracle aligned | 100/100 | 0.76 |
| C-BM25 | 100/100 | 0.33 |

**Protocol note:** the scored dev-100 A, Oracle, and C-BM25 runs used the
ZAI `glm-4-plus` backend rather than the frozen Ollama/Qwen configuration.
These measurements are not results of the frozen Ollama/Qwen protocol and
must not be described as directly comparable to it. The partial C-Dense
continuation also used the ZAI backend.

See [docs/RESEARCH_STATUS.md](docs/RESEARCH_STATUS.md) for the full condition
table, generated/judged counts, and validation state.

## 10. Current research snapshot

The S-36 snapshot records the following additional execution and validation
state:

- C-Dense: 6/100 generated, 0 judged.
- C-Hybrid, C-RRF, and D-BM25 rerank: not started at this snapshot.
- E-v1: one-example cold pilot, 1/1 judged.
- E-v2: not started.
- Human review: 0/144 labeled.
- Test-400: locked, with 2/11 gate criteria met.

These are time-bounded research-status records, not statements about whether
corresponding software modules exist. The status page preserves the complete
snapshot and the backend caveat.

## 11. Research roadmap

The roadmap distinguishes the available software from experimental execution
and validation activity. It does not assign dates or imply unrecorded progress.

| Area | Current position | Research/validation activity |
|---|---|---|
| Software framework | A/B/Oracle, RAG, fusion, reranking and structured-memory code paths are present | Maintain tests, configuration provenance and backend adapters |
| Scored conditions | A, Oracle and C-BM25 have 100/100 judged S-36 records | Preserve the ZAI backend deviation; any frozen-protocol comparison requires runs under the frozen Ollama/Qwen configuration |
| Dense retrieval | C-Dense has 6/100 generated, 0 judged | Complete generation and judging under an explicitly recorded protocol |
| Additional retrieval/reranking | C-Hybrid, C-RRF and D-BM25 rerank had not started at S-36 | Execute and evaluate only under the approved experiment protocol |
| Structured memory | E-v1 is a one-example pilot; E-v2 had not started | Further implementation-level and experimental validation is required |
| Human validation | 0/144 labels completed at S-36 | Complete the approved human-review process and report agreement/limitations |
| Held-out evaluation | Test-400 locked; 2/11 gate criteria met | Satisfy the gate criteria before any test-set evaluation |

See [Research Roadmap](docs/RESEARCH_ROADMAP.md) for the detailed separation
of implemented capabilities, conducted experiments, and remaining validation.

## 12. Limitations and reproducibility

- The S-36 scored dev-100 conditions used the ZAI `glm-4-plus` backend,
  not the frozen Ollama/Qwen configuration; no cross-backend comparability is
  claimed.
- C-Dense and the remaining conditions have the execution state reported
  above. Human review and held-out Test-400 validation remain gated.
- Benchmark files are obtained separately and are not included in this
  repository. External model weights and model-serving backends may be needed
  for real dense, reranking, and generation runs.
- The ZAI bridge process is not bundled. Never commit API keys or credentials.
- Hosted CI verifies software tests only. It does not complete experiments,
  establish scientific conclusions, or substitute for human review.

Use [Research Status](docs/RESEARCH_STATUS.md),
[Research Roadmap](docs/RESEARCH_ROADMAP.md), and the
[Reproducibility Guide](docs/reproducibility.md) when reproducing or citing
specific findings.

## 13. Citation and acknowledgements

See [CITATION.cff](CITATION.cff) for software citation metadata. Cite the
LongMemEval benchmark and upstream work separately when using that benchmark.
No author identifiers, DOI, or publication venue are asserted here because
they have not been verified.

The vendored LongMemEval reference scripts and copied prompt-template
material are attributed under the upstream MIT license in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). That upstream license does
not grant a project-wide license for NOVA-Mem code.

AI tools assisted parts of software development and documentation. This is
disclosed for provenance and does not constitute experimental evidence,
scientific validation, or authorship attribution.

### Project licensing

No repository-wide license is currently granted for NOVA-Mem project code.
The root-level `LICENSE` remains absent because project ownership,
redistribution authority, and any third-party contributions or assignments
have not been independently established. Public Git history alone does not
prove copyright ownership. Do not assume that public visibility grants reuse
or redistribution rights. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the separately licensed
upstream LongMemEval material.
