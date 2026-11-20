# Architecture overview

The codebase uses a Python `src/` package, CLI scripts, YAML configuration,
and pytest tests.

| Component | Responsibility |
|---|---|
| `src/nova_mem/data/` | Benchmark loading, question/session models, split handling, dataset statistics and fingerprints |
| `src/nova_mem/models/` | Inference-client interfaces, Ollama/OpenAI-compatible/ZAI clients, generation cache |
| `src/nova_mem/memory/` | Memory-system abstractions, prompt templates and structured-memory implementation |
| `src/nova_mem/retrieval/` | Retrieval interfaces and retrieval implementations |
| `src/nova_mem/reranking/` | Reranking adapters |
| `src/nova_mem/evaluation/` | Answer-judge prompt flow and QA metrics |
| `src/nova_mem/metrics/` | Metrics package namespace |
| `src/nova_mem/analysis/` | Experiment-ledger aggregation and statistical summaries |
| `src/nova_mem/utils/` | Shared utilities and logging |
| `scripts/` | Data preparation, experiment/pilot execution, validation, aggregation and reporting |
| `configs/` | Experiment profiles and backend registries |
| `tests/` | Unit and regression tests, including prompt-parity and integrity checks |
| `third_party/longmemeval_official/` | Attributed upstream LongMemEval reference scripts under their retained MIT license |

The repository contains configuration and implementation paths for multiple
conditions. See [research status](RESEARCH_STATUS.md): not all configured
conditions have been run.
