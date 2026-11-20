# NOVA-Mem

**Research software for resource-aware evaluation of conversational-memory
strategies on the cleaned LongMemEval benchmark.**

> **Research status: INCOMPLETE.** S-36 scored dev-100 results are A baseline
> 0.04 (100/100 judged), Oracle 0.76 (100/100), and C-BM25 0.33 (100/100).
> C-Dense is 6/100 generated and unjudged; C-Hybrid, C-RRF, D-BM25 rerank and
> E-v2 have not started; E-v1 is a one-example cold pilot. Human review is
> 0/144 and Test-400 is LOCKED (2/11 gate criteria met). **All scored dev-100
> results used ZAI glm-4-plus, not the frozen Ollama/Qwen backend; they are not
> comparable to that protocol.** See
> [docs/RESEARCH_STATUS.md](docs/RESEARCH_STATUS.md) before citing any result.

NOVA-Mem explores how an assistant can use long conversation histories:
without memory, with full context, with retrieved evidence, or with a
structured memory store. The repository contains the software pipeline,
configuration templates, tests, and public provenance notes. It does not
distribute benchmark data or non-public research artifacts.

## Scope and research motivation

The project investigates the engineering and evaluation requirements for
memory strategies under constrained compute budgets. Its present public
value is the implementation and its explicit record of incomplete and
backend-deviant research—not a completed scientific comparison or a claim of
state-of-the-art performance.

## Architecture

```text
src/nova_mem/
  data/        LongMemEval loading, splitting, statistics and fingerprints
  models/      inference clients, generation cache and ZAI backend adapter
  memory/      memory-system implementations and prompt registry
  retrieval/   BM25, dense and hybrid retrieval implementations
  reranking/   cross-encoder reranking adapters
  evaluation/  answer judging and QA metrics
  metrics/     retrieval and evaluation metrics
  analysis/    experiment-ledger aggregation and statistics
  utils/       shared utilities
scripts/       dataset preparation, experiment execution, validation and reports
configs/       experiment and backend configuration templates
prompts/       prompt registry documentation
third_party/   MIT-licensed LongMemEval upstream reference scripts
tests/         offline unit and regression tests
```

The existence of code paths for a condition does not mean that the
corresponding experiment has been run. See the research-status table above.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e .
```

The declared dependencies include `rank-bm25`, which is required by the BM25
retrieval path. Dense retrieval, reranking, and structured-memory components
may download Sentence Transformers/PyTorch model weights when used. See
[docs/reproducibility.md](docs/reproducibility.md) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Data acquisition

The raw cleaned LongMemEval benchmark is **not redistributed** here. Obtain it
from the upstream
[LongMemEval cleaned dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
and run:

```bash
python scripts/prepare_data.py
```

The script downloads the selected dataset files, verifies checksums, and
creates the local dev/test split. Downloaded data, splits, caches, and run
outputs are local artifacts excluded by `.gitignore`. See
[docs/data_provenance.md](docs/data_provenance.md).

## Usage

```bash
# Offline tests
python -m pytest tests/

# Mock plumbing smoke test (use a disposable working copy)
python scripts/run_smoke_test.py

# Small mock experiment (does not produce scientific evidence)
python scripts/run_experiment.py --system A --split dev --limit 3 --backend mock

# Aggregate locally generated experiment ledger entries
python scripts/aggregate_results.py
```

Real experiments require downloaded benchmark files and a configured model
backend. The Ollama configuration is in `configs/models.yaml`. The separate
ZAI continuation config records the `glm-4-plus` backend deviation; the
local ZAI bridge process is not included in this source release. Never commit
API keys or credentials.

## Research status and limitations

The research campaign is incomplete. Only A, Oracle, and C-BM25 have 100/100
scored dev-100 results, all on the ZAI glm-4-plus backend. C-Dense is partial
and unjudged; C-Hybrid, C-RRF, D-BM25 rerank, and E-v2 were not started; E-v1
is only a one-example pilot. Human review is 0/144. Test-400 remains locked at
2/11 gate criteria. The backend deviation means these scores are not the
frozen Ollama/Qwen protocol and are not comparable to it. No claim of a
completed scientific study, causal superiority, or human-reviewed accuracy
is made.

Other practical limitations include external dataset downloads, external
model weights for dense/reranking paths, and a ZAI bridge process that is not
bundled. The public source release excludes research-evidence archives and
restricted materials.

## Tests

Run the offline suite with `python -m pytest tests/` after installing the
declared dependencies. The CI workflow in `.github/workflows/tests.yml`
performs the same dependency installation and test invocation. Do not treat
unit-test success as evidence that the scientific experiments are complete.

## Attribution and disclosure

The vendored LongMemEval scripts and copied prompt-template content are
attributed under the upstream MIT license in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The benchmark authors and
upstream project are not affiliated with or endorsing NOVA-Mem.

AI tools assisted parts of software development and documentation. This
assistance is disclosed for provenance and does not constitute experimental
evidence or scientific validation.

No current manuscript is included. The available historical draft was
excluded because it predates the S-36 status and is not publication-ready;
see [paper/README.md](paper/README.md).

## Citation

See [CITATION.cff](CITATION.cff). Individual author identifiers and a DOI
are not supplied because they have not been verified. Cite the LongMemEval
benchmark separately when using that benchmark.

## Project licensing

No repository-wide license is granted for NOVA-Mem project code in this release.
A root-level license is intentionally omitted until the project copyright
holder and the authority to grant redistribution permissions are confirmed.
Do not assume that public availability grants permission to reuse, modify, or
redistribute NOVA-Mem code. Third-party files retain their own notices; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
`third_party/longmemeval_official/LICENSE` for the separately licensed
LongMemEval upstream reference code.
