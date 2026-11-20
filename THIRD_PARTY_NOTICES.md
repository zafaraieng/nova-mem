# Third-party notices

## LongMemEval upstream reference code — MIT

`third_party/longmemeval_official/` contains the following files copied from
[`xiaowu0162/LongMemEval`](https://github.com/xiaowu0162/LongMemEval):

- `run_generation.py` (upstream `src/generation/run_generation.py`)
- `evaluate_qa.py` (upstream `src/evaluation/evaluate_qa.py`)
- `print_qa_metrics.py` (upstream `src/evaluation/print_qa_metrics.py`)

The complete upstream MIT license and its copyright notice for Di Wu are
retained in `third_party/longmemeval_official/LICENSE`. During this audit, the
Git blob identifiers for all three files and the license matched the
corresponding upstream `main`-branch files. These files are third-party work,
not original NOVA-Mem code. Preserve their notices when redistributing them.

## Prompt-template provenance

`src/nova_mem/memory/prompts.py` contains prompt-template material based on the
LongMemEval official source. The upstream MIT attribution above applies to
those copied portions. The repository's prompt-parity tests check alignment
with the retained upstream reference implementation.

## Dataset

The cleaned LongMemEval benchmark files are not included in this repository.
See [data provenance](docs/data_provenance.md) and consult the upstream
benchmark repository and dataset card for terms governing data obtained from
those sources.

## Runtime dependencies

Python packages are declared in `requirements.txt`, `requirements-dev.txt`,
and `pyproject.toml`; they are installed by users and are not vendored here.
Each dependency remains subject to its own license. No endorsement by the
LongMemEval authors or maintainers is implied.
