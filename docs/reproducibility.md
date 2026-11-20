# Reproducibility guide

## Environment

- Python 3.10 or newer.
- Install the declared runtime and development dependencies from the
  repository root:

  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt -r requirements-dev.txt
  python -m pip install -e .
  ```

- `rank-bm25` is a declared dependency of the BM25 retrieval path. Do not
  interpret tests as fully exercised if it is missing.
- Dense retrieval and cross-encoder components can require separately
  downloaded model weights and additional compute resources.

## Data

The benchmark data is not bundled. Follow [data provenance](data_provenance.md)
and run `python scripts/prepare_data.py`. Do not regenerate a frozen split or
change benchmark inputs when attempting a like-for-like reproduction.

## Offline software checks

```bash
python -m pytest tests/
```

The CI workflow runs the same test suite after installing the declared
requirements. Dataset-dependent checks may skip when the benchmark files are
not available. Record skips and failures; do not silently remove tests or
convert missing dependencies into successful test claims.

A mock smoke test is available:

```bash
python scripts/run_smoke_test.py
```

Run this only in a disposable working copy because it creates local run
artifacts. Mock outputs are plumbing checks, not scientific evidence.

## Research-status limitation

Consult [RESEARCH_STATUS.md](RESEARCH_STATUS.md) before attempting to
reproduce or citing results. S-36 reports a ZAI `glm-4-plus` backend deviation
from the frozen Ollama/Qwen configuration. Those results are not comparable
to the frozen protocol. The study remains incomplete and Test-400 remains
locked.
