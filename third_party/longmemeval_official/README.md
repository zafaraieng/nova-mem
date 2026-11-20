# Vendored: official LongMemEval code

The three `.py` files in this directory are **vendored unmodified copies**
from the official LongMemEval repository:

- **Upstream:** https://github.com/xiaowu0162/LongMemEval
- **Upstream authors:** Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang,
  Kai-Wei Chang, Dong Yu (ICLR 2025)
- **Upstream license:** MIT — full text in `LICENSE` (Copyright (c) 2024
  Di Wu)
- **Vendored:** 2026-11-07, from the `main` branch
- **Files (upstream paths):**
  - `run_generation.py` ← `src/generation/run_generation.py`
  - `evaluate_qa.py` ← `src/evaluation/evaluate_qa.py`
  - `print_qa_metrics.py` ← `src/evaluation/print_qa_metrics.py`
- **Verified:** 2026-11-20 — byte-identical to the upstream `main` branch
  at verification time (diffed against
  `raw.githubusercontent.com/xiaowu0162/LongMemEval/main/...`).

## Why they are here

1. **Prompt parity.** `tests/test_prompt_parity.py` re-extracts the official
   prompt templates from these files via `ast` and asserts NOVA-Mem's prompt
   registry (`src/nova_mem/memory/prompts.py`) matches them verbatim. The
   vendored copies are the reference for that test.
2. **Protocol reference.** They document the exact official generation and
   evaluation semantics that NOVA-Mem's own pipeline reimplements.

## Attribution

These files are **not** authored by the NOVA-Mem project, and nothing in
NOVA-Mem should imply otherwise. If the upstream repository changes its
license or these files, re-vendor and re-audit; see also
`THIRD_PARTY_NOTICES.md` at the repository root.
