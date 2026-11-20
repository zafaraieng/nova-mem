# Prompt registry v1

The authoritative, versioned prompt registry lives in code:
`src/nova_mem/memory/prompts.py` (constant `REGISTRY_VERSION = "v1"`).

It contains the official LongMemEval templates verbatim (judge prompts from
`evaluate_qa.py`; answer/CoN/history-rendering from `run_generation.py`) plus
NOVA-Mem's own extraction prompt (clearly marked as non-official).

Parity with the vendored official code is enforced by
`tests/test_prompt_parity.py`. Changes require a registry version bump (v2, ...).
