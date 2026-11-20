"""ZAI cloud chat client — NOVA-Mem additive backend for sandbox execution.

Talks to the local ``scripts/zai_server.mjs`` Node bridge (127.0.0.1:11435),
which holds one persistent ``z-ai-web-dev-sdk`` instance. The underlying
served model is glm-4-plus (verified live; recorded in the ledger per run).

This backend was introduced for the sandbox continuation because the pinned
Ollama models (Qwen2.5-7B / Llama3.2-3B Q4_K_M, ~6.7 GB weights) cannot reside
in the 4 GB cgroup of this environment. Every run that uses it carries the
``modified-config: zai-glm4-plus-backend`` deviation tag in the ledger, and its
results are reported separately from all historical Qwen/Ollama runs. It is
NEVER mixed with, or presented as, the frozen protocol's Ollama conditions.
"""

from __future__ import annotations

import os
import time

import requests

from .client import GenerationResult, InferenceClient

DEFAULT_BASE_URL = "http://127.0.0.1:11435"


class ZAIChatClient(InferenceClient):
    """Persistent-server chat client with Ollama-compatible timings."""

    def __init__(
        self,
        model_id: str = "glm-4-plus",
        base_url: str | None = None,
        timeout_s: float = 900.0,
        max_retries: int = 250,
    ):
        self.model_id = model_id
        self.base_url = (base_url or os.environ.get("ZAI_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.timeout_s = timeout_s
        # 250 retries × ~70 s mean backoff ≈ 4.9 h: enough to ride out any
        # observed rolling-quota window instead of crashing the run.
        self.max_retries = max_retries
        self.backend = "zai"
        self.num_ctx = None

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 500,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> GenerationResult:
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                start = time.perf_counter()
                resp = requests.post(
                    f"{self.base_url}/generate", json=payload, timeout=self.timeout_s
                )
                if resp.status_code == 451:
                    # Deterministic provider content-filter rejection (code 1301).
                    # Retrying cannot help; surface immediately so the runner can
                    # record the question as filter-blocked.
                    raise ZAIContentFilteredError(resp.text[:300])
                resp.raise_for_status()
                data = resp.json()
                latency = time.perf_counter() - start
                return GenerationResult(
                    text=(data.get("text") or "").strip(),
                    prompt_tokens=int(data.get("prompt_tokens") or 0),
                    completion_tokens=int(data.get("completion_tokens") or 0),
                    latency_s=latency,
                    model_id=data.get("model") or self.model_id,
                    backend=self.backend,
                    extra={
                        "request_id": data.get("request_id"),
                        "server_timings_s": {
                            "total": (
                                data.get("latency_ms", 0) / 1000.0
                                if data.get("latency_ms") is not None
                                else None
                            ),
                            "prompt_eval": None,
                            "generation_eval": None,
                        },
                    },
                )
            except ZAIContentFilteredError:
                raise
            except Exception as e:  # noqa: BLE001 — transient cloud errors; retry with backoff
                last_err = e
                # Provider quota (429) recovers on a rolling-window basis;
                # patient fixed backoff rides out the window instead of failing.
                retriable = "502" in repr(e) or "429" in repr(e) or "upstream" in repr(e)
                if retriable:
                    time.sleep(45 if attempt < 20 else 90)
                else:
                    time.sleep(min(2**attempt, 30))
        raise RuntimeError(
            f"zai backend failed after {self.max_retries} attempts: {last_err!r}"
        ) from last_err


class ZAIContentFilteredError(RuntimeError):
    """The provider's content filter deterministically rejected this prompt."""
