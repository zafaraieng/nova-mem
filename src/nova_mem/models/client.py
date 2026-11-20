"""Inference clients: Ollama (readers/extractors), OpenAI-compatible (official
judge / vLLM), and a deterministic mock for tests and smoke runs.

All real clients are temperature-0 by default (official protocol). API keys are
read from environment variables only and never stored in configs or the ledger.
"""

from __future__ import annotations

import os
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Protocol

import requests

from ..utils import get_logger, sha256_text

logger = get_logger(__name__)


def _ollama_duration_s(data: dict, key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f"Ollama response field {key!r} must be a non-negative duration in nanoseconds"
        )
    return float(value) / 1_000_000_000


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model_id: str
    backend: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class InferenceClient(Protocol):
    model_id: str
    backend: str

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 500,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> GenerationResult: ...


class OllamaClient:
    """Ollama chat API client (single-shot, non-streaming)."""

    def __init__(
        self,
        model_id: str,
        base_url: str | None = None,
        timeout_s: float = 600.0,
        num_ctx: int | None = None,
    ):
        if num_ctx is not None and (
            not isinstance(num_ctx, int) or isinstance(num_ctx, bool) or num_ctx <= 0
        ):
            raise ValueError("num_ctx must be a positive integer")
        self.model_id = model_id
        self.base_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self.timeout_s = timeout_s
        self.num_ctx = num_ctx
        self.backend = "ollama"

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 500,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if self.num_ctx is not None:
            payload["options"]["num_ctx"] = self.num_ctx
        if seed is not None:
            payload["options"]["seed"] = seed
        start = time.perf_counter()
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        latency = time.perf_counter() - start
        return GenerationResult(
            text=data.get("message", {}).get("content", "").strip(),
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
            latency_s=latency,
            model_id=self.model_id,
            backend=self.backend,
            extra={
                "num_ctx": data.get("context"),
                "num_ctx_requested": self.num_ctx,
                "server_timings_s": {
                    "total": _ollama_duration_s(data, "total_duration"),
                    "prompt_eval": _ollama_duration_s(data, "prompt_eval_duration"),
                    "generation_eval": _ollama_duration_s(data, "eval_duration"),
                },
            },
        )


class OpenAICompatClient:
    """OpenAI-compatible chat client.

    Used for (a) the official ``gpt-4o-2024-08-06`` judge when an API key is
    provided via env var, and (b) vLLM servers exposing the OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: float = 300.0,
        max_retries: int = 5,
    ):
        self.model_id = model_id
        self.base_url = base_url  # None -> default OpenAI
        api_key = (
            os.environ.get(api_key_env)
            if base_url is None
            else os.environ.get(api_key_env, "EMPTY")
        )
        self._api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backend = "openai-compat"

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 500,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        import openai  # lazy import; only needed for API/vLLM paths

        client = openai.OpenAI(
            api_key=self._api_key, base_url=self.base_url, timeout=self.timeout_s
        )
        kwargs = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "n": 1,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
        start = time.perf_counter()
        last_err: Exception | None = None
        for _ in range(self.max_retries):
            try:
                completion = client.chat.completions.create(**kwargs)
                latency = time.perf_counter() - start
                usage = completion.usage
                return GenerationResult(
                    text=completion.choices[0].message.content.strip(),
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    latency_s=latency,
                    model_id=self.model_id,
                    backend=self.backend,
                )
            except Exception as e:  # noqa: BLE001 - retry any API error, like official backoff
                last_err = e
                logger.warning("openai-compat retry after error: %r", e)
                time.sleep(5)
        raise RuntimeError(f"openai-compat failed after {self.max_retries} retries") from last_err


class MockLLMClient:
    """Deterministic mock for tests and smoke runs — NEVER for scientific numbers.

    ``respond`` maps (prompt) -> text. A default echo-ish function is provided.
    """

    def __init__(self, model_id: str = "mock-llm", respond: Callable[[str], str] | None = None):
        self.model_id = model_id
        self.backend = "mock"
        self._respond = respond or (lambda p: f"MOCK-RESPONSE::{sha256_text(p)[:12]}")
        self.calls: list[str] = []

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 500,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        self.calls.append(prompt)
        text = self._respond(prompt)
        n_prompt_tokens = max(1, len(prompt) // 4)
        return GenerationResult(
            text=text,
            prompt_tokens=n_prompt_tokens,
            completion_tokens=max(1, len(text) // 4),
            latency_s=0.001,
            model_id=self.model_id,
            backend=self.backend,
        )


def client_from_config(cfg: dict) -> InferenceClient:
    """Build a client from a models.yaml entry {backend, model_id, base_url?, ...}."""
    backend = cfg["backend"]
    if backend == "ollama":
        return OllamaClient(
            cfg["model_id"],
            base_url=cfg.get("base_url"),
            num_ctx=cfg.get("num_ctx"),
        )
    if backend == "openai-compat":
        return OpenAICompatClient(cfg["model_id"], base_url=cfg.get("base_url"))
    if backend == "zai":
        from .zai_client import ZAIChatClient

        return ZAIChatClient(
            model_id=cfg.get("model_id", "glm-4-plus"),
            base_url=cfg.get("base_url"),
        )
    if backend == "mock":
        return MockLLMClient(model_id=cfg.get("model_id", "mock-llm"))
    raise ValueError(f"Unknown backend {backend!r}")
