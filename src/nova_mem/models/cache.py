"""Generation cache: never repeat an identical inference call.

Key: sha256(model_id | backend | prompt | max_tokens | temperature | seed).
Values are the full GenerationResult dicts. Stored under results/cache/ as JSON
files (one per key) — inspectable and portable across profiles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..utils import sha256_text


class GenerationCache:
    def __init__(self, root: str | Path | None, enabled: bool = True):
        self.root = Path(root) if root else None
        self.enabled = enabled and self.root is not None
        if self.enabled and self.root:
            self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _key(
        self,
        model_id: str,
        backend: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        seed,
        context_length: int | None = None,
    ) -> str:
        key_parts = [model_id, backend, prompt, max_tokens, temperature, seed]
        if context_length is not None:
            key_parts.append(context_length)
        return sha256_text(json.dumps(key_parts, ensure_ascii=False))

    def get(
        self,
        model_id: str,
        backend: str,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        seed,
        context_length: int | None = None,
    ) -> Optional[dict]:
        if not self.enabled:
            return None
        path = self.root / (
            f"{self._key(model_id, backend, prompt, max_tokens, temperature, seed, context_length)}.json"
        )
        if not path.exists():
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(path.read_text(encoding="utf-8"))

    def put(
        self,
        model_id: str,
        backend: str,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        seed,
        result: dict,
        context_length: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        path = self.root / (
            f"{self._key(model_id, backend, prompt, max_tokens, temperature, seed, context_length)}.json"
        )
        if path.exists():  # append-only spirit: first write wins
            return
        path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
