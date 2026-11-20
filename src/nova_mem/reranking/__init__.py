"""Cross-encoder reranking (system D).

Primary: cross-encoder/ms-marco-MiniLM-L-6-v2 (~22M params, CPU-friendly).
A deterministic mock is provided for tests. Reranking scores (query, document)
pairs and re-sorts; NOVA-Mem measures the added latency separately.
"""

from __future__ import annotations

from typing import Protocol

from ..retrieval import Document, Hit


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, docs: list[Document], top_k: int) -> tuple[list[Hit], float]:
        """Return (reranked hits, elapsed_seconds)."""


class CrossEncoderReranker:
    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length: int = 512
    ):
        from sentence_transformers import CrossEncoder  # lazy import

        self._model = CrossEncoder(model_name, max_length=max_length)
        self.name = model_name

    def rerank(self, query: str, docs: list[Document], top_k: int) -> tuple[list[Hit], float]:
        import time as _time

        if not docs:
            return [], 0.0
        start = _time.perf_counter()
        pairs = [(query, d.text) for d in docs]
        scores = self._model.predict(pairs, show_progress_bar=False)
        elapsed = _time.perf_counter() - start
        order = sorted(range(len(docs)), key=lambda i: -float(scores[i]))[:top_k]
        hits = [
            Hit(
                corpus_id=docs[i].corpus_id,
                session_id=docs[i].session_id,
                score=float(scores[i]),
                rank=r + 1,
                date=docs[i].date,
            )
            for r, i in enumerate(order)
        ]
        return hits, elapsed


class MockReranker:
    """Deterministic: reverses the input order. Tests only."""

    def __init__(self):
        self.name = "mock-reranker"

    def rerank(self, query: str, docs: list[Document], top_k: int) -> tuple[list[Hit], float]:
        hits = []
        for r, d in enumerate(reversed(docs)):
            hits.append(
                Hit(
                    corpus_id=d.corpus_id,
                    session_id=d.session_id,
                    score=-float(r),
                    rank=r + 1,
                    date=d.date,
                )
            )
        return hits[:top_k], 0.001
