"""Retrieval layer: corpora, BM25, dense embeddings (flat IP), reranking.

Corpus conventions follow the official repo: session-level corpus ids are the
dataset's ``session_id``; turn-level ids are ``<session_id>_<turn#>`` with a
1-offset. ``user_only=True`` keeps only user-role utterances in the indexed
text (the paper: "we only keep the user-side utterances" when sessions/rounds
are used as the key).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

from ..data.loader import Question, Session


@dataclass
class Document:
    corpus_id: str
    text: str
    date: str
    session_id: str  # owning session (for turn docs, the parent session)
    messages: list = field(default_factory=list)  # raw {role, content} dicts


@dataclass
class Hit:
    corpus_id: str
    session_id: str
    score: float
    rank: int
    date: str


def build_session_corpus(question: Question, user_only: bool = True) -> list[Document]:
    docs = []
    for s in question.sessions:
        text = s.text(user_only=user_only)
        docs.append(
            Document(
                corpus_id=s.session_id,
                text=text,
                date=s.date,
                session_id=s.session_id,
                messages=[{"role": m.role, "content": m.content} for m in s.messages],
            )
        )
    return docs


def build_turn_corpus(question: Question, user_only: bool = True) -> list[Document]:
    docs = []
    for s in question.sessions:
        for i, m in enumerate(s.messages):
            if user_only and m.role != "user":
                continue
            docs.append(
                Document(
                    corpus_id=f"{s.session_id}_{i + 1}",
                    text=f"{m.role}: {m.content}",
                    date=s.date,
                    session_id=s.session_id,
                    messages=[{"role": m.role, "content": m.content}],
                )
            )
    return docs


class Retriever(Protocol):
    name: str

    def build(self, docs: list[Document]) -> None: ...
    def search(self, query: str, top_k: int) -> list[Hit]: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Lexical retrieval (rank-bm25 Okapi). CPU-cheap; the plan's lexical arm."""

    def __init__(self):
        self.name = "bm25"
        self._docs: list[Document] = []
        self._bm25 = None

    def build(self, docs: list[Document]) -> None:
        from rank_bm25 import BM25Okapi

        self._docs = list(docs)
        tokenized = [_tokenize(d.text) for d in docs]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int) -> list[Hit]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        hits = []
        for rank, i in enumerate(order):
            d = self._docs[i]
            hits.append(
                Hit(
                    corpus_id=d.corpus_id,
                    session_id=d.session_id,
                    score=float(scores[i]),
                    rank=rank + 1,
                    date=d.date,
                )
            )
        return hits


class _HashEmbedder:
    """Deterministic feature-hashing embedder for tests (no model download).

    Not a semantic embedder — plumbing tests only. Real runs use
    ``SentenceTransformerEmbedder``.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = "hash-embedder"

    def encode(self, texts: list[str]) -> "list":
        import hashlib

        import numpy as np

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _tokenize(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder:
    """Dense embedder (default BAAI/bge-small-en-v1.5, 384d, normalized)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        self._st = True

    def encode(self, texts: list[str]):
        import numpy as np

        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        emb = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return emb


class DenseRetriever:
    """Flat inner-product (cosine, embeddings normalized) retriever.

    Uses numpy; FAISS is unnecessary at this scale (<=100k docs, 384d) and is
    deliberately optional to keep the dependency surface small on CPU.
    """

    def __init__(self, embedder):
        self.name = f"dense:{getattr(embedder, 'name', 'embedder')}"
        self._embedder = embedder
        self._doc_matrix = None
        self._docs: list[Document] = []

    def build(self, docs: list[Document]) -> None:
        self._docs = list(docs)
        if not docs:
            self._doc_matrix = None
            return
        embs = self._embedder.encode([d.text for d in docs])
        self._doc_matrix = embs  # (n, dim), normalized rows

    def search(self, query: str, top_k: int) -> list[Hit]:
        if self._doc_matrix is None or not self._docs:
            return []
        import numpy as np

        q = self._embedder.encode([query])[0]
        scores = self._doc_matrix @ q
        k = min(top_k, len(scores))
        order = np.argsort(-scores)[:k]
        hits = []
        for rank, i in enumerate(order):
            d = self._docs[int(i)]
            hits.append(
                Hit(
                    corpus_id=d.corpus_id,
                    session_id=d.session_id,
                    score=float(scores[i]),
                    rank=rank + 1,
                    date=d.date,
                )
            )
        return hits


def _unique_ranked_hits(hits: list[Hit]) -> dict[str, Hit]:
    unique = {}
    for hit in sorted(hits, key=lambda item: item.rank):
        unique.setdefault(hit.corpus_id, hit)
    return unique


def _minmax_scores(hits: dict[str, Hit]) -> dict[str, float]:
    if not hits:
        return {}
    scores = [hit.score for hit in hits.values()]
    low, high = min(scores), max(scores)
    if high == low:
        value = 1.0 if high > 0 else 0.0
        return {corpus_id: value for corpus_id in hits}
    return {corpus_id: (hit.score - low) / (high - low) for corpus_id, hit in hits.items()}


class HybridRetriever:
    """Equal-weight min-max score fusion over BM25 and dense top-N rankings."""

    def __init__(self, embedder, candidate_pool_size: int = 25):
        if candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be positive")
        self.name = f"hybrid:bm25+dense:{getattr(embedder, 'name', 'embedder')}"
        self.candidate_pool_size = candidate_pool_size
        self._bm25 = BM25Retriever()
        self._dense = DenseRetriever(embedder)
        self._canonical_docs: dict[str, Document] = {}

    def build(self, docs: list[Document]) -> None:
        self._bm25.build(docs)
        self._dense.build(docs)
        self._canonical_docs = {doc.corpus_id: doc for doc in docs}

    def search(self, query: str, top_k: int) -> list[Hit]:
        if top_k <= 0:
            return []
        bm25_hits = _unique_ranked_hits(self._bm25.search(query, self.candidate_pool_size))
        dense_hits = _unique_ranked_hits(self._dense.search(query, self.candidate_pool_size))
        bm25_scores = _minmax_scores(bm25_hits)
        dense_scores = _minmax_scores(dense_hits)
        candidate_order = list(bm25_hits)
        candidate_order.extend(corpus_id for corpus_id in dense_hits if corpus_id not in bm25_hits)
        insertion_order = {corpus_id: index for index, corpus_id in enumerate(candidate_order)}
        fused_scores = {
            corpus_id: 0.5 * bm25_scores.get(corpus_id, 0.0)
            + 0.5 * dense_scores.get(corpus_id, 0.0)
            for corpus_id in candidate_order
        }
        ranked_ids = sorted(
            candidate_order,
            key=lambda corpus_id: (-fused_scores[corpus_id], insertion_order[corpus_id]),
        )[:top_k]
        return [
            Hit(
                corpus_id=corpus_id,
                session_id=self._canonical_docs[corpus_id].session_id,
                score=fused_scores[corpus_id],
                rank=rank,
                date=self._canonical_docs[corpus_id].date,
            )
            for rank, corpus_id in enumerate(ranked_ids, start=1)
        ]


class ReciprocalRankFusionRetriever:
    """Equal-weight Reciprocal Rank Fusion over BM25 and dense top-N rankings."""

    def __init__(self, embedder, candidate_pool_size: int = 25, rrf_k: int = 60):
        if candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be positive")
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        self.name = f"rrf:bm25+dense:{getattr(embedder, 'name', 'embedder')}"
        self.candidate_pool_size = candidate_pool_size
        self.rrf_k = rrf_k
        self._bm25 = BM25Retriever()
        self._dense = DenseRetriever(embedder)
        self._canonical_docs: dict[str, Document] = {}

    def build(self, docs: list[Document]) -> None:
        self._bm25.build(docs)
        self._dense.build(docs)
        self._canonical_docs = {doc.corpus_id: doc for doc in docs}

    def search(self, query: str, top_k: int) -> list[Hit]:
        if top_k <= 0:
            return []
        bm25_hits = _unique_ranked_hits(self._bm25.search(query, self.candidate_pool_size))
        dense_hits = _unique_ranked_hits(self._dense.search(query, self.candidate_pool_size))
        candidate_order = list(bm25_hits)
        candidate_order.extend(corpus_id for corpus_id in dense_hits if corpus_id not in bm25_hits)
        fused_scores = {
            corpus_id: (
                (1.0 / (self.rrf_k + bm25_hits[corpus_id].rank) if corpus_id in bm25_hits else 0.0)
                + (
                    1.0 / (self.rrf_k + dense_hits[corpus_id].rank)
                    if corpus_id in dense_hits
                    else 0.0
                )
            )
            for corpus_id in candidate_order
        }
        insertion_order = {corpus_id: index for index, corpus_id in enumerate(candidate_order)}
        ranked_ids = sorted(
            candidate_order,
            key=lambda corpus_id: (-fused_scores[corpus_id], insertion_order[corpus_id]),
        )[:top_k]
        return [
            Hit(
                corpus_id=corpus_id,
                session_id=self._canonical_docs[corpus_id].session_id,
                score=fused_scores[corpus_id],
                rank=rank,
                date=self._canonical_docs[corpus_id].date,
            )
            for rank, corpus_id in enumerate(ranked_ids, start=1)
        ]


class DeterministicRetriever:
    """Returns documents in corpus order — mock for plumbing tests only."""

    def __init__(self):
        self.name = "deterministic"
        self._docs: list[Document] = []

    def build(self, docs: list[Document]) -> None:
        self._docs = list(docs)

    def search(self, query: str, top_k: int) -> list[Hit]:
        hits = []
        for rank, d in enumerate(self._docs[:top_k]):
            hits.append(
                Hit(
                    corpus_id=d.corpus_id,
                    session_id=d.session_id,
                    score=-float(rank),
                    rank=rank + 1,
                    date=d.date,
                )
            )
        return hits


def build_retriever(
    kind: str,
    embedder=None,
    *,
    candidate_pool_size: int = 25,
    rrf_k: int = 60,
) -> Retriever:
    if kind == "bm25":
        return BM25Retriever()
    if kind == "dense":
        return DenseRetriever(embedder or _HashEmbedder())
    if kind == "hybrid":
        return HybridRetriever(embedder or _HashEmbedder(), candidate_pool_size=candidate_pool_size)
    if kind == "rrf":
        return ReciprocalRankFusionRetriever(
            embedder or _HashEmbedder(),
            candidate_pool_size=candidate_pool_size,
            rrf_k=rrf_k,
        )
    if kind == "deterministic":
        return DeterministicRetriever()
    raise ValueError(f"Unknown retriever kind {kind!r}")
