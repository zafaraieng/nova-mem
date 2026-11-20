"""Retrieval metrics against official evidence labels.

Official definitions (repo + paper §3.3): session-level and turn-level
``Recall@k`` / ``NDCG@k`` computed against the human-annotated answer
locations; abstention questions are excluded from retrieval metrics
(no meaningful evidence set).

NOVA-Mem addition (clearly labeled): ``evidence_density@k`` — the fraction of
retrieved sessions that are evidence sessions (a precision-style signal;
strict precision is ill-defined because multiple sessions may legitimately
support an answer).
"""

from __future__ import annotations

import math
from collections import Counter


def _sessions_of(corpus_id: str) -> str:
    """Turn ids look like ``<session_id>_<n>`` (official 1-offset); session ids
    never end in ``_<int>`` in this dataset (verified: ids like sharegpt_xxx_0
    end in digits too — so we resolve turns via the trace's session_id when
    available, and only fall back to string surgery otherwise)."""
    return corpus_id


def session_recall_at_k(
    trace: list[dict], evidence_session_ids: list[str], k: int | None = None
) -> float | None:
    """Official session-level Recall@k.

    ``trace`` entries: {rank, session_id, ...} in retrieval order.
    ``evidence_session_ids``: official answer_session_ids (noans_ stripped).
    Returns None when there is no evidence set (abstention questions).
    """
    if not evidence_session_ids:
        return None
    items = trace[:k] if k else trace
    retrieved = {t["session_id"] for t in items}
    evidence = set(evidence_session_ids)
    if not evidence:
        return None
    return len(retrieved & evidence) / len(evidence)


def evidence_density_at_k(
    trace: list[dict], evidence_session_ids: list[str], k: int | None = None
) -> float | None:
    """NOVA-Mem addition: fraction of top-k retrieved sessions that are evidence."""
    if not evidence_session_ids:
        return None
    items = trace[:k] if k else trace
    if not items:
        return 0.0
    evidence = set(evidence_session_ids)
    return sum(1 for t in items if t["session_id"] in evidence) / len(items)


def ndcg_at_k(
    trace: list[dict], evidence_session_ids: list[str], k: int | None = None
) -> float | None:
    """Binary-relevance NDCG@k on sessions (official metric family)."""
    if not evidence_session_ids:
        return None
    items = trace[:k] if k else trace
    evidence = set(evidence_session_ids)
    dcg = 0.0
    for t in items:
        if t["session_id"] in evidence:
            dcg += 1.0 / math.log2(t["rank"] + 1)
    n_rel = min(len(evidence), len(items))
    if n_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def turn_recall_at_k(
    trace: list[dict], evidence_turn_ids: set[str], k: int | None = None
) -> float | None:
    """Turn-level recall when the trace is over turn corpus ids."""
    if not evidence_turn_ids:
        return None
    items = trace[:k] if k else trace
    retrieved = {t["corpus_id"] for t in items}
    return len(retrieved & set(evidence_turn_ids)) / len(set(evidence_turn_ids))


def retrieval_metrics_for_question(
    trace: list[dict], evidence_session_ids: list[str], ks=(5, 10)
) -> dict:
    out: dict = {}
    for k in ks:
        out[f"session_recall@{k}"] = session_recall_at_k(trace, evidence_session_ids, k)
        out[f"evidence_density@{k}"] = evidence_density_at_k(trace, evidence_session_ids, k)
        out[f"ndcg@{k}"] = ndcg_at_k(trace, evidence_session_ids, k)
    return out


def aggregate_retrieval_metrics(per_question: list[dict]) -> dict:
    """Mean over questions where the metric is defined (abstention excluded)."""
    keys: Counter = Counter()
    sums: dict[str, float] = {}
    for row in per_question:
        for k, v in row.items():
            if v is None:
                continue
            keys[k] += 1
            sums[k] = sums.get(k, 0.0) + v
    return {k: {"mean": round(sums[k] / keys[k], 4), "n": keys[k]} for k in sorted(sums)}
