"""Metrics tests: retrieval metrics, official aggregation, EM/F1."""

from __future__ import annotations

from nova_mem.evaluation import exact_match, nonofficial_metrics, official_qa_metrics, token_f1
from nova_mem.metrics import (
    aggregate_retrieval_metrics,
    evidence_density_at_k,
    ndcg_at_k,
    retrieval_metrics_for_question,
    session_recall_at_k,
)


def _trace(*session_ids):
    return [
        {
            "rank": i + 1,
            "corpus_id": s,
            "session_id": s,
            "score": 1.0 / (i + 1),
            "is_evidence": False,
        }
        for i, s in enumerate(session_ids)
    ]


def test_session_recall():
    trace = _trace("a", "b", "c", "d", "e")
    assert session_recall_at_k(trace, ["a", "c"], k=5) == 1.0
    assert session_recall_at_k(trace, ["a", "z"], k=5) == 0.5
    assert session_recall_at_k(trace, ["e"], k=2) == 0.0  # evidence at rank 5, k=2
    assert session_recall_at_k(trace, ["e"], k=5) == 1.0
    assert session_recall_at_k(trace, [], k=5) is None  # abstention: undefined


def test_evidence_density():
    trace = _trace("a", "x", "c", "x", "x")
    assert evidence_density_at_k(trace, ["a", "c"], k=5) == 2 / 5
    assert evidence_density_at_k(trace, ["a", "c"], k=2) == 1 / 2
    assert evidence_density_at_k(trace, [], k=5) is None


def test_ndcg():
    trace = _trace("a", "x", "c")
    # rel at ranks 1 and 3: DCG = 1 + 1/log2(4) = 1.5 ; IDCG (2 rel) = 1 + 1/log2(3)
    import math

    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert abs(ndcg_at_k(trace, ["a", "c"], k=3) - expected) < 1e-9


def test_aggregate_retrieval():
    rows = [
        retrieval_metrics_for_question(_trace("a", "b"), ["a"], ks=(5,)),
        retrieval_metrics_for_question(_trace("x", "b"), ["a"], ks=(5,)),
        retrieval_metrics_for_question(_trace("x", "b"), [], ks=(5,)),  # abstention -> None
    ]
    agg = aggregate_retrieval_metrics(rows)
    assert agg["session_recall@5"] == {"mean": 0.5, "n": 2}


def test_official_qa_metrics():
    """Official semantics: _abs rows count in their base type AND separately."""
    questions = [
        {"question_id": "q1", "question_type": "single-session-user"},
        {"question_id": "q2", "question_type": "single-session-user"},
        {"question_id": "q3_abs", "question_type": "temporal-reasoning"},
    ]
    records = [
        {"question_id": "q1", "autoeval_label": {"label": True}},
        {"question_id": "q2", "autoeval_label": {"label": False}},
        {"question_id": "q3_abs", "autoeval_label": {"label": True}},
    ]
    m = official_qa_metrics(records, questions)
    assert m["overall_accuracy"] == round(2 / 3, 4)  # _abs included (official)
    assert m["per_type"]["single-session-user"] == {"accuracy": 0.5, "n": 2}
    assert m["per_type"]["temporal-reasoning"] == {"accuracy": 1.0, "n": 1}
    assert m["abstention"] == {"accuracy": 1.0, "n": 1}
    assert m["task_averaged_accuracy"] == round((0.5 + 1.0) / 2, 4)


def test_em_and_f1():
    assert exact_match("The answer is 42.", "The answer is 42") == 1  # punctuation dropped
    assert exact_match("42", "43") == 0
    assert exact_match("The answer is 42.", "42") == 0  # EM requires full normalized equality
    assert abs(token_f1("Paris, France", "paris") - 2 / 3) < 1e-9  # P=.5, R=1
    assert token_f1("completely different", "paris") == 0.0


def test_nonofficial_labeled():
    questions = [{"question_id": "q1", "question_type": "t", "answer": "42"}]
    records = [
        {"question_id": "q1", "hypothesis": "The answer is 42.", "autoeval_label": {"label": True}}
    ]
    m = nonofficial_metrics(records, questions)
    assert "NON-OFFICIAL" in m["note"]
    assert m["exact_match"] == 0.0
    assert m["token_f1"] == 0.4  # pred {the,answer,is,42} vs ans {42}: P=.25, R=1
