"""Dataset statistics tests.

Regression context (2026-11-07): ``compute_stats`` originally counted abstention
questions via ``by_type.get("abstention")``, which is always 0 because the
official convention encodes abstention with the ``_abs`` question-id suffix while
``question_type`` keeps the base type. The fix counts ``Question.is_abstention``;
``data/stats_s_cleaned.json`` and ``data/README.md`` were regenerated from the
fixed counter (30 abstention questions in 500, matching the paper and
``tests/test_loader.py::test_real_file_spot_checks``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nova_mem.data.loader import Message, Question, Session
from nova_mem.data.stats import compute_stats

S_FILE = Path(__file__).resolve().parents[1] / "data" / "longmemeval_s_cleaned.json"


def _q(qid: str, qtype: str, n_sessions: int = 2, n_evidence: int = 1) -> Question:
    sessions = [
        Session(
            session_id=f"answer_x{i}" if i < n_evidence else f"sharegpt_x{i}",
            date="2023-05-01",
            messages=[Message("user", f"hello {i}"), Message("assistant", f"hi {i}")],
        )
        for i in range(n_sessions)
    ]
    return Question(
        question_id=qid,
        question_type=qtype,
        question="q?",
        answer="a",
        question_date="2023-06-01",
        sessions=sessions,
        answer_session_ids=[s.session_id for s in sessions[:n_evidence]],
    )


def test_stats_counts_abstention_by_id_suffix():
    qs = [
        _q("a1", "multi-session", n_sessions=3, n_evidence=2),
        _q("b2_abs", "single-session-user", n_sessions=2, n_evidence=1),
        _q("c3", "temporal-reasoning", n_sessions=4, n_evidence=3),
    ]
    stats = compute_stats(qs)
    assert stats.n_questions == 3
    # abstention via _abs suffix; literal 'abstention' type never occurs
    assert stats.n_abstention == 1
    # base types preserved inside by_type (official aggregation semantics)
    assert stats.by_type == {
        "multi-session": 1,
        "single-session-user": 1,
        "temporal-reasoning": 1,
    }
    # evidence stats over answerable questions only (n_evidence 2 and 3)
    assert stats.evidence_sessions_per_question == {"min": 2, "mean": 2.5, "max": 3}


def test_stats_zero_abstention_when_none_marked():
    stats = compute_stats([_q("x1", "multi-session"), _q("x2", "multi-session")])
    assert stats.n_abstention == 0


@pytest.mark.skipif(not S_FILE.exists(), reason="S file not downloaded")
def test_stats_real_file_regression():
    """The committed dataset must report exactly 30 abstention questions."""
    from nova_mem.data import load_benchmark

    stats = compute_stats(load_benchmark(S_FILE))
    assert stats.n_abstention == 30
    assert stats.by_type == {
        "knowledge-update": 78,
        "multi-session": 133,
        "single-session-assistant": 56,
        "single-session-preference": 30,
        "single-session-user": 70,
        "temporal-reasoning": 133,
    }
    assert stats.n_questions == 500
