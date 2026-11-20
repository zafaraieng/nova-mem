from __future__ import annotations

import pytest

from nova_mem.data import Message, Question, Session, dataset_comparison_fingerprint


def _question(sessions, date="2023-02-01"):
    return Question(
        question_id="q1",
        question_type="single-session-user",
        question="Where did I move?",
        answer="York",
        question_date=date,
        sessions=sessions,
        answer_session_ids=["answer_s1"],
    )


def test_fingerprint_ignores_non_evidence_filler_but_requires_shared_metadata():
    evidence = Session("answer_s1", "2023-01-01", [Message("user", "I moved to York.")])
    cleaned = _question(
        [
            evidence,
            Session("filler", "2023-01-02", [Message("user", "Unrelated conversation.")]),
        ]
    )
    oracle = _question([evidence])

    assert dataset_comparison_fingerprint([cleaned]) == dataset_comparison_fingerprint([oracle])


def test_fingerprint_changes_when_question_or_evidence_dates_differ():
    evidence = Session("answer_s1", "2023-01-01", [Message("user", "I moved to York.")])
    baseline = dataset_comparison_fingerprint([_question([evidence])])
    changed_question_date = dataset_comparison_fingerprint([_question([evidence], "2023-02-02")])
    changed_evidence_date = dataset_comparison_fingerprint(
        [_question([Session("answer_s1", "2023-01-03", evidence.messages)])]
    )

    assert baseline != changed_question_date
    assert baseline != changed_evidence_date


def test_fingerprint_rejects_missing_or_duplicated_evidence_sessions():
    q = _question([])
    with pytest.raises(ValueError, match="Missing evidence"):
        dataset_comparison_fingerprint([q])
    dup = _question(
        [
            Session("answer_s1", "2023-01-01", []),
            Session("answer_s1", "2023-01-02", []),
        ]
    )
    with pytest.raises(ValueError, match="Duplicate evidence"):
        dataset_comparison_fingerprint([dup])
    duplicate_label = _question(
        [Session("answer_s1", "2023-01-01", [Message("user", "I moved to York.")])]
    )
    duplicate_label.answer_session_ids.append("answer_s1")
    with pytest.raises(ValueError, match="Duplicate evidence label"):
        dataset_comparison_fingerprint([duplicate_label])
