from __future__ import annotations

import pytest

from scripts.align_oracle_metadata import align_oracle_records, session_content_hash


def _session_hash():
    return session_content_hash([{"role": "user", "content": "I moved to York."}])


def _records():
    messages = [{"role": "user", "content": "I moved to York."}]
    oracle = [
        {
            "question_id": "q1",
            "question": "Where did I move?",
            "answer": "York",
            "question_type": "single-session-user",
            "question_date": "2023/02/01",
            "answer_session_ids": ["answer_s1"],
            "haystack_session_ids": ["answer_s1"],
            "haystack_dates": ["2023/01/01"],
            "haystack_sessions": [messages],
        }
    ]
    cleaned = {
        "q1": {
            "question": "Where did I move?",
            "answer": "York",
            "question_type": "single-session-user",
            "question_date": "2023/02/03",
            "evidence_ids": {"answer_s1"},
            "evidence_sessions": {
                "answer_s1": {"date": "2023/01/02", "content_sha256": _session_hash()}
            },
        }
    }
    return oracle, cleaned


def test_align_oracle_metadata_updates_only_dates_and_records_sample_ids():
    oracle, cleaned = _records()
    original = oracle[0].copy()

    aligned, manifest = align_oracle_records(oracle, cleaned)

    assert aligned[0]["question_date"] == "2023/02/03"
    assert aligned[0]["haystack_dates"] == ["2023/01/02"]
    assert aligned[0]["haystack_sessions"] == original["haystack_sessions"]
    assert oracle[0]["question_date"] == original["question_date"]
    assert oracle[0]["haystack_dates"] == original["haystack_dates"]
    assert manifest["question_date_changes"] == 1
    assert manifest["evidence_session_date_changes"] == 1
    assert manifest["affected_question_ids"] == ["q1"]


def test_alignment_rejects_mismatched_evidence_content():
    oracle, cleaned = _records()
    cleaned["q1"]["evidence_sessions"]["answer_s1"]["content_sha256"] = "wrong"

    with pytest.raises(ValueError, match="Evidence content differs"):
        align_oracle_records(oracle, cleaned)


def test_alignment_rejects_question_text_mismatch():
    oracle, cleaned = _records()
    cleaned["q1"]["question"] = "Different question"

    with pytest.raises(ValueError, match="question differs"):
        align_oracle_records(oracle, cleaned)
