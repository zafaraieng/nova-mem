"""Data loader tests: synthetic schema + real-file spot checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nova_mem.data import load_benchmark, split_questions, load_split
from nova_mem.data.loader import Question, Session

REPO = Path(__file__).resolve().parents[1]
S_FILE = REPO / "data" / "longmemeval_s_cleaned.json"
ORACLE_FILE = REPO / "data" / "longmemeval_oracle.json"


def _synthetic_entry(qid="q1", qtype="single-session-user", abstention=False, noans=False):
    # Real-data convention (verified on the cleaned files): abstention rows keep
    # their BASE question_type; abstention is marked only by `_abs` in the id.
    prefix = "noans_" if noans else "answer_"
    return {
        "question_id": qid + ("_abs" if abstention else ""),
        "question_type": qtype,
        "question": "What is my job?",
        "answer": "Teacher" if not abstention else "The user did not mention their job.",
        "question_date": "2023-06-01",
        "haystack_dates": ["2023-01-01", "2023-02-01"],
        "haystack_session_ids": ["s_filler_1", prefix + "evid1"],
        "haystack_sessions": [
            [{"role": "user", "content": "hello"}],
            [
                {"role": "user", "content": "I work as a teacher"},
                {"role": "assistant", "content": "great"},
            ],
        ],
        "answer_session_ids": [prefix + "evid1"],
    }


def _write(tmp_path, entries, name="synthetic.json"):
    p = tmp_path / name
    p.write_text(json.dumps(entries), encoding="utf-8")
    return p


def test_parse_synthetic(tmp_path):
    qs = load_benchmark(_write(tmp_path, [_synthetic_entry()]))
    assert len(qs) == 1
    q = qs[0]
    assert q.question_id == "q1"
    assert not q.is_abstention
    assert q.evidence_session_ids == ["answer_evid1"]
    assert len(q.sessions) == 2
    assert q.sessions[1].messages[0].content == "I work as a teacher"
    assert q.sessions[1].turn_docs()[0][0] == "answer_evid1_1"


def test_parse_abstention_noans(tmp_path):
    # abstention marked via _abs; base type preserved (real-data convention)
    qs = load_benchmark(
        _write(
            tmp_path,
            [_synthetic_entry(qid="q2", qtype="temporal-reasoning", abstention=True, noans=True)],
        )
    )
    q = qs[0]
    assert q.is_abstention
    assert q.question_type == "temporal-reasoning"
    # noans_ prefix stripped for retrieval-metric evidence ids (official corpus convention;
    # the cleaned files themselves use answer_ even for _abs questions)
    assert q.evidence_session_ids == ["answer_evid1"]
    assert q.answer_session_ids == ["noans_evid1"]


def test_unknown_type_rejected(tmp_path):
    bad = _synthetic_entry()
    bad["question_type"] = "mystery-type"
    with pytest.raises(ValueError):
        load_benchmark(_write(tmp_path, [bad]))


def test_split_questions_partition(tmp_path):
    entries = [_synthetic_entry(qid=f"q{i}") for i in range(10)]
    qs = load_benchmark(_write(tmp_path, entries))
    split = {"dev": [f"q{i}" for i in range(4)], "test": [f"q{i}" for i in range(4, 10)]}
    parts = split_questions(qs, split)
    assert [q.question_id for q in parts["dev"]] == split["dev"]
    with pytest.raises(ValueError):  # missing coverage
        split_questions(qs, {"dev": ["q0"], "test": ["q1"]})


@pytest.mark.skipif(not S_FILE.exists(), reason="S file not downloaded")
def test_real_file_spot_checks():
    qs = load_benchmark(S_FILE)
    assert len(qs) == 500
    ids = [q.question_id for q in qs]
    assert len(set(ids)) == 500
    n_abs = sum(q.is_abstention for q in qs)
    assert n_abs == 30, f"expected 30 abstention questions, got {n_abs}"
    # abstention rows carry a BASE question_type (never the literal 'abstention')
    assert all(q.question_type != "abstention" for q in qs)
    # every question has >=1 evidence session; all evidence ids use the answer_ prefix
    # (the noans_ prefix exists only in the official retrieval corpus convention)
    for q in qs:
        assert len(q.answer_session_ids) >= 1
        assert all(a.split("_")[0] == "answer" for a in q.answer_session_ids)


@pytest.mark.skipif(
    not (S_FILE.exists() and ORACLE_FILE.exists()), reason="data files not downloaded"
)
def test_oracle_file_consistency():
    """Oracle file must carry the same 500 question ids as S."""
    s = load_benchmark(S_FILE)
    o = load_benchmark(ORACLE_FILE)
    assert {q.question_id for q in s} == {q.question_id for q in o}
    # oracle haystacks are much smaller (evidence-only)
    s_sizes = {q.question_id: len(q.sessions) for q in s}
    o_sizes = {q.question_id: len(q.sessions) for q in o}
    smaller = sum(o_sizes[qid] <= s_sizes[qid] for qid in o_sizes)
    assert smaller == 500
