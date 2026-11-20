"""Split tests: determinism, stratification, disjointness, protocol."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from nova_mem.data import Question, Session, stratified_split, split_report
from nova_mem.data.split import load_split, save_split

REPO = Path(__file__).resolve().parents[1]


def _make_questions(n_per_type: dict[str, int]) -> list[Question]:
    qs = []
    for qtype, n in n_per_type.items():
        for i in range(n):
            qs.append(
                Question(
                    question_id=f"{qtype}_{i}",
                    question_type=qtype,
                    question="q",
                    answer="a",
                    question_date="2023-01-01",
                    sessions=[Session("s1", "2023-01-01", [])],
                )
            )
    return qs


TYPE_COUNTS = {
    "single-session-user": 80,
    "single-session-assistant": 70,
    "single-session-preference": 80,
    "multi-session": 80,
    "temporal-reasoning": 80,
    "knowledge-update": 80,
    "abstention": 30,
}


def test_split_sizes_and_disjoint():
    qs = _make_questions(TYPE_COUNTS)
    split = stratified_split(qs, dev_size=100, seed=42)
    assert len(split["dev"]) == 100
    assert len(split["test"]) == 400
    assert not set(split["dev"]) & set(split["test"])


def test_split_deterministic():
    qs = _make_questions(TYPE_COUNTS)
    a = stratified_split(qs, dev_size=100, seed=42)
    b = stratified_split(qs, dev_size=100, seed=42)
    assert a == b
    c = stratified_split(qs, dev_size=100, seed=43)
    assert a != c  # different seed changes membership


def test_split_stratified():
    qs = _make_questions(TYPE_COUNTS)
    split = stratified_split(qs, dev_size=100, seed=42)
    by_id = {q.question_id: q for q in qs}
    dev_types = Counter(by_id[qid].question_type for qid in split["dev"])
    # proportional within ±1 (largest remainder)
    for t, n in TYPE_COUNTS.items():
        expected = round(n * 100 / 500)
        assert abs(dev_types[t] - expected) <= 1, (t, dev_types[t], expected)
    assert dev_types["abstention"] in (6, 7)


def test_split_roundtrip(tmp_path):
    qs = _make_questions(TYPE_COUNTS)
    split = stratified_split(qs, dev_size=100, seed=42)
    p = tmp_path / "split.json"
    save_split(split, p, meta={"x": 1})
    loaded = load_split(p)
    assert loaded == split


@pytest.mark.skipif(
    not (REPO / "data" / "splits" / "dev_test_v0.json").exists(), reason="split not generated"
)
@pytest.mark.skipif(
    not (REPO / "data" / "longmemeval_s_cleaned.json").exists(), reason="S file not downloaded"
)
def test_committed_split_protocol():
    from nova_mem.data import compute_stats, load_benchmark, split_report

    split = load_split(REPO / "data" / "splits" / "dev_test_v0.json")
    assert set(split) == {"dev", "test"}
    assert len(split["dev"]) == 100 and len(split["test"]) == 400
    assert not set(split["dev"]) & set(split["test"])
    # ids must reference real benchmark questions; report must cover every id
    questions = load_benchmark(REPO / "data" / "longmemeval_s_cleaned.json")
    rep = split_report(questions, split)
    assert sum(rep["dev"]["by_type"].values()) == 100
    assert sum(rep["test"]["by_type"].values()) == 400
    # dev must contain abstention questions (6 expected by stratification)
    n_abs_dev = sum("_abs" in qid for qid in split["dev"])
    assert 5 <= n_abs_dev <= 7


def test_split_report_shape():
    qs = _make_questions(TYPE_COUNTS)
    split = stratified_split(qs, dev_size=100, seed=42)
    rep = split_report(qs, split)
    assert rep["dev"]["n"] == 100 and rep["test"]["n"] == 400
    assert sum(rep["dev"]["by_type"].values()) == 100
