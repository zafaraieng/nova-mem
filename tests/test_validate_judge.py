"""Judge validation protocol: judge_agreement stats + validate_judge workflow.

Covers: Cohen's kappa hand-computed examples (previously untested function),
worksheet emit/score round-trip with monkeypatched benchmark, label parsing,
stratified sampling determinism, plan §5.2 verdict logic.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from nova_mem.data.loader import Question
from nova_mem.evaluation.judge import judge_agreement
from nova_mem.evaluation.judge import judge_hypotheses
from nova_mem.models.client import MockLLMClient
from validate_judge import (
    COLUMNS,
    emit_worksheet,
    load_judged_records,
    parse_label,
    score_worksheet,
    stratified_sample,
)


def rec(qid: str, label: bool, hyp: str = "h", model: str = "m"):
    return {
        "question_id": qid,
        "hypothesis": hyp,
        "autoeval_label": {"model": model, "label": label},
    }


def make_question(
    qid: str,
    qtype: str = "multi-session",
    answer: str = "Paris",
    question: str = "Where does the user live?",
) -> Question:
    return Question(
        question_id=qid,
        question_type=qtype,
        question=question,
        answer=answer,
        question_date="2023-01-01",
    )


class TestJudgeAgreement:
    def test_perfect_agreement(self):
        qs = [f"q{i}" for i in range(10)]
        a = [rec(q, i % 2 == 0) for i, q in enumerate(qs)]
        stats = judge_agreement(a, [rec(q, i % 2 == 0) for i, q in enumerate(qs)])
        assert stats["agreement"] == 1.0
        assert stats["cohens_kappa"] == 1.0
        assert stats["disagreements"] == []

    def test_hand_computed_kappa(self):
        # n=10, judge yes on q0..q5 (6), human yes on q0..q3+q4? -> construct:
        # judge: yes on {0,1,2,3,4,5}, human: yes on {0,1,2,3,4,6}
        # disagree on q5 (judge yes, human no) and q6 (judge no, human yes)
        qs = [f"q{i}" for i in range(10)]
        judge = [rec(q, i in {0, 1, 2, 3, 4, 5}) for i, q in enumerate(qs)]
        human = [rec(q, i in {0, 1, 2, 3, 4, 6}) for i, q in enumerate(qs)]
        stats = judge_agreement(judge, human)
        # pa = 8/10; p_yes both 0.6 -> pe = .36+.16 = .52; kappa = (0.8-0.52)/0.48
        assert stats["agreement"] == pytest.approx(0.8)
        assert stats["cohens_kappa"] == pytest.approx((0.8 - 0.52) / 0.48, abs=1e-4)
        assert set(stats["disagreements"]) == {"q5", "q6"}

    def test_no_common_ids_raises(self):
        with pytest.raises(ValueError, match="No common question ids"):
            judge_agreement([rec("a", True)], [rec("b", True)])


class TestParseLabel:
    def test_accepted_forms(self):
        for raw in ("yes", "YES", "Y", "true", "1"):
            assert parse_label(raw) is True
        for raw in ("no", "No", "n", "false", "0"):
            assert parse_label(raw) is False

    def test_blank_and_strict(self):
        assert parse_label("") is None
        assert parse_label(None) is None
        with pytest.raises(ValueError, match="unrecognized label"):
            parse_label("maybe", where="x")
        assert parse_label("maybe", strict=False) is None


class TestStratifiedSample:
    def test_proportional_allocation_and_determinism(self):
        records = (
            [rec(f"ms_{i:02d}", True) for i in range(20)]  # multi-session
            + [rec(f"tr_{i:02d}", True) for i in range(10)]  # temporal
            + [rec(f"ku_{i:02d}", True) for i in range(10)]  # knowledge-update
        )
        for r, t in zip(
            records,
            ["multi-session"] * 20 + ["temporal-reasoning"] * 10 + ["knowledge-update"] * 10,
        ):
            r["question_type"] = t
        s1 = stratified_sample(records, 20, seed=42)
        s2 = stratified_sample(records, 20, seed=42)
        assert [r["question_id"] for r in s1] == [r["question_id"] for r in s2]  # deterministic
        assert len(s1) == 20
        from collections import Counter

        got = Counter(r["question_type"] for r in s1)
        # 40 records, n=20 -> exactly proportional: 10/5/5
        assert got == {"multi-session": 10, "temporal-reasoning": 5, "knowledge-update": 5}

    def test_n_ge_total_returns_all(self):
        records = [{**rec(f"q{i}", True), "question_type": "multi-session"} for i in range(5)]
        out = stratified_sample(records, 10, seed=1)
        assert len(out) == 5


class TestWorksheetRoundTrip:
    @pytest.fixture
    def judged_records(self):
        return (
            [
                {
                    **rec(f"ms_{i:02d}", i % 3 != 0, hyp=f"hyp-ms-{i}"),
                    "question_type": "multi-session",
                }
                for i in range(12)
            ]
            + [
                {
                    **rec(f"tr_{i:02d}", i % 4 != 0, hyp=f"hyp-tr-{i}"),
                    "question_type": "temporal-reasoning",
                }
                for i in range(8)
            ]
            + [
                {**rec(f"ms_00_abs", False, hyp="unanswerable"), "question_type": "multi-session"},
            ]
        )

    @pytest.fixture
    def benchmark(self, judged_records):
        return [make_question(r["question_id"], r["question_type"]) for r in judged_records]

    def _patch_env(self, monkeypatch, tmp_path, judged_records, benchmark):
        import validate_judge

        eval_file = tmp_path / "eval-results-mock.jsonl"
        with open(eval_file, "w", encoding="utf-8") as f:
            for r in judged_records:
                f.write(json.dumps(r) + "\n")
        monkeypatch.setattr(validate_judge, "REPO", tmp_path)
        monkeypatch.setattr(
            validate_judge,
            "load_yaml",
            lambda _p: {"data": {"benchmark_file": "bench.json"}},
        )
        monkeypatch.setattr(validate_judge, "load_benchmark", lambda _p: benchmark)
        return validate_judge, eval_file

    def test_emit_then_score(self, monkeypatch, tmp_path, judged_records, benchmark):
        vj, eval_file = self._patch_env(monkeypatch, tmp_path, judged_records, benchmark)

        class Args:
            eval_files = [str(eval_file)]
            n = 15
            seed = 42
            tag = "unit"
            config = None

        vj.emit_worksheet(Args())
        ws = tmp_path / "results" / "judge_validation" / "worksheet_unit.csv"
        assert ws.exists()
        with open(ws, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows and set(COLUMNS) == set(rows[0].keys())
        assert all(r["human_label"] == "" for r in rows)
        assert all(r["judge_label"] in ("yes", "no") for r in rows)
        # judge prompt is single-line and non-empty (criteria visible to the labeler)
        assert all("\n" not in r["judge_prompt"] and r["judge_prompt"].strip() for r in rows)

        # human copies the judge everywhere except two flipped rows
        flip = {rows[0]["question_id"], rows[1]["question_id"]}
        for r in rows:
            if r["question_id"] in flip:
                r["human_label"] = "no" if r["judge_label"] == "yes" else "yes"
            else:
                r["human_label"] = r["judge_label"]
        with open(ws, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)

        class ScoreArgs:
            worksheet = str(ws)
            allow_missing = False

        report = vj.score_worksheet(ScoreArgs())
        assert report["n"] == len(rows)
        assert report["n_unfilled_excluded"] == 0
        # judge_agreement rounds to 4 decimals
        assert report["agreement"] == pytest.approx((len(rows) - 2) / len(rows), abs=1e-4)
        assert len(report["disagreements"]) == 2
        assert {d["question_id"] for d in report["disagreements"]} == flip
        assert report["cohens_kappa"] < 1.0
        # report file written next to the worksheet
        assert (tmp_path / "results" / "judge_validation" / "report_unit.json").exists()

    def test_missing_label_blocks_then_allows(
        self, monkeypatch, tmp_path, judged_records, benchmark
    ):
        vj, eval_file = self._patch_env(monkeypatch, tmp_path, judged_records, benchmark)

        class Args:
            eval_files = [str(eval_file)]
            n = 10
            seed = 42
            tag = "missing"
            config = None

        vj.emit_worksheet(Args())
        ws = tmp_path / "results" / "judge_validation" / "worksheet_missing.csv"
        with open(ws, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows[0]["human_label"] = "yes"  # only the first row labeled
        with open(ws, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)

        class ScoreArgs:
            worksheet = str(ws)
            allow_missing = False

        with pytest.raises(SystemExit, match="empty human_label"):
            vj.score_worksheet(ScoreArgs())

        class ScoreArgsAllow:
            worksheet = str(ws)
            allow_missing = True

        report = vj.score_worksheet(ScoreArgsAllow())
        assert report["n"] == 1
        assert report["n_unfilled_excluded"] == len(rows) - 1


class TestLoadJudgedRecords:
    def test_first_occurrence_wins_and_validation(self, tmp_path):
        f1 = tmp_path / "a.jsonl"
        f2 = tmp_path / "b.jsonl"
        f1.write_text(
            json.dumps(rec("q1", True, model="first")) + "\n" + json.dumps(rec("q2", False)) + "\n",
            encoding="utf-8",
        )
        f2.write_text(json.dumps(rec("q1", False, model="second")) + "\n", encoding="utf-8")
        out = load_judged_records([f1, f2])
        assert len(out) == 2
        assert {r["question_id"] for r in out} == {"q1", "q2"}
        winner = next(r for r in out if r["question_id"] == "q1")
        assert winner["autoeval_label"]["model"] == "first"

        bad = tmp_path / "bad.jsonl"
        bad.write_text(
            json.dumps({"question_id": "q1", "hypothesis": "h"}) + "\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="autoeval_label"):
            load_judged_records([bad])


def test_score_cli_uses_worksheet_path(monkeypatch, tmp_path):
    import validate_judge as vj

    worksheet = tmp_path / "worksheet_cli.csv"
    rows = [
        {
            "question_id": "q1",
            "question_type": "multi-session",
            "judge_label": "yes",
            "human_label": "yes",
        },
        {
            "question_id": "q2",
            "question_type": "multi-session",
            "judge_label": "no",
            "human_label": "no",
        },
        {
            "question_id": "q3",
            "question_type": "multi-session",
            "judge_label": "yes",
            "human_label": "no",
        },
    ]
    with worksheet.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    monkeypatch.setattr(sys, "argv", ["validate_judge.py", "--score", str(worksheet)])
    vj.main()
    report = json.loads((tmp_path / "report_cli.json").read_text(encoding="utf-8"))
    assert report["n"] == 3
    assert report["cohens_kappa"] == pytest.approx(0.4)
    assert (tmp_path / "report_cli.json").exists()


class TestJudgeResume:
    def test_resumes_after_existing_judgment(self, tmp_path):
        questions = [make_question("q1"), make_question("q2")]
        hypotheses = [
            {"question_id": "q1", "hypothesis": "first answer"},
            {"question_id": "q2", "hypothesis": "second answer"},
        ]
        out_path = tmp_path / "eval.jsonl"
        out_path.write_text(
            json.dumps(
                {
                    **hypotheses[0],
                    "autoeval_label": {"model": "mock-judge", "label": True},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        judge = MockLLMClient(model_id="mock-judge", respond=lambda _prompt: "yes")

        judged = judge_hypotheses(questions, hypotheses, judge, out_path)

        assert [r["question_id"] for r in judged] == ["q1", "q2"]
        assert len(judge.calls) == 1
        rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
        assert [r["question_id"] for r in rows] == ["q1", "q2"]

    def test_rejects_stale_partial_output(self, tmp_path):
        out_path = tmp_path / "eval.jsonl"
        out_path.write_text(
            json.dumps(
                {
                    "question_id": "q1",
                    "hypothesis": "old answer",
                    "autoeval_label": {"model": "mock-judge", "label": True},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        judge = MockLLMClient(model_id="mock-judge", respond=lambda _prompt: "yes")

        with pytest.raises(ValueError, match="hypothesis changed"):
            judge_hypotheses(
                [make_question("q1")],
                [{"question_id": "q1", "hypothesis": "new answer"}],
                judge,
                out_path,
            )
