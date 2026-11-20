from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import scripts.run_experiment as run_experiment


def test_generation_failures_are_persisted_without_a_success_ledger_row(monkeypatch, tmp_path):
    question_ids = ["q1", "q2", "q3"]
    questions = [SimpleNamespace(question_id=question_id) for question_id in question_ids]
    monkeypatch.setattr(run_experiment, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        run_experiment,
        "load_yaml",
        lambda _path: {
            "data": {"benchmark_file": "benchmark.json", "split_file": "split.json"},
            "cache": {"root": "cache", "enabled": True},
            "generation": {"seed": 0},
        },
    )
    monkeypatch.setattr(run_experiment, "load_config", lambda _name: {})
    monkeypatch.setattr(run_experiment, "load_benchmark", lambda _path: questions)
    monkeypatch.setattr(
        run_experiment, "dataset_comparison_fingerprint", lambda _questions: "fingerprint"
    )
    monkeypatch.setattr(run_experiment, "load_split", lambda _path: {"dev": question_ids})
    monkeypatch.setattr(run_experiment, "sha256_file", lambda _path: "dataset-hash")

    class FailingSystem:
        name = "failing-system"

        def answer(self, question, _reader, **_kwargs):
            raise TimeoutError(f"timed out {question.question_id}")

    monkeypatch.setattr(
        run_experiment,
        "build_systems",
        lambda *_args, **_kwargs: {"A": FailingSystem},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_experiment.py",
            "--system",
            "A",
            "--split",
            "dev",
            "--backend",
            "mock",
            "--experiment-id",
            "failed-run",
        ],
    )

    with pytest.raises(RuntimeError, match="generation incomplete"):
        run_experiment.main()

    run_dir = tmp_path / "results" / "runs" / "failed-run"
    failures = [
        json.loads(line)
        for line in (run_dir / "generation-errors.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert failures == [
        {
            "question_id": question_id,
            "error_type": "TimeoutError",
            "error": f"timed out {question_id}",
        }
        for question_id in question_ids
    ]
    assert (run_dir / "answers.jsonl").read_text(encoding="utf-8") == ""
    assert not (tmp_path / "results" / "ledger.jsonl").exists()


def test_test_run_requires_frozen_unmodified_complete_unique_run(monkeypatch, tmp_path):
    monkeypatch.setattr(run_experiment, "REPO_ROOT", tmp_path)
    args = SimpleNamespace(
        split="test",
        config="frozen_v1.yaml",
        overrides=[],
        limit=None,
        no_ledger=False,
        experiment_id="test-A",
        judge="local",
        backend=None,
        system="A",
    )
    run_experiment._validate_test_run(args, {"freeze": "v1"})

    (tmp_path / "results" / "runs" / "test-A").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Run directory already exists"):
        run_experiment._validate_test_run(args, {"freeze": "v1"})

    args.limit = 10
    with pytest.raises(ValueError, match="Partial test runs"):
        run_experiment._validate_test_run(args, {"freeze": "v1"})


def test_test_run_rejects_overrides_and_non_frozen_config(monkeypatch, tmp_path):
    monkeypatch.setattr(run_experiment, "REPO_ROOT", tmp_path)
    args = SimpleNamespace(
        split="test",
        config="local.yaml",
        overrides=["generation.seed=2"],
        limit=None,
        no_ledger=False,
        experiment_id="test-A",
        judge="local",
        backend=None,
        system="A",
    )
    with pytest.raises(ValueError, match="frozen_v1.yaml"):
        run_experiment._validate_test_run(args, {"freeze": "v1"})

    args.config = "frozen_v1.yaml"
    with pytest.raises(ValueError, match="overrides"):
        run_experiment._validate_test_run(args, {"freeze": "v1"})


def test_frozen_test_data_requires_pinned_dataset_and_split(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_experiment, "sha256_file", lambda _path: run_experiment.SPLIT_FILE_SHA256
    )
    ids = [f"q{index}" for index in range(500)]
    split = {"dev": ids[:100], "test": ids[100:]}
    benchmark = [SimpleNamespace(question_id=question_id) for question_id in ids]
    args = SimpleNamespace(split="test", system="A")
    cfg = {
        "data": {"split_file": "splits/dev_test_v0.json"},
        "dataset": {
            "files": {"longmemeval_s_cleaned.json": run_experiment.CLEANED_S_SHA256},
            "split": "splits/dev_test_v0.json",
            "split_seed": 42,
            "dev_size": 100,
        },
    }

    run_experiment._validate_frozen_test_data(
        args,
        cfg,
        "longmemeval_s_cleaned.json",
        run_experiment.CLEANED_S_SHA256,
        tmp_path / "split.json",
        split,
        benchmark,
    )

    broken_split = {"dev": ids[:100], "test": ids[100:-1]}
    with pytest.raises(ValueError, match="exact committed"):
        run_experiment._validate_frozen_test_data(
            args,
            cfg,
            "longmemeval_s_cleaned.json",
            run_experiment.CLEANED_S_SHA256,
            tmp_path / "split.json",
            broken_split,
            benchmark,
        )

    with pytest.raises(ValueError, match="checksum"):
        run_experiment._validate_frozen_test_data(
            args,
            cfg,
            "longmemeval_s_cleaned.json",
            "wrong-hash",
            tmp_path / "split.json",
            split,
            benchmark,
        )
