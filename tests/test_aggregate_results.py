from __future__ import annotations

import json

import pytest

from scripts.aggregate_results import _compatibility_error, cache_summary, load_labels


def _run(**overrides):
    record = {
        "model_id": "qwen",
        "reader_context_length": None,
        "quantization": "Q4_K_M",
        "dataset_sha256": "abc",
        "prompt_version": "v1",
        "backend": "ollama",
        "eval_config": {"split": "dev", "judge_model": "judge"},
    }
    record.update(overrides)
    return record


def test_compatibility_requires_matching_reader_and_evaluation():
    assert _compatibility_error(_run(), _run()) is None
    mismatch = _compatibility_error(_run(), _run(model_id="llama"))
    assert mismatch is not None
    assert "model_id differs" in mismatch


def test_dataset_hash_difference_requires_matching_evidence_fingerprint():
    a = _run(
        dataset_sha256="cleaned-s",
        dataset_comparison_group="longmemeval_s_evidence_aligned_v1",
        dataset_comparison_fingerprint="same-shared-evidence",
    )
    b = _run(
        dataset_sha256="oracle-aligned",
        dataset_comparison_group="longmemeval_s_evidence_aligned_v1",
        dataset_comparison_fingerprint="same-shared-evidence",
    )
    assert _compatibility_error(a, b) is None
    assert _compatibility_error(a, _run(dataset_sha256="oracle-aligned")) is not None


def test_load_labels_rejects_multiple_judge_files(tmp_path):
    (tmp_path / "eval-results-a.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "eval-results-b.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="expected one judge output file"):
        load_labels(tmp_path)


def test_load_labels_rejects_duplicate_question_ids(tmp_path):
    path = tmp_path / "eval-results-judge.jsonl"
    row = {"question_id": "q1", "autoeval_label": {"label": True}}
    path.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate judgment"):
        load_labels(tmp_path)


def test_cache_summary_infers_legacy_answer_artifact_hits(tmp_path, monkeypatch):
    import scripts.aggregate_results as aggregate

    monkeypatch.setattr(aggregate, "REPO", tmp_path)
    run_dir = tmp_path / "results" / "runs" / "cached"
    run_dir.mkdir(parents=True)
    (run_dir / "answers.jsonl").write_text(
        json.dumps({"timings": {"generate_s": 0.0, "generate_cached": True}})
        + "\n"
        + json.dumps({"timings": {"generate_s": 2.0, "generate_cached": False}})
        + "\n",
        encoding="utf-8",
    )
    assert cache_summary({"run_dir": "results/runs/cached"}) == (1, 1)
