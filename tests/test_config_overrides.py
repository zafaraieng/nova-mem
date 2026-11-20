"""Config override mechanism (E7 sweeps): parsing, application, failure modes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nova_mem.config import apply_overrides, config_hash, parse_override


@pytest.fixture
def cfg():
    return {
        "data": {"benchmark_file": "longmemeval_s_cleaned.json", "dev_size": 100},
        "retrieval": {
            "system_C": {
                "retriever": "bm25",
                "granularity": "session",
                "top_k": 5,
                "user_only_index": True,
                "con": True,
            },
        },
        "generation": {"max_tokens": 500, "temperature": 0.0, "seed": 0},
    }


class TestParseOverride:
    def test_scalar_types(self):
        assert parse_override("retrieval.system_C.top_k=3") == ("retrieval.system_C.top_k", 3)
        assert parse_override("generation.temperature=0.5") == ("generation.temperature", 0.5)
        assert parse_override("retrieval.system_C.con=false") == ("retrieval.system_C.con", False)
        assert parse_override("retrieval.system_C.retriever=dense") == (
            "retrieval.system_C.retriever",
            "dense",
        )

    def test_malformed(self):
        with pytest.raises(ValueError):
            parse_override("no-equals-sign")
        with pytest.raises(ValueError):
            parse_override("=3")  # empty key


class TestApplyOverrides:
    def test_nested_set(self, cfg):
        out = apply_overrides(cfg, ["retrieval.system_C.top_k=3"])
        assert out["retrieval"]["system_C"]["top_k"] == 3
        assert cfg["retrieval"]["system_C"]["top_k"] == 5  # input untouched

    def test_multiple_overrides(self, cfg):
        out = apply_overrides(
            cfg,
            [
                "retrieval.system_C.top_k=10",
                "retrieval.system_C.retriever=dense",
                "retrieval.system_C.granularity=turn",
            ],
        )
        c = out["retrieval"]["system_C"]
        assert (c["top_k"], c["retriever"], c["granularity"]) == (10, "dense", "turn")

    def test_typed_mapping_form(self, cfg):
        out = apply_overrides(cfg, {"retrieval.system_C.top_k": 3, "generation.seed": 1})
        assert out["retrieval"]["system_C"]["top_k"] == 3
        assert out["generation"]["seed"] == 1

    def test_unknown_key_fails_loudly(self, cfg):
        with pytest.raises(KeyError, match="topj_k"):
            apply_overrides(cfg, ["retrieval.system_C.topj_k=3"])
        with pytest.raises(KeyError, match="system_X"):
            apply_overrides(cfg, ["retrieval.system_X.top_k=3"])
        with pytest.raises(KeyError, match="retrieval"):
            apply_overrides(cfg, ["retrieval.top_k=3"])  # skips a level

    def test_mapping_leaf_rejected(self, cfg):
        with pytest.raises(ValueError, match="leaf scalar"):
            apply_overrides(cfg, ['retrieval.system_C={"a": 1}'])

    def test_non_string_spec_rejected(self, cfg):
        with pytest.raises(TypeError):
            apply_overrides(cfg, [("retrieval.system_C.top_k", 3)])  # tuple, not a str

    def test_hash_distinguishes_override(self, cfg):
        base = config_hash(cfg)
        swept = config_hash(apply_overrides(cfg, ["retrieval.system_C.top_k=3"]))
        assert base != swept


class TestRunExperimentIntegration:
    """End-to-end --set plumbing: mock run records overrides, no ledger write."""

    S_FILE = Path(__file__).resolve().parents[1] / "data" / "longmemeval_s_cleaned.json"

    @pytest.mark.skipif(not S_FILE.exists(), reason="S file not downloaded")
    def test_set_flag_recorded_in_summary(self, tmp_path):
        import shutil
        import subprocess
        import sys

        repo = Path(__file__).resolve().parents[1]
        ledger = repo / "results" / "ledger.jsonl"
        n_before = sum(1 for _ in open(ledger, encoding="utf-8")) if ledger.exists() else 0
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/run_experiment.py",
                "--system",
                "C",
                "--split",
                "dev",
                "--limit",
                "1",
                "--backend",
                "mock",
                "--judge",
                "local",
                "--no-ledger",
                "--set",
                "retrieval.system_C.top_k=3",
                "--experiment-id",
                "unit_override_check",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        summary = json.loads(proc.stdout[proc.stdout.index("{") :])
        assert summary["config_overrides"] == {"retrieval.system_C.top_k": 3}
        # the mock C run must have used the overridden k in its trace
        run_dir = repo / "results" / "runs" / "unit_override_check"
        with open(run_dir / "answers.jsonl", encoding="utf-8") as f:
            rec = json.loads(f.readline())
        assert rec["extras"]["n_retrieved"] == 3
        # --no-ledger must not append a record
        n_after = sum(1 for _ in open(ledger, encoding="utf-8")) if ledger.exists() else 0
        assert n_after == n_before, "--no-ledger appended to the ledger"
        # cleanup mechanism-test artifacts (run dirs are untracked by design)
        shutil.rmtree(run_dir, ignore_errors=True)
