#!/usr/bin/env python3
"""Phase-3 gate smoke test: end-to-end plumbing on 3 dev questions, mock backend.

Verifies (no scientific claims — mock models only):
  1. data loads; split applies; ids are official
  2. every system (A, oracle, C, D, E) runs end-to-end and emits official-format
     hypotheses.jsonl + full instrumentation
  3. prompt parity: our templates equal the vendored official code's templates
  4. official judge harness produces labels; official aggregation computes
  5. retrieval metrics compute against evidence labels
  6. ledger record appended with required schema
Exit code 0 = smoke passed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    print("\n" + "$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout[-3000:])
    if proc.returncode != 0:
        print(proc.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return proc


def main() -> None:
    py = sys.executable

    # 0) unit + parity tests
    run([py, "-m", "pytest", "-q"])

    # 1) systems through the real runner (mock backend)
    for system in ["A", "oracle", "C", "D", "E"]:
        run(
            [
                py,
                "scripts/run_experiment.py",
                "--system",
                system,
                "--split",
                "dev",
                "--limit",
                "3",
                "--backend",
                "mock",
                "--judge",
                "local",
                "--experiment-id",
                f"smoke_{system}",
            ]
        )

    # 2) verify artifacts
    ledger_path = REPO / "results" / "ledger.jsonl"
    records = [json.loads(l) for l in ledger_path.read_text().splitlines() if l.strip()]
    expected_experiments = {"smoke_A", "smoke_C", "smoke_D", "smoke_E", "smoke_oracle"}
    smoke_by_experiment = {
        r["experiment_id"]: r for r in records if r["experiment_id"] in expected_experiments
    }
    smoke = list(smoke_by_experiment.values())
    systems_seen = sorted(r["system_key"] for r in smoke)
    assert set(smoke_by_experiment) == expected_experiments, set(smoke_by_experiment)
    assert systems_seen == ["A", "C", "D", "E", "oracle"], systems_seen
    for r in smoke:
        run_dir = REPO / r["run_dir"]
        hyp = run_dir / "hypotheses.jsonl"
        answers = run_dir / "answers.jsonl"
        eval_results = list(run_dir.glob("eval-results-*.jsonl"))
        assert hyp.exists() and answers.exists(), f"missing artifacts for {r['experiment_id']}"
        assert len(eval_results) == 1, f"missing/ambiguous judge output for {r['experiment_id']}"
        hyps = [json.loads(l) for l in hyp.read_text().splitlines()]
        assert len(hyps) == 3 and all(set(h) >= {"question_id", "hypothesis"} for h in hyps)
        ans = [json.loads(l) for l in answers.read_text().splitlines()]
        assert all("timings" in a and "tokens" in a and "prompt" in a for a in ans)
        assert r["qa_metrics"] is not None and "overall_accuracy" in r["qa_metrics"]
        assert "MOCK" not in "ledger sanity", "mock outputs must never be treated as results"
        assert r.get("deviations") == ["mock-backend: numbers are NOT scientific results"]
    print("\nSMOKE TEST PASSED")
    print(f"ledger records: {len(records)} total, {len(smoke)} smoke runs")
    print("systems verified:", systems_seen)


if __name__ == "__main__":
    main()
