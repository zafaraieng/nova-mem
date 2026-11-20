#!/usr/bin/env python3
"""Aggregate the ledger: per-run tables + paired statistics for the primary family.

Reads results/ledger.jsonl (append-only) and computes:
  * per-system official QA metrics (already in each record) as a comparison table
  * paired bootstrap + exact McNemar for the primary family (C vs A, D vs C,
    E vs C, oracle vs C) on per-question labels from the run artifacts
  * efficiency ledger summary (runtime, tokens, storage)
Writes results/aggregate.json and prints a markdown table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nova_mem.analysis import holm_correction, mcnemar_exact, paired_bootstrap  # noqa: E402

PRIMARY_FAMILY = ["C_vs_A", "D_vs_C", "E_vs_C", "oracle_vs_C"]
SYSTEM_KEYS = {
    "A": "A",
    "C": "C",
    "C_dense": "C_dense",
    "C_hybrid": "C_hybrid",
    "C_rrf": "C_rrf",
    "D": "D",
    "E": "E",
    "oracle": "oracle",
}


def load_labels(run_dir: Path) -> dict[str, bool]:
    """Load one unambiguous local/official judge output file for a run."""
    files = sorted(run_dir.glob("eval-results-*"))
    if not files:
        return {}
    if len(files) != 1:
        raise ValueError(f"{run_dir}: expected one judge output file, found {len(files)}")
    out = {}
    for line in files[-1].read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            question_id = rec["question_id"]
            label = rec["autoeval_label"]["label"]
            if not isinstance(label, bool):
                raise ValueError(f"{files[-1]}: invalid judge label for {question_id!r}")
            if question_id in out:
                raise ValueError(f"{files[-1]}: duplicate judgment for {question_id!r}")
            out[question_id] = label
    return out


def cache_summary(run: dict) -> tuple[int, int] | None:
    """Return (cache hits, live generation calls), or None when unrecorded."""
    if "cache_hits" in run and "live_generation_calls" in run:
        return int(run["cache_hits"]), int(run["live_generation_calls"])
    run_dir = REPO / run.get("run_dir", "")
    answers_path = run_dir / "answers.jsonl"
    if not answers_path.is_file():
        return None
    statuses = [
        cached
        for line in answers_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for cached in json.loads(line).get("timings", {}).values()
        if isinstance(cached, bool)
    ]
    if not statuses:
        return None
    return sum(statuses), sum(not cached for cached in statuses)


def _compatibility_error(run_a: dict, run_b: dict) -> str | None:
    checks = [
        ("dataset_revision", run_a.get("dataset_revision"), run_b.get("dataset_revision")),
        ("model_id", run_a.get("model_id"), run_b.get("model_id")),
        ("model_digest", run_a.get("model_digest"), run_b.get("model_digest")),
        (
            "reader_context_length",
            run_a.get("reader_context_length"),
            run_b.get("reader_context_length"),
        ),
        ("quantization", run_a.get("quantization"), run_b.get("quantization")),
        ("prompt_version", run_a.get("prompt_version"), run_b.get("prompt_version")),
        ("backend", run_a.get("backend"), run_b.get("backend")),
        ("seed", run_a.get("seed"), run_b.get("seed")),
        (
            "split",
            run_a.get("eval_config", {}).get("split"),
            run_b.get("eval_config", {}).get("split"),
        ),
        (
            "judge",
            run_a.get("eval_config", {}).get("judge"),
            run_b.get("eval_config", {}).get("judge"),
        ),
        (
            "judge_model",
            run_a.get("eval_config", {}).get("judge_model"),
            run_b.get("eval_config", {}).get("judge_model"),
        ),
    ]
    for name, value_a, value_b in checks:
        if value_a != value_b:
            return f"{name} differs ({value_a!r} vs {value_b!r})"
    if run_a.get("dataset_sha256") != run_b.get("dataset_sha256"):
        same_pairing = (
            run_a.get("dataset_comparison_group") == "longmemeval_s_evidence_aligned_v1"
            and run_a.get("dataset_comparison_group") == run_b.get("dataset_comparison_group")
            and run_a.get("dataset_comparison_fingerprint") is not None
            and run_a.get("dataset_comparison_fingerprint")
            == run_b.get("dataset_comparison_fingerprint")
        )
        if not same_pairing:
            return "dataset_sha256 differs without a verified matching comparison fingerprint"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", default=str(REPO / "results" / "ledger.jsonl"))
    ap.add_argument("--filter", default=None, help="only experiment_ids containing this string")
    ap.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="exact experiment ID to include; repeat to select a comparison set",
    )
    ap.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.ledger).read_text().splitlines() if l.strip()]
    if args.filter:
        records = [r for r in records if args.filter in r["experiment_id"]]
    records = [r for r in records if not any("mock-backend" in d for d in r.get("deviations", []))]
    # The ledger is append-only, so relaunch/recovery flows can append several
    # records for one experiment_id (e.g. a 56-partial superseded by a 100-final
    # C-BM25 run, or repeated fast-fail revalidations of A/oracle). Aggregation
    # must therefore keep only the LAST record per experiment_id — the newest
    # state of the run — or duplicates would double-count rows in every table.
    latest_by_id: dict[str, dict] = {}
    for r in records:
        latest_by_id[r["experiment_id"]] = r
    dropped = len(records) - len(latest_by_id)
    if dropped:
        print(
            f"[dedup] kept last record for {dropped} superseded ledger record(s) "
            f"(append-only ledger; newest state wins)"
        )
    records = list(latest_by_id.values())
    if args.experiment:
        selected_ids = set(args.experiment)
        records = [r for r in records if r["experiment_id"] in selected_ids]
    incomplete_records = [
        r
        for r in records
        if (r.get("eval_config") or {}).get("judge") not in (None, "none")
        and r.get("qa_metrics") is None
    ]
    incomplete_ids = {r["experiment_id"] for r in incomplete_records}
    records = [r for r in records if r["experiment_id"] not in incomplete_ids]
    if args.experiment:
        available_ids = {r["experiment_id"] for r in records}
        missing_ids = sorted(set(args.experiment) - available_ids)
        if missing_ids:
            sys.exit(f"Requested experiment IDs are missing or incomplete: {missing_ids}")
    if not records:
        sys.exit("No non-mock ledger records to aggregate (run real experiments first).")
    if incomplete_records:
        print(
            f"[skip] excluded {len(incomplete_records)} incomplete judged run(s): "
            f"{sorted(incomplete_ids)}"
        )

    # ---- table ---------------------------------------------------------
    print(f"\n### Runs ({len(records)})\n")
    print(
        "| experiment | system | split | n | accuracy | task-avg | abstention | generation_s | cache hits / live calls | judge_s | total_wall_s | tokens_in | tokens_out |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        qa = r.get("qa_metrics") or {}
        acc = qa.get("overall_accuracy")
        taskavg = qa.get("task_averaged_accuracy")
        abst = (qa.get("abstention") or {}).get("accuracy")
        n = (r.get("eval_config") or {}).get("n_questions")
        generation_s = r.get("generation_runtime_s", r.get("runtime_s"))
        cache_counts = cache_summary(r)
        if cache_counts is None:
            cache_text = "unknown"
        else:
            cache_text = f"{cache_counts[0]} / {cache_counts[1]}"
        print(
            f"| {r['experiment_id']} | {r.get('system_key', r['memory_arch'])} "
            f"| {r['eval_config']['split']} | {n} | {acc} | {taskavg} | {abst} "
            f"| {generation_s} | {cache_text} | {r.get('judge_runtime_s')} | {r.get('total_wall_s')} "
            f"| {r['tokens_in']} | {r['tokens_out']} |"
        )

    # ---- paired stats (primary family) ----------------------------------
    by_system: dict[str, list[dict]] = {}
    for r in records:
        key = r.get("system_key")
        if key:
            by_system.setdefault(key, []).append(r)

    stats: dict[str, dict] = {}
    pvals: dict[str, float] = {}
    for pair in PRIMARY_FAMILY:
        a_key, b_key = pair.split("_vs_")
        a_candidates = by_system.get(a_key, [])
        b_candidates = by_system.get(b_key, [])
        if len(a_candidates) != 1 or len(b_candidates) != 1:
            print(
                f"[skip] {pair}: select exactly one run per system "
                f"(found {a_key}={len(a_candidates)}, {b_key}={len(b_candidates)}); "
                "use repeated --experiment IDs"
            )
            continue
        a_run, b_run = a_candidates[0], b_candidates[0]
        mismatch = _compatibility_error(a_run, b_run)
        if mismatch:
            print(f"[skip] {pair}: incompatible runs ({mismatch})")
            continue
        try:
            la = load_labels(REPO / a_run["run_dir"])
            lb = load_labels(REPO / b_run["run_dir"])
        except ValueError as exc:
            print(f"[skip] {pair}: {exc}")
            continue
        if set(la) != set(lb):
            print(f"[skip] {pair}: judged question sets differ ({len(la)} vs {len(lb)})")
            continue
        common = sorted(la)
        if len(common) < 10:
            print(f"[skip] {pair}: only {len(common)} paired judgments")
            continue
        labels_a = [int(la[q]) for q in common]
        labels_b = [int(lb[q]) for q in common]
        boot = paired_bootstrap(labels_a, labels_b, n_resamples=args.bootstrap_resamples)
        mc = mcnemar_exact(labels_a, labels_b)
        stats[pair] = {
            "comparison": f"{a_run['memory_arch']} vs {b_run['memory_arch']}",
            "n_common": len(common),
            "bootstrap": {
                "metric_a": boot.metric_a,
                "metric_b": boot.metric_b,
                "delta": boot.delta,
                "ci95": list(boot.ci95),
                "p": boot.p_bootstrap,
            },
            "mcnemar": mc,
        }
        pvals[pair] = boot.p_bootstrap

    if pvals:
        stats["holm_adjusted_p"] = holm_correction(pvals)
        print("\n### Primary family (paired bootstrap, 95% CI, Holm-corrected)\n")
        print("| comparison | Δ accuracy | 95% CI | p (boot) | p (Holm) | McNemar p | n |")
        print("|---|---|---|---|---|---|---|")
        for pair, s in stats.items():
            if pair == "holm_adjusted_p":
                continue
            b = s["bootstrap"]
            print(
                f"| {s['comparison']} | {b['delta']:+.4f} | [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] "
                f"| {b['p']} | {stats['holm_adjusted_p'][pair]} | {s['mcnemar']['p_exact']} | {s['n_common']} |"
            )

    out = REPO / "results" / "aggregate.json"
    out.write_text(
        json.dumps({"runs": len(records), "stats": stats}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
