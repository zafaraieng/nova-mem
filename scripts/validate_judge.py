#!/usr/bin/env python3
"""Judge validation protocol (plan §5.2, runbook §6): human-vs-judge agreement.

Two modes:

  emit-worksheet  Sample N stratified judged records (proportional by base
                  question_type, seed 42) from one or more run files and write
                  a CSV worksheet for hand-labeling. Each row carries the
                  question, gold answer, hypothesis, the judge's label, and the
                  exact official judge prompt (the labeling criteria).

  score           Read the filled worksheet (``human_label`` column: yes/no),
                  compute agreement + Cohen's kappa via
                  ``nova_mem.evaluation.judge.judge_agreement``, emit a
                  disagreement table with examples, and apply the plan §5.2
                  verdict: kappa >= 0.6 -> local judge validated; below ->
                  prompt-iteration round (max two), then escalate to D6-A.

Examples:
    python scripts/validate_judge.py --emit-worksheet \
        --eval-files results/runs/pilot_A_dev_primary/eval-results-qwen2.5-7b-instruct-q4_K_M.jsonl \
        --tag g1 --n 100
    #   ... fill the human_label column in results/judge_validation/worksheet_g1.csv ...
    python scripts/validate_judge.py --score results/judge_validation/worksheet_g1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nova_mem.config import load_yaml  # noqa: E402
from nova_mem.data import load_benchmark  # noqa: E402
from nova_mem.evaluation.judge import judge_agreement  # noqa: E402
from nova_mem.memory.prompts import judge_prompt  # noqa: E402
from nova_mem.utils import get_logger  # noqa: E402

logger = get_logger("validate_judge")

YES = {"yes", "y", "true", "1"}
NO = {"no", "n", "false", "0"}
KAPPA_THRESHOLD = 0.6  # plan §5.2

COLUMNS = [
    "question_id",
    "question_type",
    "is_abstention",
    "question",
    "gold_answer",
    "hypothesis",
    "judge_model",
    "judge_label",
    "judge_prompt",
    "human_label",
    "notes",
]


def parse_label(raw: str, *, strict: bool = True, where: str = "") -> bool | None:
    """yes/y/true/1 -> True; no/n/false/0 -> False; '' -> None (missing)."""
    s = (raw or "").strip().lower()
    if s in YES:
        return True
    if s in NO:
        return False
    if s == "":
        return None
    if strict:
        raise ValueError(
            f"unrecognized label {raw!r} in {where} " f"(expected yes/no/y/n/true/false/1/0)"
        )
    return None


def load_judged_records(paths: list[Path]) -> list[dict]:
    """Union of judged records across run files; first occurrence of a
    question_id wins (pass files in priority order)."""
    seen: dict[str, dict] = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "autoeval_label" not in rec:
                    raise ValueError(
                        f"{p}: record {rec.get('question_id')!r} has no autoeval_label"
                    )
                seen.setdefault(rec["question_id"], rec)
    return list(seen.values())


def stratified_sample(records: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Deterministic stratified proportional sample by base question_type.

    Largest-remainder quotas (same scheme as data/split.py), then a seeded
    shuffle within each type's pool of sorted question_ids.
    """
    by_type: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by_type[r["question_type"]].append(r["question_id"])
    total = len(records)
    if n >= total:
        return list(records)
    raw = {t: len(ids) * n / total for t, ids in by_type.items()}
    quota = {t: int(v) for t, v in raw.items()}
    remainder = n - sum(quota.values())
    order = sorted(raw, key=lambda t: (-(raw[t] - quota[t]), t))
    i = 0
    while remainder > 0:
        quota[order[i % len(order)]] += 1
        remainder -= 1
        i += 1
    rng = random.Random(seed)
    picked: set[str] = set()
    for t, ids in by_type.items():
        pool = sorted(ids)
        rng.shuffle(pool)
        picked.update(pool[: quota[t]])
    return [r for r in records if r["question_id"] in picked]


def emit_worksheet(args) -> Path:
    paths = [Path(p) for p in args.eval_files]
    records = load_judged_records(paths)
    if not records:
        raise SystemExit("no judged records found in the given files")

    cfg = load_yaml(REPO / "configs" / (args.config or "local.yaml"))
    benchmark = load_benchmark(REPO / "data" / cfg["data"]["benchmark_file"])
    qmap = {q.question_id: q for q in benchmark}

    # attach question_type from the benchmark (eval files don't carry it);
    # records for unknown question ids are dropped and reported
    enriched, dropped = [], 0
    for r in records:
        q = qmap.get(r["question_id"])
        if q is None:
            dropped += 1
            continue
        enriched.append({**r, "question_type": q.question_type})
    if dropped:
        logger.warning("%d judged record(s) not in the benchmark — excluded", dropped)
    if not enriched:
        raise SystemExit("no judged records match the benchmark")
    records = stratified_sample(enriched, args.n, seed=args.seed)

    out_dir = REPO / "results" / "judge_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"worksheet_{args.tag}.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in records:
            q = qmap[r["question_id"]]  # guaranteed present (enrichment step)
            prompt = judge_prompt(
                q.question_type, q.question, q.answer, r["hypothesis"], abstention=q.is_abstention
            )
            w.writerow(
                {
                    "question_id": q.question_id,
                    "question_type": q.question_type,
                    "is_abstention": q.is_abstention,
                    "question": q.question,
                    "gold_answer": q.answer,
                    "hypothesis": r["hypothesis"],
                    "judge_model": r["autoeval_label"]["model"],
                    "judge_label": "yes" if r["autoeval_label"]["label"] else "no",
                    # single-line rendering for spreadsheet ergonomics; verbatim text
                    # is reproducible via judge_prompt() and shown in the report
                    "judge_prompt": " | ".join(prompt.split("\n")),
                    "human_label": "",
                    "notes": "",
                }
            )
    counts = Counter(r["question_type"] for r in records)
    print(f"[worksheet] {out_path.relative_to(REPO)}")
    print(
        f"[worksheet] {len(records)} records "
        f"({dropped} excluded: judged ids not in the benchmark); "
        f"stratification: {dict(sorted(counts.items()))}"
    )
    print(
        "[worksheet] fill the human_label column with yes/no against the judge "
        "criteria (judge_prompt column), then run --score on this file"
    )
    return out_path


def score_worksheet(args) -> dict:
    ws_path = Path(args.worksheet)
    with open(ws_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"worksheet {ws_path} is empty or has no header")
    expected = {"question_id", "judge_label", "human_label"}
    missing_cols = expected - set(rows[0])
    if missing_cols:
        raise SystemExit(f"worksheet missing column(s): {sorted(missing_cols)}")

    judge_records, human_records = [], []
    bad, blank = [], []
    for i, row in enumerate(rows, start=2):
        where = f"{ws_path.name} line {i} ({row.get('question_id', '?')})"
        jl = parse_label(row["judge_label"], where=where)
        hl = parse_label(row["human_label"], strict=False, where=where)
        if hl is None:
            blank.append(row.get("question_id", "?"))
            continue
        judge_records.append(
            {
                "question_id": row["question_id"],
                "hypothesis": row.get("hypothesis", ""),
                "autoeval_label": {"model": row.get("judge_model", "judge"), "label": jl},
            }
        )
        human_records.append(
            {
                "question_id": row["question_id"],
                "hypothesis": row.get("hypothesis", ""),
                "autoeval_label": {"model": "human", "label": hl},
            }
        )

    if blank and not args.allow_missing:
        raise SystemExit(
            f"{len(blank)} row(s) have an empty human_label (e.g. {blank[:5]}); "
            "fill them or pass --allow-missing to score the filled rows only"
        )
    if blank:
        print(f"[score] WARNING: {len(blank)} unfilled row(s) excluded: {blank}")
    if not human_records:
        raise SystemExit("no labeled rows to score")

    stats = judge_agreement(judge_records, human_records)
    by_type = Counter(r["question_type"] for r in rows)
    dis_ids = set(stats["disagreements"])
    disagreements = [
        {
            "question_id": r["question_id"],
            "question_type": r["question_type"],
            "is_abstention": r.get("is_abstention", ""),
            "question": r.get("question", ""),
            "gold_answer": r.get("gold_answer", ""),
            "hypothesis": r.get("hypothesis", ""),
            "judge_label": r["judge_label"],
            "human_label": r["human_label"],
            "notes": r.get("notes", ""),
        }
        for r in rows
        if r["question_id"] in dis_ids
    ]

    kappa = stats["cohens_kappa"]
    if kappa >= KAPPA_THRESHOLD:
        verdict = (
            f"VALIDATED: kappa = {kappa} >= {KAPPA_THRESHOLD} — the local judge may be "
            "used for headline numbers per plan §5.2/D6"
        )
    else:
        verdict = (
            f"NOT VALIDATED: kappa = {kappa} < {KAPPA_THRESHOLD} — run one judge-prompt "
            "iteration round (max two total), then escalate to D6-A (official judge)"
        )

    report = {
        "worksheet": str(ws_path),
        "n": stats["n"],
        "n_rows_total": len(rows),
        "n_unfilled_excluded": len(blank),
        "agreement": stats["agreement"],
        "cohens_kappa": kappa,
        "kappa_threshold": KAPPA_THRESHOLD,
        "verdict": verdict,
        "stratification": dict(sorted(by_type.items())),
        "disagreements": disagreements,
    }
    tag = ws_path.stem.replace("worksheet_", "")
    out_dir = ws_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"report_{tag}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "disagreements"}, indent=2))
    print(f"[score] {len(disagreements)} disagreement(s) with full context -> {report_path}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--emit-worksheet",
        action="store_true",
        help="sample judged records and write a hand-labeling worksheet",
    )
    mode.add_argument(
        "--score",
        dest="worksheet",
        metavar="WORKSHEET_CSV",
        help="score a filled worksheet: agreement, Cohen's kappa, verdict",
    )
    ap.add_argument(
        "--eval-files",
        nargs="+",
        help="judged run output file(s), e.g. results/runs/<id>/eval-results-*.jsonl "
        "(first occurrence of a question_id wins)",
    )
    ap.add_argument("--n", type=int, default=100, help="stratified sample size (default 100)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="g1", help="worksheet/report name suffix")
    ap.add_argument("--config", default=None, help="config yaml for the benchmark file path")
    ap.add_argument(
        "--allow-missing", action="store_true", help="score only rows with a filled human_label"
    )
    args = ap.parse_args()

    if args.emit_worksheet:
        if not args.eval_files:
            ap.error("--emit-worksheet requires --eval-files")
        emit_worksheet(args)
    else:
        score_worksheet(args)


if __name__ == "__main__":
    main()
