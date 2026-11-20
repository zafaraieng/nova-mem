#!/usr/bin/env python3
"""Phase-2 data preparation: download, checksum, split, stats, data/README.md.

Idempotent: downloads only missing files; split is written once (never silently
regenerated). Outputs:
    data/<file>.sha256          checksum manifests
    data/splits/dev_test_v0.json committed dev-100/test-400 ids
    data/stats_<split>.json     dataset + per-split statistics
    data/README.md              provenance documentation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nova_mem.data import (  # noqa: E402
    SPLIT_SEED,
    compute_stats,
    load_benchmark,
    save_split,
    split_report,
    stratified_split,
    stats_dict,
)
from nova_mem.utils import sha256_file  # noqa: E402

HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
FILES = {
    "longmemeval_oracle.json": f"{HF_BASE}/longmemeval_oracle.json",
    "longmemeval_s_cleaned.json": f"{HF_BASE}/longmemeval_s_cleaned.json",
    # M (2.55 GB) is optional and not downloaded by default (H3/E8 only):
    "longmemeval_m_cleaned.json": f"{HF_BASE}/longmemeval_m_cleaned.json",
}
DEFAULT_FILES = ["longmemeval_oracle.json", "longmemeval_s_cleaned.json"]
DATA = REPO / "data"


def download(files: list[str]) -> None:
    import requests

    for name in files:
        path = DATA / name
        if path.exists() and path.stat().st_size > 0:
            print(f"[skip] {name} already present ({path.stat().st_size/1e6:.1f} MB)")
            continue
        url = FILES[name]
        print(f"[get ] {url}")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = path.with_suffix(path.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 22):
                    f.write(chunk)
        tmp.rename(path)
        print(f"[done] {name}: {path.stat().st_size/1e6:.1f} MB")


def checksum(files: list[str]) -> dict:
    manifest = {}
    for name in files:
        digest = sha256_file(DATA / name)
        manifest[name] = digest
        (DATA / f"{name}.sha256").write_text(f"{digest}  {name}\n", encoding="utf-8")
        print(f"[sha ] {name}: {digest}")
    return manifest


def make_split(benchmark: list, dev_size: int = 100) -> dict:
    split_path = DATA / "splits" / "dev_test_v0.json"
    if split_path.exists():
        from nova_mem.data import load_split

        split = load_split(split_path)
        print(
            f"[skip] split exists at {split_path} (dev={len(split['dev'])}, test={len(split['test'])})"
        )
        return split
    split = stratified_split(benchmark, dev_size=dev_size, seed=SPLIT_SEED)
    save_split(
        split,
        split_path,
        meta={
            "created": "2026-11-07",
            "protocol": "stratified by question_type; dev for all tuning; test touched once",
            "benchmark": "longmemeval_s_cleaned.json",
        },
    )
    print(f"[new ] split written: dev={len(split['dev'])}, test={len(split['test'])}")
    return split


def write_readme(manifest: dict, stats, split, report) -> None:
    from nova_mem.data import stats_markdown

    abs_counts = {k: sum(1 for qid in v if qid.endswith("_abs")) for k, v in split.items()}
    readme = DATA / "README.md"
    lines = [
        "# Data — provenance, integrity, and preprocessing",
        "",
        "## Source and license",
        "",
        "- **Benchmark:** LongMemEval (`longmemeval-cleaned`), Wu et al., ICLR 2025.",
        "  - HF: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned (MIT).",
        "  - Official code: https://github.com/xiaowu0162/LongMemEval (MIT).",
        "  - The **original** `LONGMEMEVAL` dataset is deprecated by its authors (noisy",
        "    history sessions removed in the 2025-09 cleaning). All published 2024–2025",
        "    numbers refer to the pre-cleaning release; our results are on the cleaned",
        "    release and are NOT directly comparable to them (stated in every table).",
        "- **Not redistributed here.** Files are downloaded by `scripts/prepare_data.py`.",
        "- **Privacy:** public benchmark data only — synthetic evidence sessions plus",
        "  public filler conversations (ShareGPT Apache-2.0, UltraChat MIT, and simulated",
        "  sessions; 25/25/50 mixture per the paper). No private conversations.",
        "",
        "## Files and SHA-256 checksums",
        "",
        "| File | Size | SHA-256 |",
        "|---|---|---|",
    ]
    import os

    for name, digest in manifest.items():
        size = (DATA / name).stat().st_size
        lines.append(f"| `{name}` | {size/1e6:.1f} MB | `{digest}` |")
    lines += [
        "",
        "Checksum manifests also live next to each file (`<name>.sha256`). The frozen",
        "config (`configs/frozen_v0.yaml`) records the same hashes.",
        "",
        "## Preprocessing",
        "",
        "**None beyond loading.** The benchmark is never modified: question ids, types,",
        "answers, evidence labels, session ids, dates, and message content are preserved",
        "verbatim. Parsing details: sessions are reconstructed by zipping",
        "`haystack_dates`, `haystack_session_ids`, `haystack_sessions` (the official",
        "convention); turn corpus ids use the official `<session_id>_<n>` 1-offset;",
        "abstention questions are identified by `_abs` in `question_id` (official;",
        "their `question_type` stays the base type — the literal value `abstention`",
        "never occurs); all evidence ids use the `answer_` prefix in the cleaned",
        "release (the loader also normalizes legacy `noans_` prefixes defensively",
        "for official-corpus compatibility); abstention questions are excluded from",
        "retrieval metrics per the official convention.",
        "",
        "## Dev/test split (committed, never regenerated)",
        "",
        f"- Seed {SPLIT_SEED}, stratified by `question_type`; dev-100 / test-400.",
        f"- Abstention (`_abs`) questions: dev {abs_counts.get('dev', 0)} / "
        f"test {abs_counts.get('test', 0)}.",
        "- dev: all tuning, judge validation, sensitivity sweeps. test: touched once",
        "  per frozen config, after gate G2. Ids: `splits/dev_test_v0.json`.",
        "",
        "## Dataset statistics (S split, cleaned)",
        "",
        stats_markdown(stats),
        "",
        "### Split composition",
        "",
        "```json",
        json.dumps(report, indent=2, ensure_ascii=False),
        "```",
        "",
        "### Oracle file",
        "",
        "`longmemeval_oracle.json` contains the same 500 questions with evidence-only",
        "histories; abstention questions carry `answer_*_abs` pseudo-evidence sessions",
        "(30 of 500, identified by the `_abs` id suffix, base question_type preserved).",
        "Used for the oracle-context reference system and the abstention analysis.",
        "",
        "## Storage",
        "",
        "Oracle ~15 MB + S ~277 MB (+ optional M ~2.6 GB, H3/E8 only). Caches and",
        "indexes under `data/cache/` are regenerable and not committed.",
    ]
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[doc ] {readme}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES, help="subset of files to fetch")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "splits").mkdir(exist_ok=True)

    if not args.skip_download:
        download(args.files)
    for name in args.files:
        if not (DATA / name).exists():
            sys.exit(f"Missing {name}; run without --skip-download first.")
    manifest = checksum(args.files)

    if "longmemeval_s_cleaned.json" in args.files:
        benchmark = load_benchmark(DATA / "longmemeval_s_cleaned.json")
        stats = compute_stats(benchmark)
        split = make_split(benchmark)
        report = split_report(benchmark, split)
        (DATA / "stats_s_cleaned.json").write_text(
            json.dumps({"dataset": stats_dict(stats), "split_report": report}, indent=2) + "\n",
            encoding="utf-8",
        )
        write_readme(manifest, stats, split, report)
        print(f"[stat] {stats.n_questions} questions; types: {stats.by_type}")
        print(f"[stat] session reuse ratio: {stats.session_reuse_ratio}")
    print("[ok  ] prepare_data complete")


if __name__ == "__main__":
    main()
