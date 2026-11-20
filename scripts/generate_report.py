#!/usr/bin/env python3
"""Generate figures from real ledger outputs (never from hand-entered numbers).

Figures (Phase 7): accuracy-vs-cost frontier, per-category accuracy, latency
breakdown per system. All matplotlib, English labels, constrained_layout,
legends outside plots.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", default=str(REPO / "results" / "ledger.jsonl"))
    ap.add_argument("--outdir", default=str(REPO / "figures"))
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = [json.loads(l) for l in Path(args.ledger).read_text().splitlines() if l.strip()]
    records = [r for r in records if not any("mock-backend" in d for d in r.get("deviations", []))]
    if not records:
        sys.exit("No real ledger records; nothing to plot (mock runs are excluded by design).")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- accuracy vs cost frontier --------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for r in records:
        qa = r.get("qa_metrics") or {}
        acc = qa.get("overall_accuracy")
        if acc is None:
            continue
        run_dir = REPO / r.get("run_dir", "")
        answers_path = run_dir / "answers.jsonl"
        if "cache_hits" in r:
            cache_hits = r["cache_hits"]
        elif answers_path.is_file():
            cache_hits = sum(
                cached
                for line in answers_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
                for cached in json.loads(line).get("timings", {}).values()
                if isinstance(cached, bool)
            )
        else:
            continue
        if cache_hits:
            continue
        runtime_s = r.get("generation_runtime_s", r.get("runtime_s"))
        ax.scatter(runtime_s, acc, s=48)
        ax.annotate(
            f"{r.get('system_key', '?')}\n{r['experiment_id'][:24]}",
            (runtime_s, acc),
            fontsize=6,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("generation-loop wall time (s; cache-hit runs excluded)")
    ax.set_ylabel("official judge accuracy")
    ax.set_title("Accuracy vs. compute cost (one point per run)")
    fig.savefig(outdir / "accuracy_vs_cost.png", dpi=200)

    # ---- per-category accuracy ------------------------------------------
    systems = {}
    for r in records:
        key = r.get("system_key")
        if key and r.get("qa_metrics"):
            systems[key] = r["qa_metrics"]
    if systems:
        types = sorted({t for qa in systems.values() for t in qa.get("per_type", {})})
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        width = 0.8 / max(1, len(systems))
        for i, (key, qa) in enumerate(sorted(systems.items())):
            xs = [t + i * width for t in range(len(types))]
            ys = [(qa.get("per_type", {}).get(t) or {}).get("accuracy") for t in types]
            ax.bar(xs, [y if y is not None else 0 for y in ys], width=width, label=key)
        ax.set_xticks([t + width * (len(systems) - 1) / 2 for t in range(len(types))])
        ax.set_xticklabels(
            [t.replace("single-session-", "ss-") for t in types],
            rotation=20,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel("accuracy")
        ax.set_title("Per-category official accuracy by system")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        fig.savefig(outdir / "per_category_accuracy.png", dpi=200)

    print(f"[written] figures to {outdir}")


if __name__ == "__main__":
    main()
