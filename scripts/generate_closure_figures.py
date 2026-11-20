#!/usr/bin/env python3
"""Plot descriptive closure summaries; this does not compute inference."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "results" / "closure_summary.csv",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "figures" / "closure_summary.png")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with args.summary.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {"A", "Oracle", "C BM25", "D BM25 rerank"}
    if {row["condition"] for row in rows} != expected:
        raise ValueError(f"summary must contain exactly these conditions: {sorted(expected)}")

    colors = {"A": "#4C78A8", "Oracle": "#F58518", "C BM25": "#54A24B", "D BM25 rerank": "#E45756"}
    fig, (accuracy_ax, recall_ax) = plt.subplots(1, 2, figsize=(11, 5.5))
    ordered = ["A", "Oracle", "C BM25", "D BM25 rerank"]
    summary_by_condition = {row["condition"]: row for row in rows}

    for y, condition in enumerate(ordered):
        row = summary_by_condition[condition]
        value = float(row["overall_accuracy"])
        accuracy_ax.scatter(value, y, color=colors[condition], s=65, zorder=3)
        accuracy_ax.annotate(
            f"{value:.2f}", (value, y), xytext=(7, 0), textcoords="offset points", va="center"
        )
    accuracy_ax.set_yticks(range(len(ordered)), ordered)
    accuracy_ax.invert_yaxis()
    accuracy_ax.set_xlim(0, 0.56)
    accuracy_ax.set_xlabel("Local-judge accuracy, official prompts (dev-100)")
    accuracy_ax.set_title("Answer accuracy\npoint estimates only")
    accuracy_ax.grid(axis="x", alpha=0.25)

    for condition in ["Oracle", "C BM25", "D BM25 rerank"]:
        row = summary_by_condition[condition]
        recall_ax.plot(
            [5, 10],
            [float(row["session_recall_at5"]), float(row["session_recall_at10"])],
            marker="o",
            linewidth=1.8,
            color=colors[condition],
            label=condition,
        )
    recall_ax.set_xticks([5, 10], ["@5", "@10"])
    recall_ax.set_xlim(4.5, 10.5)
    recall_ax.set_ylim(0.85, 1.02)
    recall_ax.set_ylabel("Session recall (n=94)")
    recall_ax.set_title("Retrieval evidence recall\nnot answer correctness")
    recall_ax.grid(alpha=0.25)
    recall_ax.legend(frameon=False, loc="lower right")

    fig.suptitle("NOVA-Mem development results: descriptive, not a ranking", fontsize=13)
    fig.text(
        0.5,
        0.02,
        "Oracle/C question dates differ; A used CPU while Oracle/C/D used T4; "
        "D is cache-resumed. No paired significance or causal comparison is shown.",
        ha="center",
        va="bottom",
        fontsize=8,
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
