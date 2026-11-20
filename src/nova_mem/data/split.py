"""Stratified dev/test split with committed ids.

Protocol (reproduction_plan.md §4.2, §12): dev-100 for all tuning/judge
validation; test-400 touched once. Stratify by question_type so category
proportions match the full set. Deterministic given the seed; ids are committed
under data/splits/ and never regenerated silently.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .loader import Question

SPLIT_SEED = 42


def stratified_split(
    questions: list[Question], dev_size: int = 100, seed: int = SPLIT_SEED
) -> dict[str, list[str]]:
    """Deterministic stratified split by question_type.

    Largest remainder per category; ties broken by sorted question_id order.
    Returns {"dev": [...ids], "test": [...ids]} preserving dataset order within
    each split.
    """
    rng = random.Random(seed)
    by_type: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        by_type[q.question_type].append(q.question_id)

    n = len(questions)
    quota: dict[str, int] = {}
    raw = {t: len(ids) * dev_size / n for t, ids in by_type.items()}
    for t, v in raw.items():
        quota[t] = int(v)
    remainder = dev_size - sum(quota.values())
    # distribute remainder to largest fractional parts (stable tie-break by type name)
    order = sorted(raw, key=lambda t: (-(raw[t] - quota[t]), t))
    i = 0
    while remainder > 0:
        quota[order[i % len(order)]] += 1
        remainder -= 1
        i += 1

    dev_ids: set[str] = set()
    for t, ids in by_type.items():
        pool = sorted(ids)
        rng.shuffle(pool)
        dev_ids.update(pool[: quota[t]])

    dev = [q.question_id for q in questions if q.question_id in dev_ids]
    test = [q.question_id for q in questions if q.question_id not in dev_ids]
    assert len(dev) == dev_size, f"dev split has {len(dev)} ids, expected {dev_size}"
    return {"dev": dev, "test": test}


def save_split(split: dict[str, list[str]], path: str | Path, meta: dict | None = None) -> None:
    payload = {
        "seed": SPLIT_SEED,
        "sizes": {k: len(v) for k, v in split.items()},
        "meta": meta or {},
        "splits": split,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_split(path: str | Path) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["splits"]


def split_report(questions: list[Question], split: dict[str, list[str]]) -> dict:
    by_id = {q.question_id: q for q in questions}
    report = {}
    for name, ids in split.items():
        counts = Counter(by_id[qid].question_type for qid in ids)
        report[name] = {"n": len(ids), "by_type": dict(sorted(counts.items()))}
    return report
