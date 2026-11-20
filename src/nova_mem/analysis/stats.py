"""Statistics: paired bootstrap, exact McNemar, Wilson CIs.

Design rules (reproduction_plan.md §6): all systems answer the same questions,
so only paired analyses are used; the primary family is Holm-corrected at
α=0.05; exploratory breakdowns are labeled as such.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass


@dataclass
class PairedResult:
    metric_a: float
    metric_b: float
    delta: float
    ci95: tuple[float, float]
    p_bootstrap: float
    n: int
    n_resamples: int
    seed: int


def paired_bootstrap(
    labels_a: list[int],
    labels_b: list[int],
    n_resamples: int = 10_000,
    seed: int = 42,
    metric: str = "mean",
) -> PairedResult:
    """Paired bootstrap for Δ = metric(a) − metric(b).

    ``labels_a``/``labels_b`` are paired per-question outcomes (same ordering).
    Two-sided bootstrap p: fraction of resamples with delta on the other side
    of 0 (plus a symmetric 1/(n+1) floor to avoid p=0).
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("paired arrays must have equal length")
    if not labels_a:
        raise ValueError("empty arrays")
    n = len(labels_a)
    ma = sum(labels_a) / n
    mb = sum(labels_b) / n
    delta = ma - mb

    rng = random.Random(seed)
    deltas = []
    for _ in range(n_resamples):
        sa = sb = 0
        for _ in range(n):
            i = rng.randrange(n)
            sa += labels_a[i]
            sb += labels_b[i]
        deltas.append(sa / n - sb / n)
    deltas.sort()
    lo = deltas[int(0.025 * n_resamples)]
    hi = deltas[int(0.975 * n_resamples) - 1 if int(0.975 * n_resamples) > 0 else 0]
    if delta >= 0:
        p = 2 * (sum(1 for d in deltas if d <= 0) + 1) / (n_resamples + 1)
    else:
        p = 2 * (sum(1 for d in deltas if d >= 0) + 1) / (n_resamples + 1)
    p = min(1.0, p)
    return PairedResult(
        metric_a=round(ma, 4),
        metric_b=round(mb, 4),
        delta=round(delta, 4),
        ci95=(round(lo, 4), round(hi, 4)),
        p_bootstrap=round(p, 5),
        n=n,
        n_resamples=n_resamples,
        seed=seed,
    )


def mcnemar_exact(labels_a: list[int], labels_b: list[int]) -> dict:
    """Exact McNemar test on paired binary labels.

    b = a-correct & b-wrong; c = a-wrong & b-correct. p from the exact
    binomial two-sided test on min(b, c).
    """
    b = sum(1 for x, y in zip(labels_a, labels_b) if x and not y)
    c = sum(1 for x, y in zip(labels_a, labels_b) if y and not x)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_exact": 1.0}
    k = min(b, c)
    # two-sided exact binomial: 2 * P(X <= k), X ~ Bin(n, 0.5), capped at 1
    p = 0.0
    for i in range(k + 1):
        p += math.comb(n, i) * (0.5**n)
    p = min(1.0, 2 * p)
    return {"b": b, "c": c, "p_exact": round(p, 6)}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (category accuracies)."""
    if n == 0:
        raise ValueError("n must be > 0")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(center - half, 4), round(center + half, 4))


def holm_correction(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, preserving names."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)  # enforce monotonicity
        adjusted[name] = round(running, 6)
    return adjusted


def result_dict(r: PairedResult) -> dict:
    return asdict(r)
