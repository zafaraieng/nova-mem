"""Statistics tests: paired bootstrap, McNemar, Wilson, Holm."""

from __future__ import annotations

import math

from nova_mem.analysis import holm_correction, mcnemar_exact, paired_bootstrap, wilson_ci


def test_bootstrap_identical():
    a = [1, 0, 1, 1, 0, 1, 0, 1]
    r = paired_bootstrap(a, a, n_resamples=2000, seed=1)
    assert r.delta == 0.0
    assert r.ci95 == (0.0, 0.0)
    assert r.p_bootstrap == 1.0


def test_bootstrap_perfect_separation():
    a = [1] * 50
    b = [0] * 50
    r = paired_bootstrap(a, b, n_resamples=2000, seed=1)
    assert r.delta == 1.0
    assert r.ci95 == (1.0, 1.0)
    # two-sided floor: 2 * (0 + 1) / (n_resamples + 1), up to 5-decimal rounding
    assert r.p_bootstrap <= round(2.0 / 2001, 5) + 1e-9


def test_bootstrap_small_delta_ci_brackets():
    # b slightly better; CI should bracket the empirical delta
    a = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    b = [1, 1, 1, 0, 1, 0, 1, 0, 1, 1]
    r = paired_bootstrap(a, b, n_resamples=5000, seed=7)
    assert r.delta == -0.2
    assert r.ci95[0] <= -0.2 <= r.ci95[1]


def test_mcnemar_exact():
    assert mcnemar_exact([1, 1], [1, 1]) == {"b": 0, "c": 0, "p_exact": 1.0}
    # b=1, c=0 -> two-sided exact binomial p = 2*0.5 = 1.0
    assert mcnemar_exact([1, 0], [0, 0])["p_exact"] == 1.0
    # b=5, c=0 -> p = 2*(0.5**5) = 0.0625
    a = [1, 1, 1, 1, 1, 0]
    b = [0, 0, 0, 0, 0, 0]
    assert abs(mcnemar_exact(a, b)["p_exact"] - 0.0625) < 1e-9
    # symmetric case b=3, c=3 -> p = 1.0
    a = [1, 1, 1, 0, 0, 0]
    b = [0, 0, 0, 1, 1, 1]
    assert mcnemar_exact(a, b)["p_exact"] == 1.0


def test_wilson_ci():
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and abs(hi - 0.2775) < 0.002  # Wilson 0/10 upper ≈ 0.277
    lo, hi = wilson_ci(5, 10)
    assert abs(lo - 0.2366) < 0.01 and abs(hi - 0.7634) < 0.01


def test_holm():
    adj = holm_correction({"x": 0.01, "y": 0.04, "z": 0.03})
    assert adj["x"] <= adj["z"] <= adj["y"]
    assert adj["x"] == round(min(1.0, 3 * 0.01), 6)
    assert adj["z"] == round(min(1.0, max(3 * 0.01, 2 * 0.03)), 6)
    assert adj["y"] == round(min(1.0, max(2 * 0.03, 1 * 0.04)), 6)
