"""Statistics for judging the ablation predictions.

One training run gives one number per metric, so a configuration with 10 seeds
gives a sample of 10. The rules (docs/report/zh/03_ablation.md, section 3.1):

    range        prediction "between lo and hi": hit if the mean is inside,
                 miss if the 95% CI of the mean is entirely outside, else partial
    higher/lower 95% CI of (config - baseline) entirely above / below 0
    much_higher  95% CI of the difference entirely above delta
    same         90% CI of the difference inside [-delta, delta]  (equivalence)
    not_higher   upper end of the 90% CI below delta               (non-inferiority)
    not_lower    lower end of the 90% CI above -delta
    sd_larger    two-sided F test on the variances, p < 0.05

Differences use Welch's t interval, which does not assume equal variances.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

HIT, PARTIAL, MISS, UNDECIDED = "命中", "部分命中", "不命中", "无法判断"
KINDS = ("range", "higher", "lower", "much_higher", "same", "not_higher", "not_lower", "sd_larger")


def mean_ci(x, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and its t confidence interval."""
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    if len(x) < 2:
        return m, math.nan, math.nan
    half = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return m, m - half, m + half


def welch_ci(x, y, level: float = 0.95) -> tuple[float, float, float]:
    """mean(x) - mean(y) and its Welch confidence interval."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    d = float(x.mean() - y.mean())
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = math.sqrt(vx + vy)
    if se == 0.0:
        return d, d, d
    df = (vx + vy) ** 2 / (vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1))
    half = stats.t.ppf(0.5 + level / 2, df) * se
    return d, d - half, d + half


def sd_ratio_test(x, y) -> tuple[float, float]:
    """sd(x) / sd(y) and the two-sided F-test p-value for equal variances."""
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    if vx == 0.0 or vy == 0.0:
        return math.nan, math.nan
    f = vx / vy
    dfx, dfy = len(x) - 1, len(y) - 1
    p = 2.0 * min(stats.f.cdf(f, dfx, dfy), stats.f.sf(f, dfx, dfy))
    return math.sqrt(f), min(p, 1.0)


def judge(kind: str, x, base, bounds: tuple[float, float] | None = None, delta: float | None = None) -> dict:
    """Judge one prediction about sample `x` (the configuration) against `base` (the baseline)."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    out: dict = {"kind": kind}
    if kind == "range":
        lo, hi = bounds
        m, cl, ch = mean_ci(x)
        out.update(value=m, ci95=(cl, ch))
        if lo <= m <= hi:
            out["verdict"] = HIT
        elif ch < lo or cl > hi:
            out["verdict"] = MISS
        else:
            out["verdict"] = PARTIAL
        return out
    if kind == "sd_larger":
        ratio, p = sd_ratio_test(x, base)
        out.update(value=ratio, p=p)
        if math.isnan(p) or p >= 0.05:
            out["verdict"] = UNDECIDED
        else:
            out["verdict"] = HIT if ratio > 1.0 else MISS
        return out

    d, l95, h95 = welch_ci(x, base, 0.95)
    _, l90, h90 = welch_ci(x, base, 0.90)
    out.update(value=d, ci95=(l95, h95), ci90=(l90, h90))
    if kind == "higher":
        verdict = HIT if l95 > 0 else MISS if h95 < 0 else UNDECIDED
    elif kind == "lower":
        verdict = HIT if h95 < 0 else MISS if l95 > 0 else UNDECIDED
    elif kind == "much_higher":
        verdict = HIT if l95 > delta else MISS if h95 < delta else UNDECIDED
    elif kind == "same":
        if -delta < l90 and h90 < delta:
            verdict = HIT
        elif l95 > 0 or h95 < 0:
            verdict = MISS
        else:
            verdict = UNDECIDED
    elif kind == "not_higher":
        verdict = HIT if h90 < delta else MISS if l95 > 0 else UNDECIDED
    else:  # not_lower
        verdict = HIT if l90 > -delta else MISS if h95 < 0 else UNDECIDED
    out["verdict"] = verdict
    return out
