import numpy as np
import pytest
from scipy import stats

from hvac.stats import HIT, MISS, PARTIAL, UNDECIDED, judge, mean_ci, sd_ratio_test, welch_ci

RNG = np.random.default_rng(0)
BASE = 4.7 + 0.05 * RNG.standard_normal(10)


def test_welch_ci_matches_scipy():
    x = 5.0 + 0.2 * RNG.standard_normal(10)
    d, lo, hi = welch_ci(x, BASE, 0.95)
    ref = stats.ttest_ind(x, BASE, equal_var=False).confidence_interval(0.95)
    assert d == pytest.approx(x.mean() - BASE.mean())
    assert (lo, hi) == pytest.approx((ref.low, ref.high))


def test_mean_ci_contains_mean():
    m, lo, hi = mean_ci(BASE)
    assert lo < m < hi


def test_range():
    assert judge("range", np.full(10, 5.0) + 0.01 * np.arange(10), BASE, bounds=(4.8, 5.2))["verdict"] == HIT
    assert judge("range", np.full(10, 6.0) + 0.01 * np.arange(10), BASE, bounds=(4.8, 5.2))["verdict"] == MISS
    wide = np.array([4.0, 6.0] * 5)  # mean 5.0, wide CI
    assert judge("range", wide, BASE, bounds=(5.1, 5.3))["verdict"] == PARTIAL
    assert judge("range", np.full(10, 0.01), BASE, bounds=(-np.inf, 0.05))["verdict"] == HIT


def test_direction():
    assert judge("higher", BASE + 0.5, BASE)["verdict"] == HIT
    assert judge("higher", BASE - 0.5, BASE)["verdict"] == MISS
    assert judge("lower", BASE - 0.5, BASE)["verdict"] == HIT
    assert judge("higher", BASE[::-1], BASE)["verdict"] == UNDECIDED
    assert judge("much_higher", BASE + 0.5, BASE, delta=0.1)["verdict"] == HIT
    assert judge("much_higher", BASE + 0.02, BASE, delta=0.1)["verdict"] == MISS


def test_equivalence_needs_a_tight_interval():
    assert judge("same", BASE[::-1] + 0.01, BASE, delta=0.1)["verdict"] == HIT
    assert judge("same", BASE + 0.5, BASE, delta=0.1)["verdict"] == MISS
    noisy = BASE.mean() + 0.3 * RNG.standard_normal(10)
    noisy += BASE.mean() - noisy.mean()  # same mean, much wider spread
    assert judge("same", noisy, BASE, delta=0.1)["verdict"] == UNDECIDED


def test_non_inferiority():
    assert judge("not_higher", BASE - 0.2, BASE, delta=0.1)["verdict"] == HIT
    assert judge("not_higher", BASE + 0.5, BASE, delta=0.1)["verdict"] == MISS
    assert judge("not_lower", BASE + 0.2, BASE, delta=0.1)["verdict"] == HIT


def test_spread():
    wide = BASE.mean() + 5 * (BASE - BASE.mean())
    ratio, p = sd_ratio_test(wide, BASE)
    assert ratio == pytest.approx(5.0) and p < 0.05
    assert judge("sd_larger", wide, BASE)["verdict"] == HIT
    assert judge("sd_larger", BASE, wide)["verdict"] == MISS
    assert judge("sd_larger", BASE[::-1], BASE)["verdict"] == UNDECIDED
