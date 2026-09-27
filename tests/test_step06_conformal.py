"""Step 6 tests: the conformal thresholds against exact theory."""
import numpy as np
import pytest

from pact.conformal import (DEFER, FLAG, PASS, flag_threshold, marginal_k, pac_k, pass_threshold,
                            simulate_pass_threshold, three_state, two_state, two_state_auc_floor)


def test_pass_threshold_is_kth_smallest_counting_from_one():
    """Guards against the paper's Bug 1 (sorted[k] instead of sorted[k-1])."""
    scores = np.arange(1.0, 49.0)            # n = 48 -> k = floor(49 * 0.05) = 2
    assert marginal_k(48, 0.05) == 2
    assert pass_threshold(scores, 0.05) == 2.0


def test_flag_threshold_is_kth_largest():
    scores = np.arange(1.0, 41.0)            # n = 40 -> k = floor(41 * 0.1) = 4
    assert flag_threshold(scores, 0.10) == 37.0


def test_pac_k_matches_paper_numbers():
    assert pac_k(101, 0.05, 0.025) == 1      # paper §4.5: refuse class, k = 1
    assert pac_k(78, 0.10, 0.025) == 3       # comply class, k = 3
    assert pac_k(72, 0.05, 0.025) == 1 and pac_k(71, 0.05, 0.025) == 0   # minimum n = 72
    assert pac_k(36, 0.10, 0.025) == 1 and pac_k(35, 0.10, 0.025) == 0   # minimum n = 36


def test_too_small_calibration_set_never_passes():
    assert pass_threshold(np.random.rand(40), 0.05, delta=0.025) == -np.inf
    assert flag_threshold(np.random.rand(30), 0.10, delta=0.025) == np.inf


def test_marginal_miss_rate_matches_theory():
    rates = simulate_pass_threshold(48, 0.05, None, trials=4000)
    assert rates.mean() == pytest.approx(2 / 49, abs=0.004)


def test_pac_violation_rate_below_delta():
    rates = simulate_pass_threshold(101, 0.05, 0.025, trials=4000)
    assert (rates > 0.05).mean() <= 0.025 + 0.01


def test_decisions():
    scores = np.array([0.1, 0.5, 0.9])
    assert list(two_state(scores, 0.5)) == [PASS, FLAG, FLAG]
    assert list(three_state(scores, 0.3, 0.7)) == [PASS, DEFER, FLAG]
    with pytest.raises(ValueError, match="cross"):
        three_state(scores, 0.7, 0.3)


def test_auc_floor():
    assert two_state_auc_floor(0.05, 0.10) == pytest.approx(0.855)
