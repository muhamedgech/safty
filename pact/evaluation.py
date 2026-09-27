"""Splits, metrics and uncertainty used by the experiments."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import roc_auc_score

from pact.conformal import DEFER, FLAG, PASS, flag_threshold, pass_threshold, three_state, two_state

FIT, CAL, TEST = "fit", "cal", "test"


def stratified_partition(strata: list, fractions: tuple[float, float, float], seed: int) -> np.ndarray:
    """Assign each item to fit / cal / test, keeping every stratum's proportions."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, s in enumerate(strata):
        groups[s].append(i)
    assignment = np.empty(len(strata), dtype=object)
    for key in sorted(groups, key=str):
        idx = rng.permutation(groups[key])
        n_fit = round(fractions[0] * len(idx))
        n_cal = round(fractions[1] * len(idx))
        assignment[idx[:n_fit]] = FIT
        assignment[idx[n_fit:n_fit + n_cal]] = CAL
        assignment[idx[n_fit + n_cal:]] = TEST
    return assignment


def auc(scores: np.ndarray, is_refuse: np.ndarray) -> float:
    """P(score of a refusal > score of a compliance). nan if one class is missing."""
    is_refuse = np.asarray(is_refuse).astype(int)
    if is_refuse.min() == is_refuse.max():
        return float("nan")
    return float(roc_auc_score(is_refuse, scores))


def _rate(mask: np.ndarray, within: np.ndarray) -> float:
    return float(mask[within].mean()) if within.any() else float("nan")


def certify(scores: np.ndarray, z: np.ndarray, gold_refuse: np.ndarray, cal: np.ndarray, test: np.ndarray,
            alpha_miss: float, alpha_flag: float, delta: float | None,
            frontier_alphas: list[float] = ()) -> dict:
    """Calibrate on `cal`, evaluate on `test`. z: 1 comply / 0 refuse.

    delta=None gives marginal thresholds; otherwise PAC with delta/2 per threshold
    (union bound: both hold together with probability >= 1-delta).
    """
    per_threshold = None if delta is None else delta / 2
    cal_refuse, cal_comply = scores[cal & (z == 0)], scores[cal & (z == 1)]
    s, zt = scores[test], z[test]
    refuse, comply = zt == 0, zt == 1

    tau_pass = pass_threshold(cal_refuse, alpha_miss, per_threshold)
    tau_flag = flag_threshold(cal_comply, alpha_flag, per_threshold)
    two = two_state(s, tau_pass)
    row = {
        "auc_z": auc(s, refuse),
        "auc_gold": auc(s, gold_refuse[test]),
        "orientation_ok": bool(cal_refuse.mean() > cal_comply.mean()),
        "n_cal_refuse": int(len(cal_refuse)),
        "n_cal_comply": int(len(cal_comply)),
        "miss_2state": _rate(two == PASS, refuse),
        "false_flag_2state": _rate(two == FLAG, comply),
        "flagged_fraction": float((two == FLAG).mean()),
        "feasible_3state": bool(tau_pass <= tau_flag),
        "miss_3state": float("nan"), "false_flag_3state": float("nan"), "deferral": float("nan"),
    }
    if row["feasible_3state"]:
        three = three_state(s, tau_pass, tau_flag)
        row["miss_3state"] = _rate(three == PASS, refuse)
        row["false_flag_3state"] = _rate(three == FLAG, comply)
        row["deferral"] = float((three == DEFER).mean())
    for a in frontier_alphas:  # two-state false-flag rate at looser miss targets (Table 3')
        row[f"frontier_{a}"] = _rate(two_state(s, pass_threshold(cal_refuse, a, per_threshold)) == FLAG, comply)
    return row


def corrected_ci(values, n_train: int, n_test: int, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and CI over repeated random partitions of the SAME data (Nadeau & Bengio, 2003).

    Partitions share prompts, so they are correlated and the naive sd/sqrt(J) interval is
    far too narrow. The corrected variance is (1/J + n_test/n_train) * s^2.
    """
    values = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    J = len(values)
    if J < 2:
        return (float(values.mean()) if J else float("nan"), float("nan"), float("nan"))
    mean = values.mean()
    se = np.sqrt((1 / J + n_test / n_train) * values.var(ddof=1))
    half = student_t.ppf(0.5 + level / 2, J - 1) * se
    return float(mean), float(mean - half), float(mean + half)
