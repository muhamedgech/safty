"""Step 6: prove the threshold code is right before any real result depends on it.

The paper found an off-by-one bug here only after reporting results. This script
checks the code against theory on simulated scores, and needs no GPU or data.

Run from the project root:
    python scripts/step06_verify_conformal.py

Output:
    results/conformal_check.csv   one row per check, with pass/fail
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pact.config import DEFAULT_CONFIG, load_config  # noqa: E402
from pact.conformal import (choose_k, simulate_flag_threshold, simulate_pass_threshold,  # noqa: E402
                            two_state_auc_floor)
from pact.paths import results_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--trials", type=int, default=20000)
    args = parser.parse_args()
    config = load_config(args.config)
    ccfg = config["conformal"]

    rows = []
    checks = [("pass", ccfg["alpha_miss"], n) for n in (48, 80, 101, 200)] + \
             [("flag", ccfg["alpha_flag"], n) for n in (40, 78, 200)]
    for side, alpha, n in checks:
        simulate = simulate_pass_threshold if side == "pass" else simulate_flag_threshold
        # Marginal: mean true error rate must equal k/(n+1) (within Monte Carlo error).
        k = choose_k(n, alpha, None)
        rates = simulate(n, alpha, None, args.trials, config["seed"])
        expected = k / (n + 1)
        tolerance = 4 * rates.std() / len(rates) ** 0.5 + 1e-9
        rows.append({"side": side, "regime": "marginal", "n": n, "alpha": alpha, "delta": "", "k": k,
                     "theory": round(expected, 4), "simulated": round(rates.mean(), 4),
                     "ok": abs(rates.mean() - expected) <= tolerance})
        # PAC: the share of calibration draws whose true error exceeds alpha must be <= delta.
        for delta in ccfg["deltas"]:
            k = choose_k(n, alpha, delta / 2)  # delta/2 per threshold, as in the paper
            if k == 0:
                rows.append({"side": side, "regime": "pac", "n": n, "alpha": alpha, "delta": delta, "k": 0,
                             "theory": "too few calibration points", "simulated": "", "ok": True})
                continue
            rates = simulate(n, alpha, delta / 2, args.trials, config["seed"])
            violation = (rates > alpha).mean()
            tolerance = 4 * (delta / 2 * (1 - delta / 2) / len(rates)) ** 0.5
            rows.append({"side": side, "regime": "pac", "n": n, "alpha": alpha, "delta": delta, "k": k,
                         "theory": f"<= {delta / 2}", "simulated": round(violation, 4),
                         "ok": violation <= delta / 2 + tolerance})

    out = results_dir(config, "") / "conformal_check.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'side':<6}{'regime':<10}{'n':>5}{'alpha':>7}{'delta':>7}{'k':>4}  {'theory':<28}{'simulated':<11}ok")
    for r in rows:
        print(f"{r['side']:<6}{r['regime']:<10}{r['n']:>5}{r['alpha']:>7}{r['delta']!s:>7}{r['k']:>4}  "
              f"{r['theory']!s:<28}{r['simulated']!s:<11}{r['ok']}")
    floor = two_state_auc_floor(ccfg["alpha_miss"], ccfg["alpha_flag"])
    print(f"\ntwo-state certificate needs AUC >= {floor:.3f}")
    print(f"saved {out}")
    if not all(r["ok"] for r in rows):
        sys.exit("SOME CHECKS FAILED: do not trust any certificate until this is fixed")


if __name__ == "__main__":
    main()
