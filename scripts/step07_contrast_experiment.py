"""Step 7: the decisive test (paper §4.3-4.5) on the instruct model.

Does PACT transport the model's refusal boundary, or only a topic boundary?
Compares topic control, PACT-A0, PACT-A2, PACT-TT (and the English ceiling) on
XSTest + OR-Bench-Hard, over many random fit/cal/test partitions.

Run from the project root:
    python scripts/step07_contrast_experiment.py --partitions 5      # quick try first
    python scripts/step07_contrast_experiment.py                     # full run (config: 200)
    python scripts/step07_contrast_experiment.py --label-method keyword
    python scripts/step07_contrast_experiment.py --stable-only       # back-translation filter

Output (results/contrast_<label>[_stable]/):
    partitions.csv   one row per partition x method x language x threshold regime
    summary.csv      means with corrected 95% CIs
    paired.csv       paired AUC differences (e.g. A0 - topic) with corrected CIs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from joblib import Parallel, delayed  # noqa: E402

from pact.config import DEFAULT_CONFIG, load_config  # noqa: E402
from pact.conformal import two_state_auc_floor  # noqa: E402
from pact.experiments import fit_topic_probe, load_contrast_data, run_contrast_partition  # noqa: E402
from pact.paths import results_dir  # noqa: E402
from pact.report import paired_differences, print_table, summarize, write_csv  # noqa: E402

METRICS = ["auc_z", "auc_gold", "miss_2state", "false_flag_2state", "flagged_fraction", "miss_3state",
           "false_flag_3state", "deferral", "orientation_ok", "feasible_3state"]
PAIRS = [("a0", "topic"), ("tt", "topic"), ("a2", "a0"), ("tt", "a0")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--partitions", type=int)
    parser.add_argument("--label-method", choices=["judge", "keyword"])
    parser.add_argument("--stable-only", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1, help="parallel partitions (-1 = all cores)")
    args = parser.parse_args()

    config = load_config(args.config)
    ecfg = config["contrast_experiment"]
    method = args.label_method or ecfg["label_method"]
    n_partitions = args.partitions or ecfg["n_partitions"]
    out = results_dir(config, f"contrast_{method}" + ("_stable" if args.stable_only else ""))

    data = load_contrast_data(config, method)
    print(f"{len(data.uids)} prompts, English complies with {data.z.sum()} (z=1), refuses {len(data.z) - data.z.sum()}")
    topic = fit_topic_probe(data, config)
    print(f"topic control: layer {topic.layer}, C={topic.C}, cv AUC {topic.cv_auc:.3f}")

    per_partition = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_contrast_partition)(data, topic, config, p, args.stable_only) for p in range(n_partitions))
    rows = [r for part in per_partition for r in part]
    write_csv(rows, out / "partitions.csv")

    n = len(data.uids)
    n_train, n_test = round(config["split"]["fractions"][0] * n), round(config["split"]["fractions"][2] * n)
    frontier = [f"frontier_{a}" for a in ecfg["frontier_alphas"]]
    summary = summarize(rows, ["method", "lang", "regime", "delta"], METRICS + frontier, n_train, n_test)
    write_csv(summary, out / "summary.csv")
    paired = paired_differences(rows, PAIRS, "auc_z", n_train, n_test)
    write_csv(paired, out / "paired.csv")

    order = {"topic": 0, "a0": 1, "a2": 2, "tt": 3, "english": 4}
    summary.sort(key=lambda r: (order[r["method"]], r["lang"]))
    main_delta = config["conformal"]["deltas"][0]
    floor = two_state_auc_floor(config["conformal"]["alpha_miss"], config["conformal"]["alpha_flag"])
    print_table([r for r in summary if r["regime"] == "marginal"],
                [("method", "method"), ("lang", "lang"), ("AUC vs z", "auc_z+ci"), ("AUC vs gold", "auc_gold"),
                 ("miss", "miss_2state"), ("flagged", "flagged_fraction"), ("orientation ok", "orientation_ok")],
                f"Discrimination and two-state miss rate, marginal thresholds "
                f"(two-state needs AUC >= {floor:.3f}; look at the CI, not the point)")
    for regime, delta in [("marginal", None), ("pac", main_delta)]:
        print_table([r for r in summary if r["regime"] == regime and r["delta"] == delta],
                    [("method", "method"), ("lang", "lang"), ("feasible", "feasible_3state"),
                     ("miss", "miss_3state"), ("false flag", "false_flag_3state"), ("deferral", "deferral+ci")],
                    f"Three-state certificate, {regime}" + (f" (delta={delta})" if delta else ""))
    print_table([r for r in summary if r["regime"] == "marginal"],
                [("method", "method"), ("lang", "lang")] + [(f"miss<={a}", f"frontier_{a}") for a in ecfg["frontier_alphas"]],
                "Feasibility frontier: two-state false-flag rate at each miss target (marginal)")
    print_table(paired, [("comparison", "comparison"), ("lang", "lang"), ("mean", "mean"), ("lo", "lo"),
                         ("hi", "hi"), ("excludes 0", "excludes_zero")],
                "Paired AUC differences (corrected 95% CI)")
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
