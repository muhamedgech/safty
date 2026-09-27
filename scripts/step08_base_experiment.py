"""Step 8: base-model validity (paper §4.1-4.2, Tables 1 and 1b).

PACT (English-fit probe, target-language calibration) vs target-native training on
AdvBench, and the label-shift stress test (target compliance rate resampled to pi).

Run from the project root:
    python scripts/step08_base_experiment.py --partitions 5    # quick try first
    python scripts/step08_base_experiment.py                   # full run (config: 100)

Output (results/base_<label>/): partitions.csv, summary.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from joblib import Parallel, delayed  # noqa: E402

from pact.config import DEFAULT_CONFIG, load_config  # noqa: E402
from pact.experiments import load_base_data, run_base_partition  # noqa: E402
from pact.paths import results_dir  # noqa: E402
from pact.report import print_table, summarize, write_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--partitions", type=int)
    parser.add_argument("--label-method", choices=["judge", "keyword"])
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    config = load_config(args.config)
    ecfg = config["base_experiment"]
    method = args.label_method or ecfg["label_method"]
    out = results_dir(config, f"base_{method}")

    data = load_base_data(config, method)
    print(f"{len(data.shared_uids)} shared prompts (English complies with {data.z.sum()}), "
          f"{len(data.extra_z)} English-only prompts for fitting")
    n_partitions = args.partitions or ecfg["n_partitions"]
    per_partition = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_base_partition)(data, config, p) for p in range(n_partitions))
    rows = [r for part in per_partition for r in part]
    write_csv(rows, out / "partitions.csv")

    n = len(data.shared_uids)
    n_train, n_test = round(config["split"]["fractions"][0] * n), round(config["split"]["fractions"][2] * n)
    summary = summarize(rows, ["method", "lang", "pi"], ["miss_2state", "flagged_fraction", "auc_z"], n_train, n_test)
    write_csv(summary, out / "summary.csv")

    print_table([r for r in summary if r["pi"] is None],
                [("method", "method"), ("lang", "lang"), ("miss rate", "miss_2state+ci"),
                 ("flagged", "flagged_fraction"), ("AUC vs z", "auc_z")],
                f"Table 1: miss rate (target alpha={config['conformal']['alpha_miss']}, marginal)")
    print_table(sorted([r for r in summary if r["pi"] is not None], key=lambda r: (r["lang"], r["pi"])),
                [("lang", "lang"), ("pi", "pi"), ("miss rate", "miss_2state+ci")],
                "Table 1b: label shift (PACT)")
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
