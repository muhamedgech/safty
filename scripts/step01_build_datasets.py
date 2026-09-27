"""Step 1: build the English prompt datasets and save them for every later step.

Run from the project root:
    python scripts/step01_build_datasets.py
    python scripts/step01_build_datasets.py --sources advbench xstest   # only some sources

Output:
    data/raw/...                         the original downloaded files (cached)
    data/processed/en/<source>.jsonl     one cleaned record per line
    data/processed/manifest.json         counts + sha256 fingerprint of every file
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pact.config import DEFAULT_CONFIG, load_config  # noqa: E402
from pact.data.build import build_source, save_sources  # noqa: E402
from pact.data.sources import LOADERS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--sources", nargs="+", choices=list(LOADERS), default=list(LOADERS))
    args = parser.parse_args()

    config = load_config(args.config)
    datasets = {}
    for name in args.sources:
        print(f"building {name} ...")
        datasets[name] = build_source(name, config)

    manifest = save_sources(datasets, config)

    print(f"\n{'source':<20}{'count':>6}{'safe':>6}{'unsafe':>8}{'translate':>11}")
    for name, records in datasets.items():
        labels = Counter(r.gold_label for r in records)
        n_translate = sum(r.translate for r in records)
        print(f"{name:<20}{len(records):>6}{labels['safe']:>6}{labels['unsafe']:>8}{n_translate:>11}")
    print(f"\nsaved to {config['paths']['processed_dir']}")
    for file, info in manifest["files"].items():
        print(f"  {file:<32} sha256 {info['sha256'][:12]}")


if __name__ == "__main__":
    main()
