"""Step 5: save the hidden state of each prompt's last token (one forward pass, no generation).

Run from the project root:
    python scripts/step05_extract_features.py
    python scripts/step05_extract_features.py --models base --force

Output:
    data/features/<model>/<lang>/<source>.npz   uids + layer_<L> arrays of shape (n_prompts, hidden_size)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pact.config import DEFAULT_CONFIG, load_config  # noqa: E402
from pact.models import load_model  # noqa: E402
from pact.paths import features_path  # noqa: E402
from pact.pipeline import iter_jobs, run_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    jobs = [j for j in iter_jobs(config["features"], args.models)
            if args.force or not features_path(config, *j).exists()]
    if not jobs:
        print("all features already extracted (use --force to redo)")
    for model_key in dict.fromkeys(m for m, _, _ in jobs):
        print(f"loading {model_key}: {config['models'][model_key]['path']}")
        tokenizer, model = load_model(config["models"][model_key])
        for _, lang, source in (j for j in jobs if j[0] == model_key):
            run_features(config, tokenizer, model, model_key, lang, source)
        del model


if __name__ == "__main__":
    main()
