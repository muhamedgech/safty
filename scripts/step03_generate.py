"""Step 3: let each model answer the prompts (greedy decoding).

Run from the project root:
    python scripts/step03_generate.py                    # every job in config['generate']
    python scripts/step03_generate.py --models instruct  # one model

Output:
    data/responses/<model>/<lang>/<source>.jsonl    {uid, prompt, response}
If interrupted, run it again: finished prompts are skipped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pact.config import DEFAULT_CONFIG, load_config  # noqa: E402
from pact.models import load_model  # noqa: E402
from pact.pipeline import iter_jobs, run_generation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()

    config = load_config(args.config)
    jobs = list(iter_jobs(config["generate"], args.models))
    for model_key in dict.fromkeys(m for m, _, _ in jobs):  # load each model once
        print(f"loading {model_key}: {config['models'][model_key]['path']}")
        tokenizer, model = load_model(config["models"][model_key])
        for _, lang, source in (j for j in jobs if j[0] == model_key):
            run_generation(config, tokenizer, model, model_key, lang, source)
        del model


if __name__ == "__main__":
    main()
