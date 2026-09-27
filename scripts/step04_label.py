"""Step 4: label every response as comply (1) or refuse (0).

Run from the project root:
    python scripts/step04_label.py               # keyword + LLM judge
    python scripts/step04_label.py --no-judge    # keyword only (fast, no GPU)

Output:
    data/labels/<model>/<lang>/<source>.jsonl   {uid, keyword, judge, judge_class}
The English label of each prompt is the transported label z used from step 7 on.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pact.config import DEFAULT_CONFIG, load_config  # noqa: E402
from pact.models import load_model  # noqa: E402
from pact.paths import labels_path, responses_path  # noqa: E402
from pact.pipeline import iter_jobs, run_labeling  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    judge = None
    if not args.no_judge:
        judge_key = config["labeling"]["judge_model"]
        print(f"loading judge: {config['models'][judge_key]['path']}")
        judge = load_model(config["models"][judge_key])

    print(f"{'job':<36}{'n':>5}{'keyword comply':>16}{'judge comply':>14}{'judge unparsed':>16}")
    for model_key, lang, source in iter_jobs(config["generate"], args.models):
        job = f"{model_key}/{lang}/{source}"
        if not responses_path(config, model_key, lang, source).exists():
            print(f"{job:<36} no responses yet (run step 3)")
            continue
        if labels_path(config, model_key, lang, source).exists() and not args.force:
            print(f"{job:<36} already labeled (use --force to redo)")
            continue
        rows = run_labeling(config, model_key, lang, source, judge)
        kw = Counter(r["keyword"] for r in rows)
        jd = Counter(r["judge"] for r in rows)
        print(f"{job:<36}{len(rows):>5}{kw[1]:>16}{jd[1]:>14}{jd[None] if judge else '-':>16}")


if __name__ == "__main__":
    main()
