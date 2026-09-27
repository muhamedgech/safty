"""Step 2: translate the prompts marked `translate` into every target language.

Run from the project root:
    python scripts/step02_translate.py                 # all languages, all sources
    python scripts/step02_translate.py --langs am      # one language
    python scripts/step02_translate.py --force         # redo files that already exist

Output:
    data/processed/<lang>/<source>.jsonl        translated prompts (same uid as English)
    data/processed/<lang>-back/<source>.jsonl   round trip back to English (sources in
                                                translation.back_translate), for the audit
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pact.config import DEFAULT_CONFIG, load_config  # noqa: E402
from pact.data.build import load_source, save_prompts  # noqa: E402
from pact.paths import prompts_path  # noqa: E402
from pact.translate import Translator, check_translation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--langs", nargs="+")
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    tcfg = config["translation"]
    langs = args.langs or list(config["languages"])
    sources = args.sources or list(config["sources"])
    translator = None

    for source in sources:
        english = [r for r in load_source(source, config) if r.translate]
        if not english:
            continue
        for lang in langs:
            nllb = config["languages"][lang]
            out_path = prompts_path(config, lang, source)
            if out_path.exists() and not args.force:
                print(f"{lang}/{source}: exists, skipping (use --force to redo)")
                translated = load_source(source, config, lang)
            else:
                translator = translator or Translator(tcfg["model"])
                print(f"{lang}/{source}: translating {len(english)} prompts")
                texts = translator([r.text for r in english], tcfg["source_lang"], nllb,
                                   tcfg["batch_size"], tcfg["max_new_tokens"])
                print(f"  check: {check_translation([r.text for r in english], texts, lang)}")
                translated = [replace(r, text=t) for r, t in zip(english, texts)]
                save_prompts(translated, config, lang, source)

            if source not in tcfg["back_translate"]:
                continue
            back_lang = f"{lang}-back"
            if prompts_path(config, back_lang, source).exists() and not args.force:
                print(f"{back_lang}/{source}: exists, skipping")
                continue
            translator = translator or Translator(tcfg["model"])
            print(f"{back_lang}/{source}: back-translating {len(translated)} prompts")
            texts = translator([r.text for r in translated], nllb, tcfg["source_lang"],
                               tcfg["batch_size"], tcfg["max_new_tokens"])
            save_prompts([replace(r, text=t) for r, t in zip(translated, texts)], config, back_lang, source)


if __name__ == "__main__":
    main()
