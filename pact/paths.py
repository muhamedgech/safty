"""Where every step reads and writes. All file locations are defined here and nowhere else."""
from __future__ import annotations

from pathlib import Path


def prompts_path(config: dict, lang: str, source: str) -> Path:
    """Step 1 (lang='en') and step 2 (translations, '<lang>-back' round trips)."""
    return config["paths"]["processed_dir"] / lang / f"{source}.jsonl"


def responses_path(config: dict, model: str, lang: str, source: str) -> Path:
    """Step 3: one {uid, prompt, response} per line."""
    return config["paths"]["responses_dir"] / model / lang / f"{source}.jsonl"


def labels_path(config: dict, model: str, lang: str, source: str) -> Path:
    """Step 4: one {uid, keyword, judge, judge_raw} per line."""
    return config["paths"]["labels_dir"] / model / lang / f"{source}.jsonl"


def features_path(config: dict, model: str, lang: str, source: str) -> Path:
    """Step 5: .npz with `uids` and one `layer_<L>` array per layer."""
    return config["paths"]["features_dir"] / model / lang / f"{source}.npz"


def results_dir(config: dict, name: str) -> Path:
    path = config["paths"]["results_dir"] / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_path(config: dict) -> Path:
    return config["paths"]["processed_dir"] / "manifest.json"
