"""Build every source, check it, and save it with a manifest."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pact.data.schema import PromptRecord
from pact.data.sources import LOADERS
from pact.io import read_jsonl, sha256_file, write_json, write_jsonl


def build_source(name: str, config: dict) -> list[PromptRecord]:
    """Load one source and validate it before anything is saved."""
    records = LOADERS[name](config["sources"][name], config["paths"]["raw_dir"], config["seed"])
    validate(name, records, config["sources"][name])
    return records


def validate(name: str, records: list[PromptRecord], cfg: dict) -> None:
    """Refuse to save a dataset that is not what we expect."""
    expected = cfg.get("expected_count", cfg.get("sample_n"))
    if expected is not None and len(records) != expected:
        raise ValueError(f"{name}: expected {expected} prompts, got {len(records)}")
    uids = Counter(r.uid for r in records)
    if dupes := [u for u, c in uids.items() if c > 1]:
        raise ValueError(f"{name}: duplicate uids {dupes[:5]}")
    texts = Counter(r.text for r in records)
    if dupes := [t for t, c in texts.items() if c > 1]:
        raise ValueError(f"{name}: duplicate prompt texts {dupes[:3]}")


def save_sources(datasets: dict[str, list[PromptRecord]], config: dict) -> dict:
    """Write data/processed/en/<source>.jsonl for each source, plus manifest.json."""
    processed = config["paths"]["processed_dir"]
    manifest_path = processed / "manifest.json"
    manifest = load_manifest(manifest_path)
    for name, records in datasets.items():
        path = processed / "en" / f"{name}.jsonl"
        write_jsonl((r.to_dict() for r in records), path)
        manifest["files"][f"en/{name}.jsonl"] = {
            "count": len(records),
            "gold_labels": dict(Counter(r.gold_label for r in records)),
            "n_translate": sum(r.translate for r in records),
            "sha256": sha256_file(path),
        }
    manifest["seed"] = config["seed"]
    manifest["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(manifest, manifest_path)
    return manifest


def load_manifest(path: Path) -> dict:
    """Keep entries from earlier runs, so building one source does not erase the others."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"files": {}}


def load_source(name: str, config: dict, lang: str = "en") -> list[PromptRecord]:
    """What later steps call to get a saved dataset back."""
    path = config["paths"]["processed_dir"] / lang / f"{name}.jsonl"
    return [PromptRecord.from_dict(d) for d in read_jsonl(path)]
