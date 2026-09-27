"""One loader per prompt source.

Every loader has the same shape: load_<name>(source_config, raw_dir, seed) -> list[PromptRecord].
Raw files are downloaded once into raw_dir and reused afterwards.
"""
from __future__ import annotations

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

from pact.config import resolve
from pact.data.schema import PromptRecord
from pact.io import download_cached


def read_csv(path: Path, required_columns: set[str]) -> list[dict]:
    """Read a CSV and fail loudly if an expected column is missing.

    Selecting columns by an explicit, checked name (never by guessing) is the
    guard against the paper's Appendix A bug, where the wrong column was used.
    """
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}; found {reader.fieldnames}")
        return list(reader)


def load_advbench(cfg: dict, raw_dir: Path, seed: int) -> list[PromptRecord]:
    path = download_cached(cfg["url"], raw_dir / "advbench_harmful_behaviors.csv")
    rows = read_csv(path, {"goal"})
    return [
        PromptRecord(
            uid=f"advbench-{i:04d}",
            source="advbench",
            source_index=i,
            text=row["goal"].strip(),
            gold_label="unsafe",
            translate=i < cfg["translate_first_n"],
        )
        for i, row in enumerate(rows)
    ]


def load_maliciousinstruct(cfg: dict, raw_dir: Path, seed: int) -> list[PromptRecord]:
    path = download_cached(cfg["url"], raw_dir / "maliciousinstruct.txt")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [
        PromptRecord(
            uid=f"maliciousinstruct-{i:04d}",
            source="maliciousinstruct",
            source_index=i,
            text=text,
            gold_label="unsafe",
        )
        for i, text in enumerate(lines)
    ]


def load_xstest(cfg: dict, raw_dir: Path, seed: int) -> list[PromptRecord]:
    path = download_cached(cfg["url"], raw_dir / "xstest_prompts.csv")
    rows = read_csv(path, {"id", "prompt", "type", "label"})
    return [
        PromptRecord(
            uid=f"xstest-{int(row['id']):04d}",
            source="xstest",
            source_index=int(row["id"]),
            text=row["prompt"].strip(),
            gold_label=row["label"].strip(),  # already "safe" / "unsafe"
            category=row["type"].strip(),
            translate=True,
        )
        for row in rows
    ]


def load_orbench_hard(cfg: dict, raw_dir: Path, seed: int) -> list[PromptRecord]:
    """OR-Bench-Hard: benign prompts that look harmful. A stratified sample by category."""
    path = raw_dir / "orbench_hard_1k.csv"
    if not path.exists():
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(repo_id=cfg["hf_repo"], filename=cfg["hf_file"], repo_type="dataset")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded, path)
    rows = read_csv(path, {"prompt", "category"})
    indexed = list(enumerate(rows))
    chosen = stratified_sample(indexed, key=lambda item: item[1]["category"], n=cfg["sample_n"], seed=seed)
    return [
        PromptRecord(
            uid=f"orbench-{i:04d}",
            source="orbench_hard",
            source_index=i,
            text=row["prompt"].strip(),
            gold_label="safe",  # OR-Bench-Hard prompts are benign by construction
            category=row["category"].strip(),
            translate=True,
        )
        for i, row in chosen
    ]


def load_benign(cfg: dict, raw_dir: Path, seed: int) -> list[PromptRecord]:
    """Our own hand-written everyday requests (blank lines and # comments are skipped)."""
    lines = resolve(cfg["path"]).read_text(encoding="utf-8").splitlines()
    texts = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    return [
        PromptRecord(uid=f"benign-{i:04d}", source="benign", source_index=i, text=text, gold_label="safe")
        for i, text in enumerate(texts)
    ]


def stratified_sample(items: list, key, n: int, seed: int) -> list:
    """Pick n items so each group keeps its share of the whole (proportional allocation).

    Each group first gets floor(n * share) items; the few leftover slots go to the
    groups with the largest remainders. The result is sorted back into the original
    order, and the same seed always gives the same sample.
    """
    if n > len(items):
        raise ValueError(f"cannot sample {n} from {len(items)} items")
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)

    quota = {g: n * len(members) / len(items) for g, members in groups.items()}
    alloc = {g: int(q) for g, q in quota.items()}
    leftover = n - sum(alloc.values())
    for g in sorted(quota, key=lambda g: (-(quota[g] - alloc[g]), g))[:leftover]:
        alloc[g] += 1

    rng = random.Random(seed)
    chosen = []
    for g in sorted(groups):
        chosen.extend(rng.sample(groups[g], alloc[g]))
    position = {id(item): i for i, item in enumerate(items)}
    return sorted(chosen, key=lambda item: position[id(item)])


LOADERS = {
    "advbench": load_advbench,
    "maliciousinstruct": load_maliciousinstruct,
    "xstest": load_xstest,
    "orbench_hard": load_orbench_hard,
    "benign": load_benign,
}
