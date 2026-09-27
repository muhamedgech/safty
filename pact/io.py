"""Reusable file helpers: cached downloads, JSONL read/write, and hashing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import requests


def download_cached(url: str, dest: Path, timeout: int = 60) -> Path:
    """Download `url` to `dest` once; later calls reuse the local copy."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    # Write to a temporary name first so a crash never leaves a half-written file.
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(response.content)
    tmp.replace(dest)
    return dest


def sha256_file(path: Path) -> str:
    """Fingerprint of a file; if two runs give the same hash, the data is identical."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(records: Iterable[dict], path: Path) -> int:
    """One JSON object per line. Returns the number of records written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(records: Iterable[dict], path: Path) -> None:
    """Add records to the end of a file; used to save progress after every batch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
