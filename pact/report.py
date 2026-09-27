"""Turn per-partition rows into summary tables (CSV + printed)."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from pact.evaluation import corrected_ci


def write_csv(rows: list[dict], path: Path) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], keys: list[str], metrics: list[str], n_train: int, n_test: int) -> list[dict]:
    """Mean and corrected 95% CI of each metric, per group of `keys`.

    Every summarised metric is a rate or an AUC, so the interval is clipped to [0, 1].
    """
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for key, members in groups.items():
        row = dict(zip(keys, key))
        row["n_partitions"] = len(members)
        for m in metrics:
            values = [float(r[m]) for r in members if m in r]
            mean, lo, hi = corrected_ci(values, n_train, n_test)
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = mean, max(lo, 0.0), min(hi, 1.0)
        out.append(row)
    return out


def paired_differences(rows: list[dict], pairs: list[tuple[str, str]], metric: str, n_train: int,
                       n_test: int, regime: str = "marginal") -> list[dict]:
    """metric(a) - metric(b) on the same partition and language, e.g. A0 minus topic control."""
    value = {(r["partition"], r["method"], r["lang"]): r[metric] for r in rows if r["regime"] == regime}
    langs = sorted({r["lang"] for r in rows if r["lang"] != "en"})
    out = []
    for a, b in pairs:
        for lang in langs:
            diffs = [value[(p, a, lang)] - value[(p, b, lang)] for (p, m, l) in value
                     if m == a and l == lang and (p, b, lang) in value]
            mean, lo, hi = corrected_ci(diffs, n_train, n_test)
            out.append({"comparison": f"{a} - {b}", "lang": lang, "metric": metric,
                        "mean": mean, "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)})
    return out


def fmt(x, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def print_table(rows: list[dict], columns: list[tuple[str, str]], title: str) -> None:
    """columns: (header, key). A key ending in '+ci' prints 'mean [lo, hi]'."""
    print(f"\n{title}")
    cells = []
    for r in rows:
        line = []
        for _, key in columns:
            if key.endswith("+ci"):
                k = key[:-3]
                line.append(f"{fmt(r[k])} [{fmt(r[k + '_lo'])}, {fmt(r[k + '_hi'])}]")
            else:
                line.append(fmt(r.get(key)))
        cells.append(line)
    widths = [max(len(h), *(len(c[i]) for c in cells)) if cells else len(h) for i, (h, _) in enumerate(columns)]
    print("  ".join(h.ljust(w) for (h, _), w in zip(columns, widths)))
    for line in cells:
        print("  ".join(c.ljust(w) for c, w in zip(line, widths)))
