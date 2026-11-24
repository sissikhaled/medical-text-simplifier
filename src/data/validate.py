"""Sanity checks on data/processed/*.jsonl.

Checks:
  1. Split sizes match the paper (train 3568, val 411, test 480 = 4459 pairs),
     minus the one broken train pair dropped by download.py -> train 3567.
  2. No empty sources, targets or ids.
  3. No duplicate ids or duplicate sources within a split.
  4. No leakage: no id or source text appears in more than one split.
  5. No pair whose target is identical to its source.

Usage:
    python -m src.data.validate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
EXPECTED = {"train": 3567, "val": 411, "test": 480}


def load(split: str) -> list[dict]:
    with (PROCESSED_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> int:
    data = {s: load(s) for s in EXPECTED}
    problems: list[str] = []

    for split, rows in data.items():
        n = len(rows)
        status = "ok" if n == EXPECTED[split] else f"expected {EXPECTED[split]}"
        print(f"{split:<5} {n:>5} pairs  [{status}]")
        if n != EXPECTED[split]:
            problems.append(f"{split}: {n} pairs, expected {EXPECTED[split]}")

        for field in ("id", "source", "target"):
            empty = sum(1 for r in rows if not r[field])
            if empty:
                problems.append(f"{split}: {empty} empty '{field}'")

        for field in ("id", "source"):
            dupes = n - len({r[field] for r in rows})
            if dupes:
                problems.append(f"{split}: {dupes} duplicate '{field}' values")

        same = sum(1 for r in rows if r["source"] == r["target"])
        if same:
            problems.append(f"{split}: {same} pairs with target == source")

    splits = list(data)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            for field in ("id", "source"):
                shared = {r[field] for r in data[a]} & {r[field] for r in data[b]}
                if shared:
                    problems.append(f"{a}/{b}: {len(shared)} shared '{field}' values")

    print()
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
