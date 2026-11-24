"""Download the Cochrane paragraph-simplification dataset and convert it to JSONL.

Source: Devaraj et al., "Paragraph-level Simplification of Medical Texts",
NAACL 2021. Data license: CC BY 4.0.
https://github.com/AshOlogn/Paragraph-level-Simplification-of-Medical-Texts

The authors' `data/data-1024` folder holds the official train/val/test split as
line-aligned text files: line i of `<split>.source` (technical abstract),
`<split>.target` (plain-language summary) and `<split>.doi` (review DOI) form
one example.

Outputs:
    data/raw/cochrane/<split>.{source,target,doi}   untouched downloads (git-ignored)
    data/processed/<split>.jsonl                    {"id", "source", "target"} per line

Usage:
    python -m src.data.download
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

# Pinned to a specific commit so the data can never change underneath us.
REPO = "AshOlogn/Paragraph-level-Simplification-of-Medical-Texts"
COMMIT = "edf6504ea28b2458ec6b4c172482ad15387aeeef"
BASE_URL = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/data/data-1024"

SPLITS = ("train", "val", "test")
FIELDS = ("source", "target", "doi")

# Byte sizes listed by GitHub for this commit, used to detect broken downloads.
EXPECTED_SIZES = {
    "train.source": 8184937, "train.target": 4907890, "train.doi": 109608,
    "val.source": 933042, "val.target": 543503, "val.doi": 12561,
    "test.source": 1097248, "test.target": 663706, "test.doi": 14780,
}

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "cochrane"
PROCESSED_DIR = ROOT / "data" / "processed"


def download_raw(force: bool = False) -> None:
    """Fetch the 9 split files unless they are already present and complete."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, size in EXPECTED_SIZES.items():
        path = RAW_DIR / name
        if path.exists() and path.stat().st_size == size and not force:
            print(f"  have      {name}")
            continue
        print(f"  download  {name}")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", path)
        actual = path.stat().st_size
        if actual != size:
            raise RuntimeError(f"{name}: got {actual} bytes, expected {size}")


def clean(text: str) -> str:
    """Normalize whitespace only; the wording is left exactly as written."""
    return re.sub(r"\s+", " ", text).strip()


def read_lines(path: Path) -> list[str]:
    # Split on "\n" only: str.splitlines() would also split on rare Unicode
    # line separators inside a paragraph and misalign the three files.
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def build_split(split: str) -> list[dict]:
    columns = {f: read_lines(RAW_DIR / f"{split}.{f}") for f in FIELDS}
    lengths = {f: len(v) for f, v in columns.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"{split}: files are not line-aligned: {lengths}")
    rows = [
        {"id": clean(doi), "source": clean(src), "target": clean(tgt)}
        for src, tgt, doi in zip(columns["source"], columns["target"], columns["doi"])
    ]
    # A pair whose "plain" summary is the abstract itself is a scraping error
    # (one exists in train: CD004957.pub3). It would teach the model to copy.
    broken = [r["id"] for r in rows if r["source"] == r["target"]]
    for doi in broken:
        print(f"  {split:<5} dropped {doi} (target identical to source)")
    return [r for r in rows if r["source"] != r["target"]]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    print(f"Raw files -> {RAW_DIR.relative_to(ROOT)}")
    download_raw()
    print(f"Processed -> {PROCESSED_DIR.relative_to(ROOT)}")
    for split in SPLITS:
        rows = build_split(split)
        write_jsonl(rows, PROCESSED_DIR / f"{split}.jsonl")
        print(f"  {split:<5} {len(rows):>5} pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
