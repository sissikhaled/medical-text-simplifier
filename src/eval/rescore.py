"""Recompute metrics.json from saved predictions (no generation needed).

Use after the metric code changes, e.g.:
    python -m src.eval.rescore runs/baseline_copy runs/baseline_zeroshot
Each folder must contain predictions.jsonl with source, reference, prediction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.eval.metrics import compute_all


def rescore(folder: Path) -> dict:
    with (folder / "predictions.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    metrics = compute_all([r["source"] for r in rows], [r["prediction"] for r in rows],
                          [r["reference"] for r in rows])
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for arg in sys.argv[1:]:
        print(arg, json.dumps(rescore(Path(arg)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
