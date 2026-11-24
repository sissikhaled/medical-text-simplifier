"""Reference points for the results table (no training involved).

- copy:     the "simplification" is the input itself. Shows what a system that
            changes nothing scores (high overlap, low SARI, no readability gain).
- zeroshot: google/flan-t5-base used as-is with a natural-language instruction.
            Shows what the pre-trained model can do before fine-tuning.

Sources and references are normalized with the same `normalize_text` used for
training, so all systems are scored on identical text.

Outputs per baseline in runs/baseline_<name>/:
    config.json, predictions.jsonl (id, source, reference, prediction), metrics.json

Usage:
    python -m src.eval.baselines --which copy
    python -m src.eval.baselines --which zeroshot            # full test set
    python -m src.eval.baselines --which zeroshot --limit 10 # quick look
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.data.preprocess import normalize_text
from src.eval.metrics import compute_all

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
RUNS = ROOT / "runs"

ZERO_SHOT_MODEL = "google/flan-t5-base"
# FLAN-T5 was instruction-tuned, so it gets a plain-English instruction rather
# than the short "simplify: " prefix it has never seen (that prefix is only
# meaningful after fine-tuning).
ZERO_SHOT_PROMPT = (
    "Rewrite the following medical text in plain language that a patient "
    "without medical training can understand:\n\n{text}"
)


def load_split(split: str, limit: int | None = None) -> list[dict]:
    with (DATA / f"{split}.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    rows = rows[:limit] if limit else rows
    return [
        {"id": r["id"], "source": normalize_text(r["source"]),
         "reference": normalize_text(r["target"])}
        for r in rows
    ]


def save_run(name: str, rows: list[dict], predictions: list[str], config: dict) -> dict:
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r, p in zip(rows, predictions):
            f.write(json.dumps({**r, "prediction": p}, ensure_ascii=False) + "\n")
    metrics = compute_all([r["source"] for r in rows], predictions,
                          [r["reference"] for r in rows])
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\n{name}: {json.dumps(metrics, indent=2)}")
    return metrics


def run_copy(rows: list[dict], suffix: str) -> None:
    save_run(f"baseline_copy{suffix}", rows, [r["source"] for r in rows],
             {"system": "copy-input", "split": "test", "n_examples": len(rows)})


def run_zeroshot(rows: list[dict], suffix: str, batch_size: int) -> None:
    from src.generate import DEFAULT_GENERATION, generate, load_model

    model, tokenizer, device = load_model(ZERO_SHOT_MODEL)
    print(f"Zero-shot {ZERO_SHOT_MODEL} on {device}, {len(rows)} examples")
    start = time.time()
    predictions = generate(
        model, tokenizer, [ZERO_SHOT_PROMPT.format(text=r["source"]) for r in rows],
        device, batch_size=batch_size,
    )
    minutes = round((time.time() - start) / 60, 1)
    save_run(f"baseline_zeroshot{suffix}", rows, predictions, {
        "system": "zero-shot", "model": ZERO_SHOT_MODEL, "prompt": ZERO_SHOT_PROMPT,
        "generation": DEFAULT_GENERATION, "batch_size": batch_size,
        "device": str(device), "split": "test", "n_examples": len(rows),
        "generation_minutes": minutes,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--which", choices=["copy", "zeroshot", "all"], default="all")
    parser.add_argument("--limit", type=int, help="only the first N test examples")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    rows = load_split("test", args.limit)
    suffix = f"_first{args.limit}" if args.limit else ""
    if args.which in ("copy", "all"):
        run_copy(rows, suffix)
    if args.which in ("zeroshot", "all"):
        run_zeroshot(rows, suffix, args.batch_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
