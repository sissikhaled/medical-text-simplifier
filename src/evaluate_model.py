"""Evaluate a fine-tuned model on the test set (480 pairs).

Generates one simplification per test abstract, then computes every project
metric (src/eval/metrics.py) and saves, in runs/<run>/test_<decoding>/:

    predictions.jsonl   id, source, reference, prediction (test-set order)
    metrics.json        SARI, BLEU, ROUGE, FKGL, length, copying, unsupported numbers
    config.json         model, decoding settings, device, minutes

The task prefix and the input length are read from the run's training config
(runs/<run>/config.yaml), so each model is tested the way it was trained
(e.g. the 512-token variation sees 512 tokens).

Generation is slow (long outputs, beam search), so progress is saved after
every chunk to predictions.partial.jsonl; running the same command again
continues where it stopped.

Usage:
    python -m src.evaluate_model --run flan-t5-base_lr3e-4_src1024 --model-dir models/flan-t5-base_lr3e-4_src1024
    python -m src.evaluate_model --run ... --model-dir ... --decoding greedy
    python -m src.evaluate_model --run ... --model-dir ... --limit 8     # quick check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from src.eval.baselines import load_split
from src.eval.metrics import compute_all
from src.generate import DEFAULT_GENERATION, generate, load_model

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

# Same settings except the search: beam search keeps the 4 most likely partial
# outputs at each step, greedy keeps only the single most likely next token.
DECODING = {
    "beam4": dict(DEFAULT_GENERATION),
    "greedy": {**DEFAULT_GENERATION, "num_beams": 1, "early_stopping": False},
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(run: str, model_dir: Path, decoding: str, limit: int | None,
             batch_size: int, chunk: int, runs_dir: Path = RUNS) -> dict:
    run_cfg_path = runs_dir / run / "config.yaml"
    if not run_cfg_path.exists():
        raise SystemExit(f"{run_cfg_path} not found: --run must name a trained run in {runs_dir}")
    run_cfg = yaml.safe_load(run_cfg_path.read_text())
    prefix = run_cfg.get("prefix", "simplify: ")
    max_source = run_cfg["max_source_tokens"]
    params = DECODING[decoding]

    out_dir = runs_dir / run / (f"test_{decoding}" + (f"_first{limit}" if limit else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / "predictions.partial.jsonl"

    rows = load_split("test", limit)
    # Finished earlier (predictions.jsonl) or interrupted (partial): reuse what exists.
    done = {r["id"]: r["prediction"]
            for r in read_jsonl(out_dir / "predictions.jsonl") + read_jsonl(partial)}
    todo = [r for r in rows if r["id"] not in done]
    # Longest first: similar lengths share a batch, and the ETA is pessimistic.
    todo.sort(key=lambda r: -len(r["source"]))
    print(f"{run} | {decoding} | inputs {max_source} tokens | {len(rows)} test examples, "
          f"{len(done)} already done")

    model, tokenizer, device = load_model(str(model_dir))
    print(f"Model {model_dir} on {device}")

    start, generated = time.time(), 0
    with partial.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            preds = generate(model, tokenizer, [prefix + r["source"] for r in part], device,
                             batch_size=batch_size, max_source_tokens=max_source,
                             log_every=0, **params)
            for r, p in zip(part, preds):
                f.write(json.dumps({"id": r["id"], "prediction": p}, ensure_ascii=False) + "\n")
                done[r["id"]] = p
            f.flush()
            generated += len(part)
            elapsed = (time.time() - start) / 60
            eta = elapsed / generated * (len(todo) - generated)
            print(f"  {len(done)}/{len(rows)} done  elapsed {elapsed:.1f} min  "
                  f"eta {eta:.1f} min (shorter inputs come later, so likely less)", flush=True)
    minutes = round((time.time() - start) / 60, 1)

    predictions = [done[r["id"]] for r in rows]
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r, p in zip(rows, predictions):
            f.write(json.dumps({**r, "prediction": p}, ensure_ascii=False) + "\n")
    partial.unlink()

    metrics = compute_all([r["source"] for r in rows], predictions,
                          [r["reference"] for r in rows])
    config = {"run": run, "model_dir": model_dir.name, "decoding": decoding,
              "generation": params, "prefix": prefix, "max_source_tokens": max_source,
              "batch_size": batch_size, "device": str(device), "split": "test",
              "n_examples": len(rows), "generation_minutes_this_session": minutes}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="run name, e.g. flan-t5-base_lr3e-4_src1024")
    parser.add_argument("--model-dir", required=True, type=Path,
                        help="folder with the trained model (the run's best/ folder)")
    parser.add_argument("--decoding", choices=sorted(DECODING), default="beam4")
    parser.add_argument("--limit", type=int, help="only the first N test examples")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=20,
                        help="examples generated between progress saves")
    parser.add_argument("--runs-dir", type=Path, default=RUNS)
    args = parser.parse_args()
    evaluate(args.run, args.model_dir, args.decoding, args.limit, args.batch_size,
             args.chunk, args.runs_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
