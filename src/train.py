"""Fine-tune FLAN-T5 on the Cochrane simplification data.

Every setting comes from a YAML config (see configs/). One run writes:

    runs/<run_name>/                  small files, committed to git
        config.yaml                   exact settings used (incl. resolved device/precision)
        log_history.csv               train loss per logging step, val loss per epoch
        loss_curve.png
        train_summary.json            best val loss/epoch, steps, minutes, paths
        val_samples.jsonl             a few validation outputs from the best model
    <output_dir>/                     large files, NOT in git (local disk or Google Drive)
        checkpoint-*/                 per-epoch checkpoints (Trainer keeps best + latest)
        best/                         best model + tokenizer, ready for from_pretrained

Usage:
    python -m src.train --config configs/smoke_mac.yaml
    python -m src.train --config configs/base.yaml --output-dir /content/drive/MyDrive/mts
    python -m src.train --config configs/base.yaml --resume   # continue an interrupted run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)

from src.data.preprocess import make_tokenize_fn, normalize_text

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RUNS = ROOT / "runs"


def display_path(path: str | Path | None) -> str | None:
    """Path relative to the repo when inside it, so saved run files don't
    contain machine-specific absolute paths (e.g. the user's home folder)."""
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- config

def load_config(path: Path) -> dict:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    required = ["run_name", "model_name", "learning_rate", "num_train_epochs",
                "per_device_train_batch_size", "gradient_accumulation_steps",
                "max_source_tokens", "max_target_tokens"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing config keys {missing}")
    return cfg


def resolve_precision(setting: str) -> bool:
    """Return True to train in bf16.

    T5 overflows in fp16, so fp16 is never used. bf16 is used only on GPUs
    with native support (CUDA compute capability >= 8, e.g. A100/L4); the
    Colab T4 (7.5) and Apple MPS train in fp32.
    """
    if setting == "fp32":
        return False
    if setting == "bf16":
        return True
    if setting != "auto":
        raise ValueError(f"precision must be auto, bf16 or fp32, got {setting!r}")
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


# --------------------------------------------------------------------------- data

def load_split(split: str, limit: int | None) -> Dataset:
    with (DATA / f"{split}.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if limit:
        rows = rows[:limit]
    return Dataset.from_list([
        {"id": r["id"], "source": normalize_text(r["source"]),
         "target": normalize_text(r["target"])}
        for r in rows
    ])


# --------------------------------------------------------------------------- logging

class CsvLogger(TrainerCallback):
    """Write every Trainer log entry to log_history.csv as training goes.

    transformers 5.x reports T5's training loss summed over the gradient-
    accumulation micro-batches (verified in the Phase 4 smoke test), so we
    also store `train_loss_per_example_batch` = loss / gradient_accumulation_steps,
    which is on the same scale as the validation loss.
    """

    FIELDS = ["step", "epoch", "train_loss", "train_loss_per_example_batch",
              "learning_rate", "grad_norm", "eval_loss"]

    def __init__(self, path: Path, grad_accum: int):
        self.path = path
        self.grad_accum = grad_accum
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("w", newline="") as f:
                csv.writer(f).writerow(self.FIELDS)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" not in logs and "eval_loss" not in logs:
            return  # end-of-training summary line
        loss = logs.get("loss")
        row = {
            "step": state.global_step,
            "epoch": round(state.epoch or 0, 4),
            "train_loss": loss,
            "train_loss_per_example_batch": loss / self.grad_accum if loss is not None else None,
            "learning_rate": logs.get("learning_rate"),
            "grad_norm": logs.get("grad_norm"),
            "eval_loss": logs.get("eval_loss"),
        }
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow(["" if row[k] is None else row[k] for k in self.FIELDS])


def plot_losses(csv_path: Path, out_path: Path, run_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, val = [], []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r["train_loss_per_example_batch"]:
                train.append((float(r["epoch"]), float(r["train_loss_per_example_batch"])))
            if r["eval_loss"]:
                val.append((float(r["epoch"]), float(r["eval_loss"])))
    fig, ax = plt.subplots(figsize=(7, 4))
    if train:
        ax.plot(*zip(*train), label="train loss", color="#4C72B0", alpha=0.8)
    if val:
        ax.plot(*zip(*val), "o-", label="validation loss", color="#C44E52")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"Loss — {run_name}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- main

def train(cfg: dict, output_dir: Path, resume: bool) -> dict:
    set_seed(cfg.get("seed", 42))
    run_dir = RUNS / cfg["run_name"]
    ckpt_dir = output_dir / cfg["run_name"]
    if not resume:
        for stale in (run_dir / "log_history.csv",):
            stale.unlink(missing_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    use_bf16 = resolve_precision(cfg.get("precision", "auto"))
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    resolved = {**cfg, "resolved_device": device, "resolved_bf16": use_bf16,
                "checkpoint_dir": display_path(ckpt_dir)}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    print(f"Run '{cfg['run_name']}' on {device} ({'bf16' if use_bf16 else 'fp32'}); "
          f"checkpoints -> {display_path(ckpt_dir)}")

    # Data
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    train_ds = load_split("train", cfg.get("limit_train"))
    val_ds = load_split("val", cfg.get("limit_val"))
    tokenize = make_tokenize_fn(tokenizer, cfg["max_source_tokens"], cfg["max_target_tokens"],
                                cfg.get("prefix", "simplify: "))
    train_tok = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)
    print(f"train {len(train_tok)} pairs, val {len(val_tok)} pairs")

    # Model. Eager attention so attention weights are available later
    # (SDPA in transformers 5.x returns none; needed for the heatmap bonus).
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"],
                                                  attn_implementation="eager")
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    args = Seq2SeqTrainingArguments(
        output_dir=str(ckpt_dir),
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size",
                                           cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_steps=cfg.get("warmup_steps", 0.05),
        weight_decay=cfg.get("weight_decay", 0.01),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        optim=cfg.get("optim", "adamw_torch"),
        activation_checkpointing=cfg.get("activation_checkpointing", False),
        bf16=use_bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,                 # with load_best: keeps best + latest
        save_only_model=cfg.get("save_only_model", False),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=cfg.get("logging_steps", 10),
        logging_first_step=True,
        predict_with_generate=False,        # generation during eval is slow
        report_to="none",
        seed=cfg.get("seed", 42),
        dataloader_num_workers=0,
    )
    callbacks = [CsvLogger(run_dir / "log_history.csv", cfg["gradient_accumulation_steps"])]
    if cfg.get("early_stopping_patience"):
        callbacks.append(EarlyStoppingCallback(cfg["early_stopping_patience"]))

    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok,
        data_collator=collator, processing_class=tokenizer, callbacks=callbacks,
    )

    start = time.time()
    trainer.train(resume_from_checkpoint=True if resume else None)
    minutes = (time.time() - start) / 60

    # Save the best model (already loaded by load_best_model_at_end).
    best_dir = ckpt_dir / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    plot_losses(run_dir / "log_history.csv", run_dir / "loss_curve.png", cfg["run_name"])

    evals = [h for h in trainer.state.log_history if "eval_loss" in h]
    best = min(evals, key=lambda h: h["eval_loss"]) if evals else {}
    summary = {
        "run_name": cfg["run_name"],
        "best_eval_loss": best.get("eval_loss"),
        "best_epoch": best.get("epoch"),
        "best_eval_perplexity": math.exp(best["eval_loss"]) if best else None,
        "epochs_completed": round(trainer.state.epoch or 0, 2),
        "global_steps": trainer.state.global_step,
        "stopped_early": (trainer.state.epoch or 0) < cfg["num_train_epochs"] - 1e-6,
        "train_minutes": round(minutes, 1),
        "device": device,
        "bf16": use_bf16,
        "best_model_dir": display_path(best_dir),
        "best_checkpoint": display_path(trainer.state.best_model_checkpoint),
        "resumed": resume,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    n_samples = cfg.get("val_samples", 3)
    if n_samples:
        write_val_samples(best_dir, val_ds.select(range(min(n_samples, len(val_ds)))),
                          run_dir / "val_samples.jsonl", cfg)
    return summary


def write_val_samples(model_dir: Path, rows: Dataset, out: Path, cfg: dict) -> None:
    """Reload the saved best model from disk (proves it is usable) and simplify a few
    validation abstracts."""
    from src.generate import generate, load_model

    model, tokenizer, device = load_model(str(model_dir))
    prefix = cfg.get("prefix", "simplify: ")
    preds = generate(model, tokenizer, [prefix + s for s in rows["source"]], device,
                     batch_size=2, max_source_tokens=cfg["max_source_tokens"],
                     max_new_tokens=cfg["max_target_tokens"], log_every=0)
    with out.open("w", encoding="utf-8") as f:
        for r, p in zip(rows, preds):
            f.write(json.dumps({"id": r["id"], "source": r["source"], "reference": r["target"],
                                "prediction": p}, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(preds)} validation samples to {out.relative_to(ROOT)}")
    print("first sample:\n", preds[0][:600])


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune FLAN-T5 for medical text simplification.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints",
                        help="where checkpoints and the best model go (default: ./checkpoints)")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest checkpoint in the output dir. Note: the "
                             "early-stopping patience counter restarts from 0.")
    args = parser.parse_args()
    train(load_config(args.config), args.output_dir, args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
