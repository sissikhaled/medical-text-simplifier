"""Shared text generation for baselines, evaluation and the demo.

Decoding defaults (from the plan): beam search with 4 beams and
`no_repeat_ngram_size=3` (stops the model from repeating phrases), inputs
truncated to 1024 tokens, outputs up to 768 new tokens.
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.data.preprocess import MAX_SOURCE_TOKENS, MAX_TARGET_TOKENS

DEFAULT_GENERATION = {
    "num_beams": 4,
    "no_repeat_ngram_size": 3,
    "max_new_tokens": MAX_TARGET_TOKENS,
    "early_stopping": True,
}


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(name_or_path: str, device: torch.device | None = None):
    device = device or pick_device()
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path).to(device).eval()
    return model, tokenizer, device


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    inputs: list[str],
    device: torch.device,
    batch_size: int = 4,
    max_source_tokens: int = MAX_SOURCE_TOKENS,
    log_every: int = 10,
    **generation_kwargs,
) -> list[str]:
    """Generate one output per input string (inputs already carry any prompt).

    Inputs are processed longest-first so each batch has similar lengths
    (less padding), and the outputs are returned in the original order.
    """
    params = {**DEFAULT_GENERATION, **generation_kwargs}
    order = sorted(range(len(inputs)), key=lambda i: -len(inputs[i]))
    outputs: list[str | None] = [None] * len(inputs)
    n_batches = (len(inputs) + batch_size - 1) // batch_size
    start = time.time()

    for b in range(n_batches):
        idx = order[b * batch_size:(b + 1) * batch_size]
        enc = tokenizer(
            [inputs[i] for i in idx],
            max_length=max_source_tokens,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        out = model.generate(**enc, **params)
        for i, text in zip(idx, tokenizer.batch_decode(out, skip_special_tokens=True)):
            outputs[i] = text.strip()
        if log_every and ((b + 1) % log_every == 0 or b + 1 == n_batches):
            elapsed = time.time() - start
            eta = elapsed / (b + 1) * (n_batches - b - 1)
            print(f"  batch {b + 1}/{n_batches}  elapsed {elapsed / 60:.1f} min  "
                  f"eta {eta / 60:.1f} min", flush=True)
    return outputs  # type: ignore[return-value]
