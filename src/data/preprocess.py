"""Text preprocessing and tokenization for FLAN-T5.

Two steps:
1. `normalize_text`: rewrite characters FLAN-T5's vocabulary cannot represent
   (they would become <unk> and could never be produced by the model) into
   words or ASCII the model knows. Applied identically to sources and targets.
   Phase 2 found such characters in 16.7% of sources (mostly −, <, µ, ±, α, ≥).
2. `make_tokenize_fn`: add the task prefix and turn text into token ids for
   the Seq2SeqTrainer (`input_ids`, `attention_mask`, `labels`).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable

PREFIX = "simplify: "
MAX_SOURCE_TOKENS = 1024
MAX_TARGET_TOKENS = 768

# Applied before Unicode decomposition. NFKD would turn ½ into 1⁄2 (itself
# unknown to the tokenizer), and ˜ / ´ into a space + combining accent that
# the accent-stripping step then deletes, silently losing "approximately".
_PRE_NFKD = {"½": " 1/2 ", "˜": " approximately ", "´": "'"}

# Applied after NFKD. Order matters: multi-character keys first
# (e.g. "μg" before "μ"). The micro sign µ (U+00B5) becomes Greek μ under NFKD.
_SYMBOLS = [
    ("≤", " less than or equal to "),
    ("≦", " less than or equal to "),
    ("≥", " greater than or equal to "),
    ("≧", " greater than or equal to "),
    ("<", " less than "),             # '<' is not in the T5 vocabulary; '>' is
    ("±", " +/- "),
    ("μg", "mcg"),                    # standard medical abbreviation
    ("μ", "micro"),
    ("α", "alpha"), ("ɑ", "alpha"),
    ("β", "beta"),
    ("γ", "gamma"),
    ("Δ", "delta"), ("δ", "delta"),
    ("κ", "kappa"),
    ("ω", "omega"),
    ("−", "-"), ("‒", "-"), ("‐", "-"), ("‑", "-"),
    ("~", " approximately "),
    ("×", " x "),
    ("∞", " infinity "),
    ("·", " - "), ("▪", " - "), ("∙", " - "),   # list bullets
    ("`", "'"),
    ("©", "(c)"),
    ("{", "("), ("}", ")"),
    ("^", " to the power of "),
    ("ø", "o"), ("ƞ", "n"),
]

# A middle dot between digits is a decimal point ("0·5"), not a bullet.
_DOT_DECIMAL = re.compile(r"(?<=\d)[·∙](?=\d)")
_SPACES = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:)])")
_SPACE_AFTER_PAREN = re.compile(r"\(\s+")


def normalize_text(text: str) -> str:
    """Rewrite unrepresentable characters; leave everything else unchanged."""
    for old, new in _PRE_NFKD.items():
        text = text.replace(old, new)
    # Decompose accented letters and drop the accent marks (Å -> A, ú -> u).
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = _DOT_DECIMAL.sub(".", text)
    for old, new in _SYMBOLS:
        text = text.replace(old, new)
    text = _SPACES.sub(" ", text).strip()
    text = _SPACE_AFTER_PAREN.sub("(", text)
    return _SPACE_BEFORE_PUNCT.sub(r"\1", text)


def make_tokenize_fn(
    tokenizer,
    max_source_tokens: int = MAX_SOURCE_TOKENS,
    max_target_tokens: int = MAX_TARGET_TOKENS,
    prefix: str = PREFIX,
) -> Callable[[dict], dict]:
    """Return a function for `datasets.Dataset.map(..., batched=True)`.

    Expects already-normalized `source`/`target` columns. Longer texts are
    truncated (Phase 2: 2.5% of sources at 1024, 0.7% of targets at 768).
    Padding is left to the data collator so each batch is padded only to its
    own longest example.
    """

    def tokenize(batch: dict) -> dict:
        model_inputs = tokenizer(
            [prefix + s for s in batch["source"]],
            max_length=max_source_tokens,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=max_target_tokens,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return tokenize
