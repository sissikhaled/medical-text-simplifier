"""Tests for src/data/preprocess.py and src/eval/readability.py."""

import json
from pathlib import Path

import pytest

from src.data.preprocess import PREFIX, make_tokenize_fn, normalize_text
from src.eval.readability import fkgl, normalize_for_readability

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("P<0.05", "P less than 0.05"),
        ("95% CI −0.2 to 1.1", "95% CI -0.2 to 1.1"),
        ("5 µg/kg ± 2", "5 mcg/kg +/- 2"),
        ("aged ≥ 65 and ≤ 80", "aged greater than or equal to 65 and less than or equal to 80"),
        ("IL‐1β and TNF‐α", "IL-1beta and TNF-alpha"),
        ("adverse event (˜60%)", "adverse event (approximately 60%)"),
        ("at 0·5 mg", "at 0.5 mg"),
        ("Crohn´s disease", "Crohn's disease"),
        ("ranged {median 9}", "ranged (median 9)"),
        ("Ångström", "Angstrom"),
        ("Plain text stays the same.", "Plain text stays the same."),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("google/flan-t5-base")


def test_no_unknown_tokens_after_normalization(tokenizer):
    """Every character left in the normalized dataset is in the T5 vocabulary."""
    texts = []
    for split in ("train", "val", "test"):
        with (ROOT / "data" / "processed" / f"{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                texts += [normalize_text(row["source"]), normalize_text(row["target"])]
    chars = set("".join(texts))
    unknown = {c for c in chars
               if tokenizer.unk_token_id in tokenizer(c, add_special_tokens=False)["input_ids"]}
    assert unknown == set()


def test_tokenize_fn_prefix_and_truncation(tokenizer):
    tokenize = make_tokenize_fn(tokenizer, max_source_tokens=16, max_target_tokens=8)
    batch = {"source": ["word " * 50], "target": ["simple " * 50]}
    out = tokenize(batch)
    assert len(out["input_ids"][0]) == 16          # truncated, ends with </s>
    assert len(out["labels"][0]) == 8
    assert out["input_ids"][0][-1] == tokenizer.eos_token_id
    decoded = tokenizer.decode(out["input_ids"][0], skip_special_tokens=True)
    assert decoded.startswith(PREFIX.strip())


def test_readability_ignores_decimal_points():
    text = "The risk was lower (RR 0.85, 95% CI 0.71 to 1.02), e.g. in adults vs. children."
    assert "0.85" not in normalize_for_readability(text)
    # One real sentence: the decimals must not make it look easier.
    import textstat
    assert textstat.sentence_count(normalize_for_readability(text)) == 1
    assert fkgl(text) > textstat.flesch_kincaid_grade(text)
