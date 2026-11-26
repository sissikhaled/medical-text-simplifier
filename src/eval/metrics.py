"""Evaluation metrics for medical text simplification.

All metrics take parallel lists: `sources` (model inputs), `predictions`
(model outputs) and `references` (human plain-language summaries).

- SARI (0-100, higher = better): the standard simplification metric. It
  compares the prediction with BOTH the source and the reference and averages
  three scores: words correctly ADDED, correctly KEPT, and correctly DELETED.
  Copying the input earns nothing for adding/deleting, so it scores low.
  Implementation: Hugging Face `evaluate` "sari" (Xu et al., 2016).
  Also reported as its three parts (sari_keep, sari_delete, sari_add; SARI is
  their mean). Note: the implementation scores "deleted nothing" as a perfect
  deletion (0/0 = 1), so copying the input gets 100 for deletion for free.
- BLEU (0-100): n-gram precision against the reference (sacrebleu, corpus level).
- ROUGE-1/2/L (0-100): n-gram / longest-common-subsequence overlap F1 with the
  reference (rouge_score, stemming on, averaged over examples).
- FKGL: Flesch-Kincaid Grade Level of predictions, sources and references
  (decimal-safe version from `src/eval/readability.py`; lower = easier).
- Length: mean words per prediction, and the ratio to the source length.
- Copying (0-100): share of the prediction's word trigrams that also occur in
  the source, averaged over examples. 100 = every 3-word phrase was copied;
  shows how extractive a system is. Also reported for the references, i.e. how
  much the human writers copy.
- Unsupported numbers (0-100): share of all numbers in the predictions that do
  not occur anywhere in the source (corpus level). For medical text this is a
  simple proxy for invented statistics. Also reported for the references.
"""

from __future__ import annotations

import re
import sys
from statistics import mean

import evaluate
import sacrebleu
from rouge_score import rouge_scorer

from src.eval.readability import fkgl

_sari = None


def _get_sari():
    global _sari
    if _sari is None:
        _sari = evaluate.load("sari")
    return _sari


def sari(sources: list[str], predictions: list[str], references: list[str]) -> float:
    result = _get_sari().compute(
        sources=sources, predictions=predictions, references=[[r] for r in references]
    )
    return float(result["sari"])


def _ngrams(words: list[str], n: int) -> list[str]:
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def sari_components(sources: list[str], predictions: list[str],
                    references: list[str]) -> dict[str, float]:
    """SARI split into keep / delete / add (each 0-100, averaged over examples).

    Uses the same n-gram scoring function and text normalization as the
    `evaluate` SARI metric, so the mean of the three equals `sari()`.
    """
    module = sys.modules[type(_get_sari()).__module__]
    keep, delete, add = [], [], []
    for src, pred, ref in zip(sources, predictions, references):
        s_words, c_words, r_words = (module.normalize(t).split(" ") for t in (src, pred, ref))
        parts = [module.SARIngram(_ngrams(s_words, n), _ngrams(c_words, n),
                                  [_ngrams(r_words, n)], 1) for n in (1, 2, 3, 4)]
        keep.append(mean(p[0] for p in parts))
        delete.append(mean(p[1] for p in parts))
        add.append(mean(p[2] for p in parts))
    return {"sari_keep": 100 * mean(keep), "sari_delete": 100 * mean(delete),
            "sari_add": 100 * mean(add)}


def bleu(predictions: list[str], references: list[str]) -> float:
    return float(sacrebleu.corpus_bleu(predictions, [references]).score)


def rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]
    return {k: 100 * mean(s[k].fmeasure for s in scores) for k in ("rouge1", "rouge2", "rougeL")}


_WORD = re.compile(r"\w+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _trigrams(text: str) -> list[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    return [tuple(words[i:i + 3]) for i in range(len(words) - 2)]


def copied_trigrams(sources: list[str], outputs: list[str]) -> float:
    """Mean % of each output's word trigrams found in its source (outputs with
    fewer than 3 words are skipped)."""
    shares = []
    for src, out in zip(sources, outputs):
        grams = _trigrams(out)
        if grams:
            src_grams = set(_trigrams(src))
            shares.append(100 * sum(g in src_grams for g in grams) / len(grams))
    return mean(shares) if shares else 0.0


def _numbers(text: str) -> list[str]:
    return [n.replace(",", "") for n in _NUMBER.findall(text)]


def unsupported_numbers(sources: list[str], outputs: list[str]) -> float:
    """% of all numbers in the outputs that never appear in the matching source."""
    total = unsupported = 0
    for src, out in zip(sources, outputs):
        src_numbers = set(_numbers(src))
        nums = _numbers(out)
        total += len(nums)
        unsupported += sum(n not in src_numbers for n in nums)
    return 100 * unsupported / total if total else 0.0


def compute_all(
    sources: list[str], predictions: list[str], references: list[str]
) -> dict[str, float]:
    """Every project metric in one dictionary (values rounded to 2 decimals)."""
    if not (len(sources) == len(predictions) == len(references)):
        raise ValueError("sources, predictions and references must have equal length")

    pred_words = [len(p.split()) for p in predictions]
    src_words = [len(s.split()) for s in sources]
    results = {
        "sari": sari(sources, predictions, references),
        **sari_components(sources, predictions, references),
        "bleu": bleu(predictions, references),
        **rouge(predictions, references),
        "fkgl_prediction": mean(fkgl(p) for p in predictions),
        "fkgl_source": mean(fkgl(s) for s in sources),
        "fkgl_reference": mean(fkgl(r) for r in references),
        "words_prediction": mean(pred_words),
        "length_ratio": mean(p / s for p, s in zip(pred_words, src_words)),
        "copied_trigrams": copied_trigrams(sources, predictions),
        "copied_trigrams_reference": copied_trigrams(sources, references),
        "unsupported_numbers": unsupported_numbers(sources, predictions),
        "unsupported_numbers_reference": unsupported_numbers(sources, references),
        "n_examples": len(predictions),
    }
    return {k: round(float(v), 2) if k != "n_examples" else v for k, v in results.items()}
