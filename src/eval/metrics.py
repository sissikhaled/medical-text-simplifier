"""Evaluation metrics for medical text simplification.

All metrics take parallel lists: `sources` (model inputs), `predictions`
(model outputs) and `references` (human plain-language summaries).

- SARI (0-100, higher = better): the standard simplification metric. It
  compares the prediction with BOTH the source and the reference and averages
  three scores: words correctly ADDED, correctly KEPT, and correctly DELETED.
  Copying the input earns nothing for adding/deleting, so it scores low.
  Implementation: Hugging Face `evaluate` "sari" (Xu et al., 2016).
- BLEU (0-100): n-gram precision against the reference (sacrebleu, corpus level).
- ROUGE-1/2/L (0-100): n-gram / longest-common-subsequence overlap F1 with the
  reference (rouge_score, stemming on, averaged over examples).
- FKGL: Flesch-Kincaid Grade Level of predictions, sources and references
  (decimal-safe version from `src/eval/readability.py`; lower = easier).
- Length: mean words per prediction, and the ratio to the source length.
"""

from __future__ import annotations

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


def bleu(predictions: list[str], references: list[str]) -> float:
    return float(sacrebleu.corpus_bleu(predictions, [references]).score)


def rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]
    return {k: 100 * mean(s[k].fmeasure for s in scores) for k in ("rouge1", "rouge2", "rougeL")}


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
        "bleu": bleu(predictions, references),
        **rouge(predictions, references),
        "fkgl_prediction": mean(fkgl(p) for p in predictions),
        "fkgl_source": mean(fkgl(s) for s in sources),
        "fkgl_reference": mean(fkgl(r) for r in references),
        "words_prediction": mean(pred_words),
        "length_ratio": mean(p / s for p, s in zip(pred_words, src_words)),
        "n_examples": len(predictions),
    }
    return {k: round(float(v), 2) if k != "n_examples" else v for k, v in results.items()}
