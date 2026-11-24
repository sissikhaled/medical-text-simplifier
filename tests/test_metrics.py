"""Sanity checks for src/eval/metrics.py on a tiny hand-made set.

Expected behavior (why each metric is in the project):
- A perfect prediction (= reference) gets BLEU and ROUGE of 100.
- Copying the source scores well below the reference on SARI, because SARI
  rewards deleting the hard words and adding the simple ones.
- A simplification reads easier (lower FKGL) than its source.
"""

import pytest

from src.eval.metrics import compute_all

SOURCES = [
    "Participants receiving the intervention demonstrated a statistically significant "
    "reduction in systolic blood pressure compared with placebo.",
    "Adverse events, predominantly gastrointestinal disturbances, occurred more "
    "frequently in the pharmacological treatment arm.",
    "The certainty of the evidence was downgraded owing to methodological limitations "
    "and imprecision.",
]
REFERENCES = [
    "People who got the treatment had lower blood pressure than people who got a dummy pill.",
    "Side effects, mostly stomach problems, were more common in people who took the medicine.",
    "We are not very confident in the results because the studies had problems and were small.",
]


@pytest.fixture(scope="module")
def perfect():
    return compute_all(SOURCES, REFERENCES, REFERENCES)


@pytest.fixture(scope="module")
def copy():
    return compute_all(SOURCES, SOURCES, REFERENCES)


def test_perfect_prediction_scores_100(perfect):
    assert perfect["bleu"] == pytest.approx(100, abs=0.01)
    assert perfect["rougeL"] == pytest.approx(100, abs=0.01)


def test_sari_prefers_simplification_over_copy(perfect, copy):
    assert perfect["sari"] > copy["sari"] + 30


def test_copy_has_source_readability(copy):
    assert copy["fkgl_prediction"] == copy["fkgl_source"]
    assert copy["length_ratio"] == 1.0


def test_simplification_is_easier_to_read(perfect):
    assert perfect["fkgl_prediction"] < perfect["fkgl_source"]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_all(SOURCES, REFERENCES[:2], REFERENCES)
