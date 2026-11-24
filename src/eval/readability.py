"""Flesch-Kincaid Grade Level (FKGL) that is safe for medical text.

Problem: `textstat` treats every "." as a sentence end. Medical abstracts are
full of decimals ("RR 0.85", "95% CI 1.2 to 3.4") and abbreviations ("e.g.",
"vs.", "et al."), so they get split into many fake short sentences and look
far easier to read than they are. On the Cochrane validation set this made
technical abstracts (11.6) look *easier* than plain summaries (13.7).

Fix: hide the dots that do not end a sentence before scoring. The formula
itself is unchanged (textstat's implementation):
    FKGL = 0.39 * words/sentences + 11.8 * syllables/words - 15.59
"""

from __future__ import annotations

import re

import textstat

# Dots inside numbers: 0.85, 1.2, .05
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)|(?<=\s)\.(?=\d)")
# Common abbreviations in medical/scientific writing whose dot is not a sentence end.
_ABBREVIATIONS = re.compile(
    r"\b(e\.g|i\.e|et al|vs|approx|ca|cf|no|fig|incl|resp|min|max|mg|ml|kg)\.",
    flags=re.IGNORECASE,
)


def normalize_for_readability(text: str) -> str:
    """Remove dots that would be mistaken for sentence boundaries."""
    text = _DECIMAL.sub("", text)
    return _ABBREVIATIONS.sub(lambda m: m.group(1).replace(".", ""), text)


def fkgl(text: str) -> float:
    """Flesch-Kincaid Grade Level (lower = easier to read)."""
    return textstat.flesch_kincaid_grade(normalize_for_readability(text))
