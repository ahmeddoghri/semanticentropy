"""Cluster answer samples by meaning, not by string.

This is the load-bearing idea. "Paris", "It's Paris", and "The capital is Paris."
are one answer. "Paris", "Lyon", "Berlin" are three. If you measure entropy over
raw strings you conflate paraphrase with disagreement, and the signal vanishes.

The real paper uses a natural-language-inference model for bidirectional
entailment. We use a transparent normalized-overlap check instead, so the whole
thing runs with no model and you can read exactly why two samples were judged
equivalent. Swap in an NLI model at this seam and nothing downstream changes.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")

# Filler that carries no answer content. Removing it means "Paris" and
# "The answer is Paris" reduce to the same core token set.
_FILLER = {
    "the", "a", "an", "is", "it", "its", "of", "to", "in", "on", "at", "and",
    "answer", "capital", "that", "this", "would", "be", "i", "think", "believe",
    "probably", "likely", "definitely", "as", "for", "was", "were", "are",
    # approximation hedges: "about 300000" and "roughly 300000" are one answer.
    # note we deliberately keep magnitude words like "million" and "thousand",
    # since "2 million" and "2 thousand" are different answers.
    "about", "roughly", "around", "close", "approximately", "nearly",
    "per", "second", "kilometers", "kilometer", "km", "degrees", "degree",
}


def _core_tokens(text: str) -> frozenset[str]:
    # drop thousands separators so "300,000" and "300000" tokenize the same
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    toks = [t for t in _WORD.findall(text) if t not in _FILLER]
    return frozenset(toks)


def equivalent(a: str, b: str, threshold: float = 0.6) -> bool:
    """True if two answers mean the same thing.

    Uses containment, not symmetric overlap: two answers are equivalent if the
    smaller answer's content is mostly contained in the larger one. This mirrors
    bidirectional entailment cheaply. "Paris" is contained in "It is Paris", so
    they match; "Paris" and "Berlin" share nothing, so they do not. Numbers are
    normalized so "300,000" and "300000" count as the same token.
    """
    ca, cb = _core_tokens(a), _core_tokens(b)
    if not ca and not cb:
        return True
    if not ca or not cb:
        return False
    inter = len(ca & cb)
    # containment: fraction of the SMALLER answer's content found in the larger
    coverage = inter / min(len(ca), len(cb))
    return coverage >= threshold


def cluster_by_meaning(samples: list[str], threshold: float = 0.6) -> list[list[str]]:
    """Group samples into meaning clusters.

    Greedy single-link clustering: each sample joins the first existing cluster
    it is equivalent to, or starts a new one. Good enough for the short factual
    answers this is designed for, and fully deterministic.
    """
    clusters: list[list[str]] = []
    for s in samples:
        placed = False
        for cluster in clusters:
            if equivalent(s, cluster[0], threshold):
                cluster.append(s)
                placed = True
                break
        if not placed:
            clusters.append([s])
    return clusters
