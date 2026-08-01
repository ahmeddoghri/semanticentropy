"""The containment check, fixed for verbose hedged answers.

``cluster.equivalent`` measures containment: the smaller answer's content
words just need to be mostly present in the larger one. That's the right
idea for "Paris" vs. "It is Paris". It breaks down the moment a model
hedges at length: two samples that share a long boilerplate wrapper and
differ only in the one word that actually answers the question score as
near-identical, because the shared wrapper dominates the token overlap.

    a = "Based on my knowledge and available historical records, the
         composer credited with this work was likely Verdi."
    b = "Based on my knowledge and available historical records, the
         composer credited with this work was likely Mozart."

``cluster.equivalent(a, b)`` returns ``True`` (coverage 0.909, since only
"verdi"/"mozart" differ out of eleven content tokens). That is exactly
backwards: these are two different, contradictory answers wrapped in
identical hedge phrasing, the textbook shape of a hallucination this whole
library exists to catch. Fed into ``SemanticEntropy``, five samples like
this collapse into a single cluster and score **0.00 normalized entropy,
"consistent"**, a hallucination invisible to the exact failure mode the
paper targets.

The fix does not touch the containment mechanism, which is still the right
default for short answers. It adds one narrow, high-precision veto: if each
answer contains exactly one *salient* token, an unfiltered word that is
capitalized outside the sentence-initial position, or a number, and those
two salient tokens differ, the answers are never equivalent, no matter how
much surrounding hedge text they share. A capitalized non-initial word or a
number is almost always the actual named answer (Verdi, 1912, Nairobi);
requiring exactly one on each side keeps the rule narrow enough not to
misfire on longer answers that legitimately mention more than one entity.
"""
from __future__ import annotations

import re
from typing import Optional

from .cluster import _FILLER, Equivalence, _core_tokens

# decimals first, so "2.3" stays one token instead of splitting on the dot
_WORD = re.compile(r"\d+\.\d+|[a-zA-Z0-9]+")
_NUMERIC = re.compile(r"\d+(\.\d+)?$")


def _salient_tokens(text: str) -> frozenset[str]:
    text_nc = re.sub(r"(?<=\d),(?=\d)", "", text)
    words = _WORD.findall(text_nc)
    sal = set()
    for i, w in enumerate(words):
        lw = w.lower()
        if lw in _FILLER:
            continue
        if _NUMERIC.match(w):
            sal.add(lw)
        elif w[0].isupper() and i > 0:
            sal.add(lw)
    return frozenset(sal)


def equivalent_v2(a: str, b: str, threshold: float = 0.6) -> bool:
    sa, sb = _salient_tokens(a), _salient_tokens(b)
    if len(sa) == 1 and len(sb) == 1 and sa != sb:
        return False

    ca, cb = _core_tokens(a), _core_tokens(b)
    if not ca and not cb:
        return True
    if not ca or not cb:
        return False
    inter = len(ca & cb)
    coverage = inter / min(len(ca), len(cb))
    return coverage >= threshold


def cluster_by_meaning_v2(
    samples: list[str],
    threshold: float = 0.6,
    equivalence: Optional[Equivalence] = None,
) -> list[list[str]]:
    if equivalence is None:
        def equivalence(a: str, b: str) -> bool:
            return equivalent_v2(a, b, threshold)

    clusters: list[list[str]] = []
    for s in samples:
        placed = False
        for cluster in clusters:
            if equivalence(s, cluster[0]):
                cluster.append(s)
                placed = True
                break
        if not placed:
            clusters.append([s])
    return clusters
