"""Turn meaning-clusters into a semantic entropy score.

Once samples are grouped by meaning, entropy is straightforward: treat each
cluster's share of the samples as a probability, and compute Shannon entropy over
those cluster probabilities.

  - All samples in one cluster  -> entropy 0.0  -> the model is consistent -> trust it
  - Samples spread across clusters -> high entropy -> the model disagrees with
    itself -> likely a hallucination

We report both the raw entropy (in bits) and a normalized [0, 1] version so you
can set a single threshold that works regardless of how many samples you drew.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from semanticentropy.cluster import Equivalence, cluster_by_meaning


@dataclass
class EntropyResult:
    entropy_bits: float
    normalized: float          # entropy / log2(n_samples), in [0, 1]
    n_clusters: int
    n_samples: int
    clusters: list[list[str]]
    hallucinated: bool

    @property
    def top_cluster_share(self) -> float:
        if not self.clusters:
            return 0.0
        return max(len(c) for c in self.clusters) / self.n_samples

    def explain(self) -> str:
        state = "LIKELY HALLUCINATION" if self.hallucinated else "consistent"
        sizes = ", ".join(str(len(c)) for c in sorted(self.clusters, key=len, reverse=True))
        return (f"{state}: {self.n_clusters} meaning(s) across {self.n_samples} samples "
                f"(cluster sizes {sizes}), normalized entropy {self.normalized:.2f}")


class SemanticEntropy:
    """Score a set of answer samples for hallucination via semantic entropy.

    ``threshold`` is the normalized-entropy level at or above which the answer is
    flagged as a likely hallucination. The default of 0.5 flags an answer whose
    samples split roughly evenly across two or more distinct meanings.

    ``equivalence`` swaps the meaning-equivalence check. None means the built-in
    zero-dependency containment check; pass ``semanticentropy.nli.NLIEquivalence``
    (the ``[nli]`` extra) for real bidirectional entailment.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        cluster_threshold: float = 0.6,
        equivalence: Optional[Equivalence] = None,
    ) -> None:
        self.threshold = threshold
        self.cluster_threshold = cluster_threshold
        self.equivalence = equivalence

    def score(self, samples: list[str]) -> EntropyResult:
        n = len(samples)
        if n == 0:
            return EntropyResult(0.0, 0.0, 0, 0, [], hallucinated=False)

        clusters = cluster_by_meaning(samples, self.cluster_threshold, self.equivalence)
        probs = [len(c) / n for c in clusters]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0) + 0.0  # avoid -0.0
        max_entropy = math.log2(n) if n > 1 else 1.0
        normalized = (entropy / max_entropy if max_entropy > 0 else 0.0) + 0.0

        return EntropyResult(
            entropy_bits=entropy,
            normalized=normalized,
            n_clusters=len(clusters),
            n_samples=n,
            clusters=clusters,
            hallucinated=normalized >= self.threshold,
        )
