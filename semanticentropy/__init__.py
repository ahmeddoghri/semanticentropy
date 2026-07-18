"""semanticentropy: catch hallucinations by asking the model the same thing five times.

When a model knows an answer, it says the same thing every time you ask, in
different words. When it is making something up, it contradicts itself: ask five
times and you get five different names, dates, or numbers. That instability is a
measurable signal, and it does not need a labeled dataset or a second judge model.

The trick, from Farquhar et al. (Nature, 2024), is that you cannot just measure
entropy over the raw strings. "Paris" and "It is Paris" are the same answer worded
differently, and counting them as two would hide the model's actual confidence.
So you first cluster the samples by meaning (semantic equivalence), then measure
entropy over the clusters. High semantic entropy means the model disagrees with
itself about what the answer *is*, which is the fingerprint of a confabulation.

This package implements that pipeline with a transparent, dependency-free
equivalence check, and ships a benchmark showing it separates confident correct
answers from hallucinated ones.
"""
from semanticentropy.cluster import cluster_by_meaning
from semanticentropy.entropy import SemanticEntropy, EntropyResult

__all__ = ["cluster_by_meaning", "SemanticEntropy", "EntropyResult"]

__version__ = "0.1.0"
