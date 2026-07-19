"""The equivalence seam and realbench scoring plumbing, no models required."""
from semanticentropy.cluster import cluster_by_meaning
from semanticentropy.entropy import SemanticEntropy
from semanticentropy.realbench import auroc, is_correct


def test_custom_equivalence_is_used():
    # an equivalence that says everything matches -> one cluster
    clusters = cluster_by_meaning(["a", "b", "c"], equivalence=lambda a, b: True)
    assert len(clusters) == 1
    # an equivalence that says nothing matches -> all singletons
    clusters = cluster_by_meaning(["a", "b", "c"], equivalence=lambda a, b: False)
    assert len(clusters) == 3


def test_semantic_entropy_accepts_equivalence():
    se = SemanticEntropy(equivalence=lambda a, b: True)
    r = se.score(["totally", "different", "strings"])
    assert r.n_clusters == 1
    assert r.entropy_bits == 0.0


def test_is_correct_lenient_containment():
    assert is_correct("The capital is Canberra.", ["Canberra"])
    assert is_correct("Gabriel García Márquez", ["Garcia Marquez"])  # accents stripped
    assert not is_correct("Sydney", ["Canberra"])


def test_auroc_orders_and_ties():
    # perfect separation
    assert auroc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]) == 1.0
    # random-equivalent: all tied scores
    assert auroc([0.5, 0.5, 0.5, 0.5], [True, True, False, False]) == 0.5
    # perfect inversion
    assert auroc([0.1, 0.2, 0.9, 0.8], [True, True, False, False]) == 0.0
