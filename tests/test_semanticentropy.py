from semanticentropy.cluster import cluster_by_meaning, equivalent
from semanticentropy.corpus import GROUPS
from semanticentropy.entropy import SemanticEntropy


def test_paraphrases_are_equivalent():
    assert equivalent("Paris", "It is Paris")
    assert equivalent("H2O", "The formula is H2O")


def test_different_answers_not_equivalent():
    assert not equivalent("Paris", "Berlin")
    assert not equivalent("Jupiter", "Saturn")


def test_consistent_answers_cluster_together():
    clusters = cluster_by_meaning(["Paris", "It's Paris", "The capital is Paris"])
    assert len(clusters) == 1


def test_contradictory_answers_split():
    clusters = cluster_by_meaning(["Verenza", "Kolmar", "Astoria"])
    assert len(clusters) == 3


def test_consistent_answer_has_zero_entropy():
    se = SemanticEntropy()
    r = se.score(["Paris", "It's Paris", "Paris.", "The capital is Paris", "Paris"])
    assert r.entropy_bits == 0.0
    assert not r.hallucinated


def test_hallucination_has_high_entropy():
    se = SemanticEntropy()
    r = se.score(["Verenza", "Kolmar", "Astoria", "Dunmoor", "Marne"])
    assert r.normalized > 0.5
    assert r.hallucinated


def test_empty_input_is_safe():
    se = SemanticEntropy()
    r = se.score([])
    assert r.n_samples == 0
    assert not r.hallucinated


def test_top_cluster_share():
    se = SemanticEntropy()
    # 4 say Paris, 1 says Berlin: top cluster share should be 0.8
    r = se.score(["Paris", "Paris", "It's Paris", "Paris.", "Berlin"])
    assert abs(r.top_cluster_share - 0.8) < 1e-9


def test_explain_is_human_readable():
    se = SemanticEntropy()
    assert "HALLUCINATION" in se.score(["A", "B", "C", "D", "E"]).explain()
    assert "consistent" in se.score(["Paris"] * 5).explain()


def test_benchmark_separates_the_classes():
    se = SemanticEntropy()
    correct = sum(
        1 for _, is_h, samples in GROUPS
        if se.score(samples).hallucinated == is_h
    )
    assert correct / len(GROUPS) >= 0.9
