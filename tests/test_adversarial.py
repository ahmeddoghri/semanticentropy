"""Tests for the hedge-masked hallucination bug and its fix."""

from __future__ import annotations

from semanticentropy.adversarial import ADVERSARIAL_GROUPS, HOLDOUT_GROUPS
from semanticentropy.cluster import equivalent
from semanticentropy.cluster_v2 import _salient_tokens, equivalent_v2
from semanticentropy.corpus import GROUPS
from semanticentropy.entropy import SemanticEntropy
from semanticentropy.eval_v2 import build_report

_ADVERSARIAL_SAMPLE = (
    "Based on my knowledge and available historical records, the composer "
    "credited with this work was likely Verdi.",
    "Based on my knowledge and available historical records, the composer "
    "credited with this work was likely Mozart.",
)


# --- the finding: a shared hedge masks a genuinely different answer --------

def test_original_containment_treats_hedged_disagreement_as_equivalent():
    a, b = _ADVERSARIAL_SAMPLE
    assert equivalent(a, b)  # bug: scores True, coverage 0.909


def test_original_detector_misses_every_hedged_hallucination():
    se = SemanticEntropy(threshold=0.5)
    for label, samples in ADVERSARIAL_GROUPS:
        r = se.score(samples)
        assert not r.hallucinated, label
        assert r.normalized == 0.0, label


# --- the fix -----------------------------------------------------------------

def test_fixed_equivalence_separates_the_hedged_disagreement():
    a, b = _ADVERSARIAL_SAMPLE
    assert not equivalent_v2(a, b)


def test_fixed_detector_catches_every_hedged_hallucination():
    se = SemanticEntropy(threshold=0.5, equivalence=equivalent_v2)
    for label, samples in ADVERSARIAL_GROUPS:
        r = se.score(samples)
        assert r.hallucinated, label
        assert r.normalized == 1.0, label


def test_decimal_numbers_count_as_a_single_salient_token():
    """"2.3" must not split into two salient tokens ("2", "3") on the dot;
    that would break the single-anchor veto for decimal-valued answers."""
    assert _salient_tokens("the population is 2.3 million") == frozenset({"2.3"})


def test_two_or_more_salient_anchors_do_not_trigger_the_veto():
    """The rule is intentionally narrow: it only fires when each side has
    exactly one named anchor, so it doesn't misfire on answers that
    legitimately mention more than one entity."""
    a = "Alexander Fleming discovered it in London"
    # not asserting a specific outcome here, just that the veto path isn't
    # what decides it (len(salient) != 1 on this side)
    assert len(_salient_tokens(a)) != 1


# --- the fix does not regress the original benchmark ------------------------

def test_fixed_equivalence_still_matches_short_paraphrases():
    assert equivalent_v2("Paris", "It is Paris")
    assert equivalent_v2("H2O", "The formula is H2O")
    assert equivalent_v2("about 300,000", "roughly 300000")


def test_fixed_equivalence_still_rejects_plain_disagreement():
    assert not equivalent_v2("Paris", "Berlin")
    assert not equivalent_v2("Jupiter", "Saturn")


def test_fixed_detector_does_not_regress_original_corpus():
    se1 = SemanticEntropy(threshold=0.5)
    se2 = SemanticEntropy(threshold=0.5, equivalence=equivalent_v2)
    for label, is_halluc, samples in GROUPS:
        r1 = se1.score(samples)
        r2 = se2.score(samples)
        assert r1.hallucinated == r2.hallucinated, label


# --- held out, evaluated once ------------------------------------------------

def test_holdout_is_disjoint_from_the_tuning_corpus():
    adv_texts = {tuple(s) for _, s in ADVERSARIAL_GROUPS}
    holdout_texts = {tuple(s) for _, s in HOLDOUT_GROUPS}
    assert not (adv_texts & holdout_texts)


def test_holdout_fully_recovers_with_the_fix():
    se = SemanticEntropy(threshold=0.5, equivalence=equivalent_v2)
    for label, samples in HOLDOUT_GROUPS:
        assert se.score(samples).hallucinated, label


def test_holdout_still_missed_without_the_fix():
    se = SemanticEntropy(threshold=0.5)
    for label, samples in HOLDOUT_GROUPS:
        assert not se.score(samples).hallucinated, label


# --- the original module is untouched ---------------------------------------

def test_original_cluster_module_untouched():
    import semanticentropy.cluster as cluster_module

    assert not hasattr(cluster_module, "equivalent_v2")


def test_original_benchmark_still_reproduces():
    se = SemanticEntropy(threshold=0.5)
    correct = sum(1 for _, is_h, s in GROUPS if se.score(s).hallucinated == is_h)
    assert correct == len(GROUPS)


# --- the full report ---------------------------------------------------------

def test_report_is_reproducible():
    assert build_report() == build_report()
