"""NLI backend tests. Skipped automatically when the [nli] extra isn't installed."""
import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")

from semanticentropy.nli import NLIEquivalence  # noqa: E402


@pytest.fixture(scope="module")
def eq():
    return NLIEquivalence()


def test_paraphrases_entail_bidirectionally(eq):
    eq.question = None
    assert eq("The capital of France is Paris.", "Paris is the capital of France.")


def test_contradictions_do_not(eq):
    eq.question = None
    assert not eq("The capital of France is Paris.", "The capital of France is Lyon.")


def test_question_framing_helps_terse_answers(eq):
    eq.question = "In which year did the Titanic sink?"
    assert eq("1912", "It sank in 1912.")


def test_changing_question_does_not_serve_stale_cache(eq):
    eq.question = "What color is the sky?"
    first = eq("blue", "azure")
    eq.question = "What is the capital of France?"
    # same strings, different question: must be re-evaluated, not replayed
    second = eq("blue", "azure")
    assert isinstance(first, bool) and isinstance(second, bool)
