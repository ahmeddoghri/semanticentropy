"""Real bidirectional entailment, the way the paper does it.

The zero-dependency containment check in ``cluster.py`` is a good stand-in for
short factual answers, but it is still string matching. Farquhar et al. (Nature
2024) judge two answers equivalent when each entails the other according to a
natural-language-inference model. This module plugs exactly that into the
``equivalence`` seam, so nothing downstream changes.

This is the one optional heavyweight in the repo. It needs the ``[nli]`` extra:

    pip install "semanticentropy[nli]"

Everything else stays dependency-free.
"""
from __future__ import annotations

from typing import Optional

_INSTALL_HINT = (
    "The NLI backend needs torch and transformers. "
    'Install them with: pip install "semanticentropy[nli]"'
)


class NLIEquivalence:
    """Meaning equivalence via bidirectional entailment.

    Two answers are the same answer when A entails B and B entails A, judged by
    a cross-encoder NLI model. If a ``question`` is set, both answers are framed
    as "Question: ... Answer: ..." before the check, which is how the original
    paper conditions entailment on the question being answered. That matters:
    "1912" and "1912, when the Titanic sank" entail each other as answers to the
    same question, but a bare NLI model can miss that without the frame.

    Usable anywhere the ``equivalence`` seam is accepted:

        eq = NLIEquivalence()
        eq.question = "In which year did the Titanic sink?"
        SemanticEntropy(equivalence=eq).score(samples)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        entail_threshold: float = 0.5,
        question: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:  # pragma: no cover - exercised only without extras
            raise ImportError(_INSTALL_HINT) from e

        self._torch = torch
        self.model_name = model_name
        self.entail_threshold = entail_threshold
        self.question = question
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        if device:
            self._model.to(device)
        self._device = device
        # find the entailment label index instead of assuming a layout
        id2label = {i: str(lbl).lower() for i, lbl in self._model.config.id2label.items()}
        self._entail_idx = next(i for i, lbl in id2label.items() if "entail" in lbl)
        # cache keyed on (question, a, b) so changing the question can't serve
        # stale verdicts from a previous question's framing
        self._cache: dict[tuple[Optional[str], str, str], bool] = {}

    def _frame(self, answer: str) -> str:
        if self.question:
            return f"Question: {self.question} Answer: {answer}"
        return answer

    def _entails(self, premise: str, hypothesis: str) -> float:
        inputs = self._tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        if self._device:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
        probs = self._torch.softmax(logits, dim=-1)[0]
        return float(probs[self._entail_idx])

    def _bidirectional(self, a: str, b: str) -> bool:
        fa, fb = self._frame(a), self._frame(b)
        return (
            self._entails(fa, fb) >= self.entail_threshold
            and self._entails(fb, fa) >= self.entail_threshold
        )

    def __call__(self, a: str, b: str) -> bool:
        if a == b:
            return True
        # cache key is order-independent; entailment itself is checked both ways
        first, second = sorted((a.strip(), b.strip()))
        key = (self.question, first, second)
        if key not in self._cache:
            self._cache[key] = self._bidirectional(first, second)
        return self._cache[key]
