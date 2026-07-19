"""The real benchmark: a real model, real sampling, real mistakes.

Protocol, straight from Farquhar et al. (Nature 2024):

1. Ask a small local LLM each question once at temperature 0. That greedy
   answer is what you would have shipped. Score it right or wrong against the
   gold answer (lenient containment match).
2. Ask the same question 5 more times at temperature 1.0.
3. Cluster the 5 samples by meaning with bidirectional NLI entailment,
   conditioned on the question.
4. Compute normalized semantic entropy over the clusters.
5. Report AUROC: how well does entropy alone predict which greedy answers
   were wrong, with zero knowledge of the gold answers?

Needs the extras: pip install "semanticentropy[nli]"
Run: python -m semanticentropy.realbench

Generations are cached to .realbench_cache.json (gitignored) so repeat runs
skip the slow part. Everything is seeded; the cached file reproduces.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from semanticentropy.entropy import SemanticEntropy
from semanticentropy.nli import NLIEquivalence
from semanticentropy.realqa import QUESTIONS

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
N_SAMPLES = 5
TEMPERATURE = 1.0
CACHE_PATH = Path(__file__).resolve().parent.parent / ".realbench_cache.json"

_WORD = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    # strip accents so "Márquez" matches "Marquez" before tokenizing
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(_WORD.findall(text.lower()))


def is_correct(answer: str, golds: list[str]) -> bool:
    """Lenient containment: the normalized gold string appears in the answer."""
    na = _norm(answer)
    return any(_norm(g) in na for g in golds if _norm(g))


def auroc(scores: list[float], labels: list[bool]) -> float:
    """AUROC of score predicting label=True, rank-based with tie handling."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _generate_all() -> list[dict]:
    """One greedy answer + N_SAMPLES sampled answers per question, cached."""
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()

    def ask(question: str, sample_seed: int | None) -> str:
        msgs = [
            {"role": "system", "content": "Answer with just the short answer, no explanation."},
            {"role": "user", "content": question},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt")
        kwargs: dict = dict(max_new_tokens=32, pad_token_id=tok.eos_token_id)
        if sample_seed is None:
            kwargs["do_sample"] = False
        else:
            torch.manual_seed(sample_seed)
            kwargs.update(do_sample=True, temperature=TEMPERATURE, top_p=0.95)
        with torch.no_grad():
            out = model.generate(**inputs, **kwargs)
        return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    rows = []
    for qi, (question, golds) in enumerate(QUESTIONS):
        greedy = ask(question, sample_seed=None)
        samples = [ask(question, sample_seed=qi * 100 + k) for k in range(N_SAMPLES)]
        rows.append({"question": question, "golds": golds, "greedy": greedy, "samples": samples})
        print(f"  [{qi + 1:2d}/{len(QUESTIONS)}] {question[:52]:<52} -> {greedy[:30]}")
    CACHE_PATH.write_text(json.dumps(rows, indent=1))
    return rows


def run() -> dict:
    print(f"realbench: {MODEL_NAME}, {len(QUESTIONS)} questions, "
          f"{N_SAMPLES} samples each at T={TEMPERATURE}")
    rows = _generate_all()

    eq = NLIEquivalence()
    se = SemanticEntropy(equivalence=eq)

    entropies: list[float] = []
    wrong: list[bool] = []
    for row in rows:
        eq.question = row["question"]
        result = se.score(row["samples"])
        entropies.append(result.normalized)
        wrong.append(not is_correct(row["greedy"], row["golds"]))

    n_wrong = sum(wrong)
    score = auroc(entropies, wrong)
    avg_wrong = sum(e for e, w in zip(entropies, wrong) if w) / max(n_wrong, 1)
    avg_right = sum(e for e, w in zip(entropies, wrong) if not w) / max(len(wrong) - n_wrong, 1)

    print()
    print(f"greedy accuracy        {(len(wrong) - n_wrong)}/{len(wrong)}"
          f" = {(len(wrong) - n_wrong) / len(wrong):.0%}")
    print(f"avg entropy | correct  {avg_right:.2f}")
    print(f"avg entropy | wrong    {avg_wrong:.2f}")
    print(f"AUROC (entropy -> wrong)  {score:.3f}")
    return {"auroc": score, "n_wrong": n_wrong, "n": len(wrong),
            "avg_right": avg_right, "avg_wrong": avg_wrong}


if __name__ == "__main__":
    run()
