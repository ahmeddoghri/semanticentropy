"""Benchmark: does semantic entropy separate hallucinations from confident answers?

We score every labeled sample group and check whether the hallucination flag
matches the ground-truth label. The headline is how cleanly the two classes
separate on entropy alone, with no labels at inference time and no judge model.

We also print the average normalized entropy for each class, because the gap
between them is the real result: if consistent answers sit near zero and
hallucinations sit near one, a single threshold does the job.

Run it:

    python -m semanticentropy.eval

Deterministic. No model, no network, no API keys.
"""
from __future__ import annotations

from semanticentropy.corpus import GROUPS
from semanticentropy.entropy import SemanticEntropy


def run(threshold: float = 0.5) -> None:
    se = SemanticEntropy(threshold=threshold)

    tp = fp = tn = fn = 0
    ent_halluc = []
    ent_consistent = []

    print(f"semantic entropy benchmark: {len(GROUPS)} questions, "
          f"5 sampled answers each\n")
    print(f"  {'question':>20}  {'truth':>13}  {'norm entropy':>12}  {'flag':>13}")

    for label, is_halluc, samples in GROUPS:
        r = se.score(samples)
        flag = "hallucination" if r.hallucinated else "consistent"
        truth = "hallucination" if is_halluc else "consistent"
        if is_halluc:
            ent_halluc.append(r.normalized)
        else:
            ent_consistent.append(r.normalized)
        if r.hallucinated and is_halluc:
            tp += 1
        elif r.hallucinated and not is_halluc:
            fp += 1
        elif not r.hallucinated and not is_halluc:
            tn += 1
        else:
            fn += 1
        print(f"  {label:>20}  {truth:>13}  {r.normalized:>12.2f}  {flag:>13}")

    n = len(GROUPS)
    acc = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    avg_h = sum(ent_halluc) / len(ent_halluc) if ent_halluc else 0.0
    avg_c = sum(ent_consistent) / len(ent_consistent) if ent_consistent else 0.0

    print(f"\n  accuracy   {acc:.0%}    precision {precision:.0%}    recall {recall:.0%}")
    print(f"  avg normalized entropy: consistent answers {avg_c:.2f}, "
          f"hallucinations {avg_h:.2f}")
    print(f"\nthe two classes separate with a gap of {avg_h - avg_c:.2f} on entropy alone,")
    print("no labels and no second model required. a model that knows the answer")
    print("says the same thing five times; a model that is guessing does not.")


if __name__ == "__main__":
    run()
