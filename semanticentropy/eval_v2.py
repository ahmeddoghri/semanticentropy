"""Does the containment clustering survive a model that hedges at length?

``semanticentropy.eval`` uses short, clean samples: "Paris", "It's Paris".
Real models hedge, wrapping the actual answer in boilerplate like "Based on
my knowledge and available historical records...". When five samples share
that wrapper and disagree only on the one word that answers the question,
the shared wrapper dominates the containment score and the samples
collapse into a single cluster: a genuine hallucination scores 0.00
entropy, "consistent".

This module reruns the detection task against :mod:`semanticentropy.adversarial`'s
hedge-wrapped hallucination groups (every one a real disagreement dressed
in confident, consistent phrasing) for both the original containment check
and the fixed one (:mod:`semanticentropy.cluster_v2`).

    python -m semanticentropy.eval_v2
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Sequence

from .adversarial import ADVERSARIAL_GROUPS, HOLDOUT_GROUPS
from .cluster_v2 import equivalent_v2
from .entropy import SemanticEntropy


def _recall(groups: Sequence[tuple], se: SemanticEntropy) -> Dict:
    caught = 0
    misses: List[str] = []
    for label, samples in groups:
        r = se.score(samples)
        if r.hallucinated:
            caught += 1
        else:
            misses.append(label)
    return {"caught": caught, "n": len(groups), "recall": round(caught / len(groups), 4), "misses": misses}


def build_report() -> Dict:
    se1 = SemanticEntropy(threshold=0.5)
    se2 = SemanticEntropy(threshold=0.5, equivalence=equivalent_v2)
    return {
        "adversarial": {"v1": _recall(ADVERSARIAL_GROUPS, se1), "v2": _recall(ADVERSARIAL_GROUPS, se2)},
        "holdout": {"v1": _recall(HOLDOUT_GROUPS, se1), "v2": _recall(HOLDOUT_GROUPS, se2)},
    }


def format_report(report: Dict) -> str:
    lines = [
        "recall on hedge-wrapped hallucinations (a real disagreement, consistent phrasing)",
        "=" * 74,
        f"{'corpus / version':<22}{'caught':>10}{'recall':>10}",
        "-" * 74,
    ]
    for corpus_name in ("adversarial", "holdout"):
        for v in ("v1", "v2"):
            row = report[corpus_name][v]
            lines.append(f"{corpus_name + ' / ' + v:<22}{row['caught']:>6}/{row['n']:<3}{row['recall']:>10.0%}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    report = build_report()
    print(format_report(report))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
