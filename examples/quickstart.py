"""Sixty-second tour of semanticentropy.

    python examples/quickstart.py
"""
from semanticentropy.entropy import SemanticEntropy

se = SemanticEntropy(threshold=0.5)

# Five samples from a model that actually knows the answer.
confident = ["Paris", "It is Paris", "Paris.", "The capital is Paris", "Paris is the capital"]
r1 = se.score(confident)
print(f"confident answer: {r1.explain()}")

# Five samples from a model that is making it up.
hallucinated = ["Verenza", "Kolmar", "Astoria", "Dunmoor", "Marne"]
r2 = se.score(hallucinated)
print(f"hallucination:    {r2.explain()}")
