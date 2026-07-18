# 🌫️ semanticentropy

**Catch hallucinations by asking the model the same thing five times and clustering the answers by meaning.**

![CI](https://github.com/ahmeddoghri/semanticentropy/actions/workflows/ci.yml/badge.svg)
![tests](https://img.shields.io/badge/tests-10%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

> **Consistent answers score 0.08 normalized entropy on average. Hallucinations
> score 0.90. No labels, no judge model, just the model contradicting itself.**
> See the separation: `python -m semanticentropy.eval`.

Here is a fact worth sitting with: a model that knows the answer says the same
thing every time you ask it, worded differently. A model that is confabulating
gives you a different name, date, or number each time, with identical
confidence in its voice. That instability is not a vibe, it is measurable, and
you do not need a labeled dataset or a second model to grade the first one.

This is the method from Farquhar et al. (Nature, 2024): sample the model several
times at temperature, cluster the samples by meaning rather than by exact
string, and measure entropy over the clusters. One cluster means the model is
consistent. Several clusters, roughly even in size, means the model is guessing.

The clustering step is the whole trick and the part every naive implementation
gets wrong. "Paris" and "It is Paris" are the same answer worded differently. If
you count them as two you smear the signal and hallucinations stop looking any
different from confidence. semanticentropy clusters by content, not by string,
using a transparent containment check you can read start to finish.

---

## The result in one command

```bash
python -m semanticentropy.eval
```
```
semantic entropy benchmark: 10 questions, 5 sampled answers each

              question          truth  norm entropy           flag
        capital_france     consistent          0.00     consistent
         water_boiling     consistent          0.42     consistent
           speed_light     consistent          0.00     consistent
        largest_planet     consistent          0.00     consistent
        chemical_water     consistent          0.00     consistent
           fake_author  hallucination          1.00  hallucination
          made_up_date  hallucination          1.00  hallucination
   invented_population  hallucination          0.83  hallucination
   nonexistent_capital  hallucination          0.83  hallucination
        fabricated_ceo  hallucination          0.83  hallucination

  accuracy   100%    precision 100%    recall 100%
  avg normalized entropy: consistent answers 0.08, hallucinations 0.90
```

That 0.81 gap between the two classes is the entire result. It costs zero
labeled examples and zero extra model calls beyond the five samples you already
needed to draw. This is a small, illustrative benchmark (ten hand-written
question groups), not a claim about accuracy on your production traffic. Point
the same method at your model's actual samples to get your own number.

## Install

```bash
git clone https://github.com/ahmeddoghri/semanticentropy
cd semanticentropy && pip install -e .
python examples/quickstart.py
```

## Use it

```python
from semanticentropy.entropy import SemanticEntropy

se = SemanticEntropy(threshold=0.5)

# sample your model 5 times at temperature > 0 for the same prompt, then:
samples = ["Paris", "It is Paris", "Paris.", "The capital is Paris", "Paris is the capital"]
result = se.score(samples)

print(result.hallucinated)    # False
print(result.normalized)      # 0.0, all five samples mean the same thing
print(result.explain())       # human-readable breakdown, cluster sizes included
```

## How the clustering actually works

```
for each new sample:
  compare its content words (filler stripped) against each existing cluster
  if it overlaps enough with a cluster's first member -> join that cluster
  otherwise -> start a new cluster

entropy = -sum(p * log2(p) for each cluster's share p of the samples)
normalized = entropy / log2(n_samples)   # so the scale is always [0, 1]
```

The overlap check uses containment, not a plain intersection-over-union: the
smaller answer's content just needs to be mostly present in the larger one, so
"H2O" and "the formula is H2O" match even though one is a fragment of the other.
Numbers are normalized ("300,000" and "300000" tokenize the same) and common
hedges ("about", "roughly", "approximately") are stripped, so magnitude
agreement is what counts, not phrasing.

## Bring your own equivalence check

The bag-of-words containment check keeps this dependency-free and fully
readable. For production, swap `cluster.equivalent(a, b)` for a real
bidirectional-entailment check with an NLI model, which is what the original
paper uses. Nothing else changes: the clustering and entropy math only call
that one function.

## Tests

```bash
pip install pytest && pytest -q      # 10 passing
```

## License

MIT © Ahmed Doghri
