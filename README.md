# 🌫️ semanticentropy

**Catch hallucinations by asking the model the same thing five times and clustering the answers by meaning.**

![CI](https://github.com/ahmeddoghri/semanticentropy/actions/workflows/ci.yml/badge.svg)
![tests](https://img.shields.io/badge/tests-18%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

> **On a real model (Qwen2.5-0.5B answering 40 trivia questions), semantic
> entropy predicts which answers are wrong with 0.878 AUROC. No labels, no
> judge model, just the model contradicting itself.**
> Zero-dep demo: `python -m semanticentropy.eval`. Real thing:
> `python -m semanticentropy.realbench`.

Here's a fact worth sitting with, ideally with a coffee: a model that knows
the answer says the same thing every time you ask it, worded differently. A
model that's confabulating gives you a different name, date, or number each
time, with the exact same unbothered confidence in its voice either way,
because confidence was never actually tied to being right. That instability
isn't a vibe, it's measurable, and you don't need a labeled dataset or a
second, more expensive model to babysit the first one.

This is the method from Farquhar et al. (Nature, 2024): sample the model several
times at temperature, cluster the samples by meaning rather than by exact
string, and measure entropy over the clusters. One cluster means the model is
consistent. Several clusters, roughly even in size, means the model is guessing.

The clustering step is the whole trick and the part every naive implementation
botches. "Paris" and "It is Paris" are the same answer said two different
ways by someone in a hurry. Count them as two distinct answers and you smear
the signal until hallucinations stop looking any different from confidence,
which defeats the entire point. semanticentropy clusters by content, not by
string, using a transparent containment check you can read start to finish
without a linguistics degree.

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

## The real thing: a real model, making real mistakes

The synthetic benchmark above proves the math. This one proves the method. It
runs the actual protocol from the paper against a real local LLM, end to end,
still with no API keys:

```bash
pip install -e ".[nli]"                  # torch + transformers, the one opt-in heavyweight
python -m semanticentropy.realbench
```

What it does: asks Qwen2.5-0.5B-Instruct 40 short factual questions, once
greedily (that's the answer you would have shipped) and five more times at
temperature 1.0. The five samples get clustered by bidirectional NLI
entailment (cross-encoder/nli-deberta-v3-small), conditioned on the question,
exactly as in Farquhar et al. Then one question: does entropy alone predict
which greedy answers were wrong, without ever seeing a gold label?

```
greedy accuracy        31/40 = 78%
avg entropy | correct  0.25
avg entropy | wrong    0.83
AUROC (entropy -> wrong)  0.878
```

It does. When the model knew the answer (Canberra, 1912, Fleming), the five
samples agreed and entropy stayed low. When it was confabulating (it thinks
J.D. Salinger won the 1976 Nobel and that Verdi wrote The Magic Flute), the
samples scattered and entropy spiked. 0.878 AUROC from nothing but the model
disagreeing with itself.

Honest caveats: one small model, 40 hand-curated questions (embedded in the
repo, no dataset download), lenient string matching for correctness labels,
and generations cached to a local JSON so re-runs are fast. This is a faithful
small-scale reproduction of the paper's result, not a leaderboard entry.

## Bring your own equivalence check

Both checks plug into the same seam, and so can yours:

```python
from semanticentropy.entropy import SemanticEntropy
from semanticentropy.nli import NLIEquivalence   # needs the [nli] extra

eq = NLIEquivalence()
eq.question = "In which year did the Titanic sink?"
se = SemanticEntropy(equivalence=eq)             # or equivalence=any (a, b) -> bool
```

Leave `equivalence` unset and you get the zero-dependency containment check,
which is wrong slightly more often and infinitely easier to debug.

## Tests

```bash
pip install pytest && pytest -q      # 14 passing, +4 NLI tests when [nli] is installed
```

## License

MIT © Ahmed Doghri
