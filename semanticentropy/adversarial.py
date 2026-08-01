"""Sample groups where a shared hedge phrase masks a genuine hallucination.

``cluster.equivalent`` clusters by containment: fine for "Paris" vs. "It is
Paris", broken the moment a model hedges at length and only the named
answer differs. Five samples that share a long boilerplate wrapper and
disagree only on the one word that actually answers the question collapse
into a single cluster, scoring 0.00 normalized entropy: exactly backwards,
since disagreement wrapped in confident-sounding hedges is the textbook
shape of a hallucination.

Every group below is a genuine hallucination (five different named
answers) dressed in a consistent hedge, the kind of phrasing a real model
produces when it's uncertain but still speaks fluently. None of the hedge
wrappers are copied between groups, so a fix that special-cases one
specific phrase would not pass all of them.
"""
from __future__ import annotations

# (label, [five sampled answers], all genuinely disagreeing on the named answer)
ADVERSARIAL_GROUPS: list[tuple[str, list[str]]] = [
    ("hedged_composer", [
        "Based on my knowledge and available historical records, the composer credited with this work was likely Verdi.",
        "Based on my knowledge and available historical records, the composer credited with this work was likely Mozart.",
        "Based on my knowledge and available historical records, the composer credited with this work was likely Rossini.",
        "Based on my knowledge and available historical records, the composer credited with this work was likely Puccini.",
        "Based on my knowledge and available historical records, the composer credited with this work was likely Wagner.",
    ]),
    ("hedged_discoverer", [
        "From what I recall about the history of this discovery, it was most likely made by Fleming.",
        "From what I recall about the history of this discovery, it was most likely made by Pasteur.",
        "From what I recall about the history of this discovery, it was most likely made by Curie.",
        "From what I recall about the history of this discovery, it was most likely made by Koch.",
        "From what I recall about the history of this discovery, it was most likely made by Lister.",
    ]),
    ("hedged_year", [
        "According to my understanding of the historical timeline, this event most likely took place around 1847.",
        "According to my understanding of the historical timeline, this event most likely took place around 1852.",
        "According to my understanding of the historical timeline, this event most likely took place around 1861.",
        "According to my understanding of the historical timeline, this event most likely took place around 1839.",
        "According to my understanding of the historical timeline, this event most likely took place around 1874.",
    ]),
    ("hedged_capital", [
        "Drawing on general geographic knowledge, the capital city in question is most likely Astoria.",
        "Drawing on general geographic knowledge, the capital city in question is most likely Kolmar.",
        "Drawing on general geographic knowledge, the capital city in question is most likely Verenza.",
        "Drawing on general geographic knowledge, the capital city in question is most likely Dunmoor.",
        "Drawing on general geographic knowledge, the capital city in question is most likely Belmara.",
    ]),
]

# Written after cluster_v2's salient-anchor rule was frozen against
# ADVERSARIAL_GROUPS above. Evaluated exactly once.
HOLDOUT_GROUPS: list[tuple[str, list[str]]] = [
    ("hedged_ceo_holdout", [
        "Based on publicly available business records, the executive who most likely leads this company is Hart.",
        "Based on publicly available business records, the executive who most likely leads this company is Voss.",
        "Based on publicly available business records, the executive who most likely leads this company is Kern.",
        "Based on publicly available business records, the executive who most likely leads this company is Marsh.",
        "Based on publicly available business records, the executive who most likely leads this company is Reid.",
    ]),
    ("hedged_population_holdout", [
        "As far as I can estimate from general demographic trends, the population figure is roughly 2.3 million.",
        "As far as I can estimate from general demographic trends, the population figure is roughly 850,000.",
        "As far as I can estimate from general demographic trends, the population figure is roughly 4.1 million.",
        "As far as I can estimate from general demographic trends, the population figure is roughly 1.2 million.",
        "As far as I can estimate from general demographic trends, the population figure is roughly 600,000.",
    ]),
    ("hedged_author_holdout", [
        "Given the style and period, this piece was most likely written by Hollow.",
        "Given the style and period, this piece was most likely written by Vance.",
        "Given the style and period, this piece was most likely written by Reed.",
        "Given the style and period, this piece was most likely written by Linden.",
        "Given the style and period, this piece was most likely written by Ashcroft.",
    ]),
]
