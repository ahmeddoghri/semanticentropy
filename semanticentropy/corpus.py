"""Labeled sample groups: questions where the model was confident (and right)
versus questions where it confabulated.

Each entry is five sampled answers to one question, exactly what you would get
by calling a model five times at temperature. The confident cases restate the
same fact in different words. The hallucinated cases contradict themselves,
which is the pattern semantic entropy is built to detect.

These sample groups are illustrative and hand-written to be realistic. The point
is to show the *method* separates the two classes; point it at your own model's
samples to get your own numbers.
"""
from __future__ import annotations

# (label, is_hallucination, [five sampled answers])
GROUPS: list[tuple[str, bool, list[str]]] = [
    # --- confident, consistent answers (not hallucinations) ---
    ("capital_france", False, [
        "Paris", "The capital is Paris", "It's Paris", "Paris.", "Paris is the capital",
    ]),
    ("water_boiling", False, [
        "100 degrees Celsius", "It boils at 100 C", "100 Celsius at sea level",
        "100 degrees C", "The boiling point is 100 Celsius",
    ]),
    ("speed_light", False, [
        "about 300,000 km per second", "roughly 300000 kilometers per second",
        "300,000 km/s approximately", "close to 300000 km per second",
        "around 300,000 kilometers a second",
    ]),
    ("largest_planet", False, [
        "Jupiter", "It is Jupiter", "Jupiter is the largest", "Jupiter.",
        "The largest planet is Jupiter",
    ]),
    ("chemical_water", False, [
        "H2O", "It's H2O", "The formula is H2O", "H2O.", "water is H2O",
    ]),
    # --- hallucinations: the model contradicts itself across samples ---
    ("fake_author", True, [
        "It was written by James Hollow", "The author is Margaret Vance",
        "I believe it was Thomas Reed", "Probably by Sarah Linden",
        "It was Robert Ashcroft",
    ]),
    ("made_up_date", True, [
        "In 1847", "Around 1852", "It happened in 1839", "Likely 1861", "In 1844",
    ]),
    ("invented_population", True, [
        "About 2.3 million", "Roughly 850,000", "Around 4 million",
        "Close to 1.2 million", "Maybe 600,000",
    ]),
    ("nonexistent_capital", True, [
        "The capital is Verenza", "It's Kolmar", "I think it's Astoria",
        "Probably Verenza", "It could be Dunmoor",
    ]),
    ("fabricated_ceo", True, [
        "The CEO is David Kern", "It's run by Elena Marsh", "I believe Peter Voss",
        "Maybe Angela Hart", "The CEO is David Kern",
    ]),
]
