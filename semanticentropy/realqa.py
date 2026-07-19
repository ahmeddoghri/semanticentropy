"""Question set for the real-model benchmark.

Forty short-answer factual questions, hand-curated rather than pulled from
TriviaQA, for two honest reasons: embedding a subset of a multi-gigabyte
dataset adds a download step this repo promised to never have, and a curated
set lets the difficulty span from questions a 0.5B model reliably gets right
to questions it reliably confabulates on. Semantic entropy needs both kinds
present to have anything to predict.

Each entry is (question, [acceptable answers]). Acceptable answers are matched
leniently (case-insensitive containment on normalized tokens), so "Canberra"
accepts "The capital is Canberra." but not "Sydney".
"""
from __future__ import annotations

QUESTIONS: list[tuple[str, list[str]]] = [
    # --- easy: a 0.5B instruct model gets most of these right ---
    ("What is the capital of Australia?", ["Canberra"]),
    ("What is the capital of Canada?", ["Ottawa"]),
    ("What is the chemical symbol for gold?", ["Au"]),
    ("How many continents are there on Earth?", ["seven", "7"]),
    ("What planet is known as the Red Planet?", ["Mars"]),
    ("Who wrote the play Romeo and Juliet?", ["Shakespeare"]),
    ("What is the largest ocean on Earth?", ["Pacific"]),
    ("In which year did World War II end?", ["1945"]),
    ("What is the chemical formula for water?", ["H2O"]),
    ("What is the tallest mountain on Earth above sea level?", ["Everest"]),
    ("Which country has the largest population in the world?", ["India", "China"]),
    ("What gas do plants primarily absorb for photosynthesis?", ["carbon dioxide", "CO2"]),
    ("Who painted the Mona Lisa?", ["Leonardo da Vinci", "da Vinci", "Leonardo"]),
    ("What is the currency of Japan?", ["yen"]),
    ("How many sides does a hexagon have?", ["six", "6"]),
    ("What is the freezing point of water in degrees Celsius?", ["0", "zero"]),
    ("Which planet is closest to the Sun?", ["Mercury"]),
    ("What is the longest river in South America?", ["Amazon"]),
    ("In which city is the Eiffel Tower located?", ["Paris"]),
    ("What is the square root of 144?", ["12", "twelve"]),
    # --- hard: obscure enough that a 0.5B model usually guesses ---
    ("Who won the Nobel Prize in Literature in 1976?", ["Saul Bellow", "Bellow"]),
    ("In which year was the city of St. Petersburg founded?", ["1703"]),
    ("What is the capital of the Canadian province of Saskatchewan?", ["Regina"]),
    ("Who was the second person to walk on the Moon?", ["Buzz Aldrin", "Aldrin"]),
    ("In which year did the Chernobyl disaster occur?", ["1986"]),
    ("What is the smallest country in Africa by land area?", ["Seychelles", "Gambia"]),
    ("Who composed the opera The Magic Flute?", ["Mozart"]),
    ("In which year was the Hubble Space Telescope launched?", ["1990"]),
    ("What is the capital of Kazakhstan?", ["Astana", "Nur-Sultan"]),
    ("Who discovered penicillin?", ["Fleming", "Alexander Fleming"]),
    ("In which year did the Berlin Wall fall?", ["1989"]),
    ("What is the deepest point in the world's oceans called?", ["Challenger Deep", "Mariana"]),
    ("Who wrote the novel One Hundred Years of Solitude?", ["Marquez", "Garcia Marquez", "Gabriel Garcia Marquez"]),
    ("In which year did the Titanic sink?", ["1912"]),
    ("What is the capital of Mongolia?", ["Ulaanbaatar", "Ulan Bator"]),
    ("Which element has the atomic number 79?", ["gold"]),
    ("Who was the first woman to win a Nobel Prize?", ["Marie Curie", "Curie"]),
    ("In which decade was the first email sent?", ["1970s", "1971"]),
    ("What is the second-longest river in Africa?", ["Congo"]),
    ("Which country hosted the 1968 Summer Olympics?", ["Mexico"]),
]
