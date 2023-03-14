#!/usr/bin/env python3

import sys
from HMM import load_HMM
from dictionaries import rhyming_dictionary, syllable_dictionary, word_to_id, id_to_word


# helpers
def prefix_word(word):
    if word in syllable_dictionary:
        counts = syllable_dictionary[word]
    else:
        for punct in [",", ".", "!", "?", ":", "'", ";", "(", ")", "[", "]", "{", "}"]:
            word = word.replace(punct, "")
            if word in syllable_dictionary:
                break
        counts = syllable_dictionary[word]

    end_counts = [c for c in counts if c[0] == "E"]
    count = end_counts[0] if len(end_counts) > 0 else counts[0]

    stress = "/"
    prefix = ""
    for _ in range(int(count.replace("E", ""))):
        prefix = stress + prefix
        stress = "/" if stress == "$" else "$"

    return prefix + word


def print_emission(emission, remove_special=False):
    if remove_special:
        print(
            " ".join(
                [id_to_word[id].replace("$", "").replace("/", "") for id in emission]
            )
        )
    else:
        print(" ".join([id_to_word[id] for id in emission]))


def invert_words(words):
    return [
        word_to_id[prefix_word(word)]
        for word in words
        if prefix_word(word) in word_to_id
    ]


# read command line argument for file
filename = sys.argv[1]
remove_special = len(sys.argv) <= 2

hmm = load_HMM(filename)

line_pairs = []
for _ in range(7):
    emission1, _ = hmm.generate_emission(
        10, end_words=invert_words(rhyming_dictionary.keys())
    )
    last_word = id_to_word[emission1[-1]].replace("$", "").replace("/", "")
    emission2, _ = hmm.generate_emission(
        10, end_words=invert_words(rhyming_dictionary[last_word])
    )

    line_pairs.append((emission1, emission2))

poem = [
    [
        line_pairs[0][0],  # a1
        line_pairs[1][0],  # b1
        line_pairs[0][1],  # a2
        line_pairs[1][1],  # b2
    ],
    [
        line_pairs[2][0],  # c1
        line_pairs[3][0],  # d1
        line_pairs[2][1],  # c2
        line_pairs[3][1],  # d2
    ],
    [
        line_pairs[4][0],  # e1
        line_pairs[5][0],  # f1
        line_pairs[4][1],  # e2
        line_pairs[5][1],  # f2
    ],
    [line_pairs[6][0], line_pairs[6][1]],  # g1  # g2
]

for i, stanza in enumerate(poem):
    for line in stanza:
        if i == 3:
            print("  ", end="")
        print_emission(line, remove_special=remove_special)
    print()
