import pandas as pd
from dictionaries import syllable_dictionary


punctuation = [
    ",",
    ".",
    "!",
    "?",
    ":",
    "'",
    ";",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
]

# process the sonnets
def process_line(line):
    word_syllables_counts = []
    for word in line.split(" "):
        word = word.lower()
        try:
            counts = syllable_dictionary[word]
        except KeyError:
            for punct in punctuation:
                word = word.replace(punct, "")
                if word in syllable_dictionary:
                    break
            counts = syllable_dictionary[word]
        word_syllables_counts.append((word, counts))
    syllables = process_counts(word_syllables_counts)
    return syllables


def process_counts(word_syllables_counts):
    def dfs(wsc, remaining_syllabes):
        word, counts = wsc[0]
        if len(wsc) == 1:
            for count in counts:
                count = int(count.replace("E", ""))
                if count == remaining_syllabes:
                    return [[(word, count)]]
        else:
            valid_paths = []
            for count in counts:
                if "E" in count:
                    continue
                count = int(count)
                if count < remaining_syllabes:
                    paths = dfs(wsc[1:], remaining_syllabes - count)
                    for path in paths:
                        valid_paths.append([(word, count)] + path)
            return valid_paths
        return []

    valid_paths = dfs(word_syllables_counts, 10)

    if len(valid_paths) == 0:
        return None

    # choose first valid path
    path = valid_paths[0]

    return path


shakespeare = pd.read_csv("data/shakespeare.txt", sep="\t", header=None, names=["line"])
shakespeare = shakespeare[~shakespeare.line.str.contains(r"\d")]

processed_lines = []
for index, row in shakespeare.iterrows():
    line = row["line"].strip()
    processed_line = process_line(line)
    if processed_line is not None:
        new_line = []
        prev_stress = "$"
        for word, count in processed_line:
            prefix = ""
            for i in range(count):
                prefix += prev_stress
                prev_stress = "/" if prev_stress == "$" else "$"
            new_line.append(prefix + word)
        processed_lines.append(new_line)

processed_lines_integers = []
word_map = {}
counter = 0
for line in processed_lines:
    new_line = []
    for word in line:
        if word not in word_map:
            word_map[word] = counter
            id = counter
            counter += 1
        else:
            id = word_map[word]
        new_line.append(id)
    processed_lines_integers.append(new_line)

# save the processed lines integers
with open("data/shakespeare_processed.txt", "w") as file:
    for line in processed_lines_integers:
        file.write(" ".join(str(x) for x in line) + "\n")

# save word_map
with open("data/word_map.txt", "w") as file:
    for key in word_map:
        file.write(str(key) + " " + str(word_map[key]) + "\n")
