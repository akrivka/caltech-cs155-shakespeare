# process the syllable dictionary
with open("data/Syllable_dictionary.txt") as file:
    syllable_dictionary = {}
    for line in file:
        data = line.split()
        key, value = data[0], data[1:]
        syllable_dictionary[key] = value


def get_rhyme(filename):
    """
    Returns dictionary with key as initial word and value as its rhymes
    """
    with open(filename) as file:
        rdict = {}
        poemdict = {}
        new_lwl = []
        last_words_list = []
        curr = None
        for line in file:
            if any(char.isdigit() for char in line) or not line.strip():
                poemdict[line] = []
                curr = line
            else:
                poemdict[curr].append(line)

        for poem in poemdict:
            tmp = []
            for line in poemdict[poem]:
                words = line.split()
                last_word = words[-1].strip(".,?!;:)")
                tmp.append(last_word)
            new_lwl.append(tmp)

        for last_words_list in new_lwl:
            if len(last_words_list) == 14:
                for i, key in enumerate(last_words_list):
                    if i == 0 or i == 1 or i == 4 or i == 5 or i == 8 or i == 9:
                        first = last_words_list[i] + " " + str(i)
                        second = last_words_list[i + 2] + " " + str(i)
                        rdict[first] = second
                        rdict[second] = first
                    if i == 12:
                        first = last_words_list[i] + " " + str(i)
                        second = last_words_list[i + 1] + " " + str(i)
                        rdict[first] = second
                        rdict[second] = first

        rdictcombined = {}
        for each in rdict:
            split = each.split()
            newkey = split[0].lower()
            split = rdict[each].split()
            newval = split[0].lower()

            if newkey not in rdictcombined:
                rdictcombined[newkey] = set([])
            rdictcombined[newkey].add(newval)

        return rdictcombined


rhyming_dictionary = get_rhyme("data/shakespeare.txt")

# read word_map
word_to_id = {}
id_to_word = {}
with open("data/word_map.txt") as file:
    for line in file:
        line = line.strip()
        word, id = line.split()
        word_to_id[word] = int(id)
        id_to_word[int(id)] = word
