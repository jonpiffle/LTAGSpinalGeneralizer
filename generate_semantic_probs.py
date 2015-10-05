import pickle
from itertools import tee
from nltk.corpus import BracketParseCorpusReader
from collections import defaultdict

def window(iterable, size, left_nulls=False):
    """
    Iterates over an iterable size elements at a time
    [1, 2, 3, 4, 5], 3 ->
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]
    """

    # Pad left with None's so that the the first iteration is [None, ..., None, iterable[0]]
    if left_nulls:
        iterable = [None] * (size - 1) + iterable

    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)


corpus_root = "wsj"
file_pattern = ".*/wsj_.*\.mrg"
ptb = BracketParseCorpusReader(corpus_root, file_pattern)

counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for sent in ptb.sents():
    for word1, word2, word3, word4, word5 in window(sent, 5):
        counts[-2][word3][word1] += 1
        counts[-1][word3][word2] += 1
        counts[1][word3][word4] += 1
        counts[2][word3][word5] += 1
counts = dict(counts)

for index, outer_dict in counts.items():
    for word, inner_dict in outer_dict.items():
        counts[index][word] = dict(inner_dict)
    counts[index] = dict(outer_dict)

pickle.dump(counts, open('semantic_counts.pickle', 'wb'))

