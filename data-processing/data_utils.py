import csv
from collections import Counter
import pickle

def count_words(filename):
    counts = Counter()
    with open(filename) as tsv:
        stopafter = 10
        for line in csv.reader(tsv, dialect="excel-tab"):
            counts[line[0]] += 1
            counts[line[2]] += 1
            tokens = line[1].split()
            for token in tokens:
                counts[token] += 1
    return counts

def write_pickle(obj, output_name):
    output = open(output_name, "wb")
    pickle.dump(obj, output, -1)
    output.close()

def load_pickle(filename):
    f = open(filename, "rb")
    return pickle.load(f)

def create_word_to_index(counter):
    index = 0
    word_to_index = Counter()
    for key in counter:
        word_to_index[key] = index
        index += 1
    return word_to_index

def invert_dict(counter):
    return {v: k for k, v in counter.iteritems()}

