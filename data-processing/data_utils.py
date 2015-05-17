from collections import Counter
import pickle
import os

# We assume the only files in the director
# are the files that we want to count
def count_words(directory):
    counts = Counter()
    for filename in os.listdir(directory):
        with open(directory + filename) as f:
            content = f.read()
            words = content.split()
            for word in words:
                counts[word.strip()] += 1
    return counts

def write_pickle(path, obj, output_name):
    output = open(path + output_name + ".pkl", "wb")
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

def get_and_write_counts(directory):
    data_loc = "../data/"
    counts = count_words(directory)
    word_to_index = create_word_to_index(counts)
    write_pickle(data_loc, counts, "counts")
    write_pickle(data_loc, word_to_index, "word_to_index")
