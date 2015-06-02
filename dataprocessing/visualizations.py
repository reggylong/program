import matplotlib.pyplot as plt
from data_utils import load_pickle, invert_dict
import numpy as np

# Plot top word vectors
def plot_wvecs(filename):
    wv = load_pickle(filename)
    word_to_index = load_pickle("../data/word_to_index.pkl")
    index_to_word = invert_dict(word_to_index)
    counts = load_pickle("../data/counts.pkl")
    reduced_word_to_index = {}

    for k,v in counts.iteritems():
        if v > 1800:
            reduced_word_to_index[k] = v
    
    indices = [word_to_index[k] for k, v in reduced_word_to_index.iteritems()]
    words = [index_to_word[i] for i in indices]
    word_vecs = [wv[i] for i in indices]

    temp = (word_vecs - np.mean(word_vecs, axis=0))
    covariance = 1.0 / len(word_vecs) * temp.T.dot(temp)
    U, S, V = np.linalg.svd(covariance)
    coord = temp.dot(U[:,0:2])

    for i in xrange(len(words)):
        plt.text(coord[i,0], coord[i,1], words[i], bbox=dict(facecolor='green', alpha=0.1))
    
    plt.xlim((np.min(coord[:,0]), np.max(coord[:,0]) + 2))
    plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))
    plt.show()

f = "../data/siamese_word_vectors.pkl"
plot_wvecs(f)
