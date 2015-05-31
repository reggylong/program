import matplotlib.pyplot as plt
from data_utils import load_pickle, invert_dict
import numpy as np

def plot_wvecs(filename, word_to_index_path):
    wv = load_pickle(filename)
    word_to_index = load_pickle(word_to_index_path)
    index_to_word = invert_dict(word_to_index)
    temp = (wv - np.mean(wv, axis=0))
    words = [index_to_word[i] for i in xrange(len(index_to_word))]
    covariance = 1.0 / len(wv) * temp.T.dot(temp)
    U, S, V = np.linalg.svd(covariance)
    coord = temp.dot(U[:,0:2])

    for i in xrange(len(words)):
        plt.text(coord[i,0], coord[i,1], words[i], bbox=dict(facecolor='green', alpha=0.1))
    
    plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
    plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))
    plt.show()

f = "../data/siamese_word_vectors.pkl"
g = "../data/word_to_index.pkl"
plot_wvecs(f, g)
