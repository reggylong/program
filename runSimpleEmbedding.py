import sys
import pickle
import os
from dataprocessing.data_utils import *
from dataprocessing.parseTree import ParseNode
from linearembedding.simpleembedding import SimpleLinear
import itertools
from numpy import *

args = sys.argv
trainingExamples = load_pickle(args[1])
correctExamples = load_pickle(args[2])
utterances = load_pickle(args[3])
zipAll = [(all, correct[0], utter) for all, correct, utter in zip(trainingExamples, correctExamples, utterances) if len(correct) > 0]
trainingSet = [(all, correct) for all, correct, utter in zipAll]
utterances = [utter for all, correct, utter in zipAll]
vectorDim = int(args[4])
saveFile = args[5]
worddict = load_pickle("data/word_to_index.pkl")
wv = sqrt(0.1)*random.standard_normal((len(worddict.keys()), vectorDim))

def randgen(N, ntrain):
    for i in range(N): 
        yield random.randint(0, ntrain)

def alphagen(N, alphastart):
    curralpha = alphastart
    print(curralpha)
    for i in range(N):
        if i % (N/2) == 0:
            curralpha /= 3 
        yield curralpha

simpleEmbedder = SimpleLinear(W = wv, alpha=0.002)
rand_gen = randgen(N=2000000, ntrain=len(trainingSet) - 1)
alpha_gen = alphagen(N=2000000, alphastart = 0.002)
simpleEmbedder.train_sgd(trainingSet, utterances, rand_gen, simpleEmbedder.annealiter(0.002, 300000), printevery=1000, costevery=1000)
write_pickle("models/", simpleEmbedder, saveFile)
