import sys
import pickle
import os
from dataprocessing.data_utils import *
from dataprocessing.parseTree import ParseNode
from recurrentnn.rnn import *
from recurrentnn.rnnlm import *
import itertools
from numpy import *

args = sys.argv
trainUtterSet = load_pickle(args[1])
trainingExamples = load_pickle(args[2])
correctExamples = load_pickle(args[3])
devUtterSet = load_pickle(args[4])
zipAll = [(all, correct[0], utter) for all, correct, utter in zip(trainingExamples, correctExamples, trainUtterSet) if len(correct) > 0]
trainingSet = [(all, correct) for all, correct, utter in zipAll]
trainUtters = [utter for all, correct, utter in zipAll]
devExamples = load_pickle(args[5])
devCorrect = load_pickle(args[6])
zipAll = [(all, correct[0], utter) for all, correct, utter in zip(devExamples, devCorrect, devUtterSet) if len(correct) > 0]
devSet = [(all, correct) for all, correct, utter in zipAll]
devUtters = [utter for all, correct, utter in zipAll]
vectorDim = int(args[7])
saveFile = args[8]
#if len(args) > 9:
#  loadFile = args[9]
#  recurrent = load
worddict = load_pickle("data/word_to_index.pkl")

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

wv = sqrt(0.1)*random.standard_normal((len(worddict.keys()), vectorDim))
recurrent = RNNLM(wv, alpha=0.002, bptt = 3)
rand_gen = randgen(N=600000, ntrain=len(trainingSet) - 1)
alpha_gen = alphagen(N=600000, alphastart = 0.002)
recurrent.train_sgd(trainingSet, trainUtters, rand_gen, recurrent.annealiter(0.002, 300000), printevery=1000, costevery=10000)
write_pickle("models/", recurrent, saveFile)
print("Dev loss: " + str(recurrent.compute_display_loss(devSet, devUtters)))
