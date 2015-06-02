import sys
import pickle
import os
from dataprocessing.data_utils import *
from dataprocessing.parseTree import ParseNode
from linearembedding.simpleembedding import *
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
#  simpleEmbedder = load
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
rand_gen = randgen(N=1000000, ntrain=len(trainingSet) - 1)
alpha_gen = alphagen(N=1000000, alphastart = 0.002)
simpleEmbedder.train_sgd(trainingSet, trainUtters, rand_gen, simpleEmbedder.annealiter(0.002, 300000), printevery=1000, costevery=10000)
write_pickle("models/", simpleEmbedder, saveFile)
predictions = simpleEmbedder.predict(devSet, devUtters)
print("Dev loss: " + str(simpleEmbedder.compute_display_loss(devSet, devUtters)))
correct = 0
trainpredictions = simpleEmbedder.predict(trainingSet, trainUtters)
for ind, predict in enumerate(trainpredictions): 
    if checkParseEqual(trainingSet[ind][1], trainingSet[ind][0][predict]):
        correct += 1
print("Train accuracy " + str(float(correct)/len(trainpredictions)))
correct = 0
for ind, predict in enumerate(predictions): 
    if checkParseEqual(devSet[ind][1], devSet[ind][0][predict]):
        correct += 1
print("Accuracy " + str(float(correct)/len(predictions)))
