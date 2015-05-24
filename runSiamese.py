import sys
import pickle
import os
from dataprocessing.data_utils import *
from dataprocessing.parseTree import ParseNode
from linearembedding.simpleembedding import *
from siamese.siamese import *
import itertools
from numpy import *

random.seed(100)
args = sys.argv
devCutoff = int(args[1])
trainingExamples = load_pickle(args[2])
correctExamples = load_pickle(args[3])
utterances = load_pickle(args[4])
zipAll = [(all, correct[0], utter) for all, correct, utter in zip(trainingExamples, correctExamples, utterances[:devCutoff]) if len(correct) > 0]
trainingSet = [(all, correct) for all, correct, utter in zipAll]
trainUtters = [utter for all, correct, utter in zipAll]
devExamples = load_pickle(args[5])
devCorrect = load_pickle(args[6])
zipAll = [(all, correct[0], utter) for all, correct, utter in zip(devExamples, devCorrect, utterances[devCutoff:]) if len(correct) > 0]
devSet = [(all, correct) for all, correct, utter in zipAll]
devUtters = [utter for all, correct, utter in zipAll]
vectorDim = int(args[7])
saveFile = args[8]
#if len(args) > 9:
#  loadFile = args[9]
#  siamese = load
worddict = load_pickle("data/word_to_index.pkl")
wv = sqrt(0.1)*random.standard_normal((len(worddict.keys()), vectorDim))
W = sqrt(0.1) * random.standard_normal((350, vectorDim))

siamese = SiameseNet(wv, W, alpha=0.002)
siamese.train_sgd(trainUtters, trainingSet, siamese.randomiter(1000000, 1427), printevery=10000, costevery=10000)
write_pickle("models/", siamese, saveFile)
predictions = siamese.predict(devUtters, devSet)
print("Dev loss: " + str(siamese.compute_display_loss(devUtters, devSet)))
correct = 0
trainpredictions = siamese.predict(trainUtters, trainingSet)
for ind, predict in enumerate(trainpredictions): 
    if checkParseEqual(trainingSet[ind][1], trainingSet[ind][0][predict]):
        correct += 1
print("Train accuracy " + str(float(correct)/len(trainpredictions)))
correct = 0
for ind, predict in enumerate(predictions): 
    if checkParseEqual(devSet[ind][1], devSet[ind][0][predict]):
        correct += 1
print("Accuracy " + str(float(correct)/len(predictions)))
