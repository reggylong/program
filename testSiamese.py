import sys
import pickle
import os
from dataprocessing.data_utils import *
from dataprocessing.parseTree import ParseNode
from siamese.siamese import *
import itertools
from numpy import *

args = sys.argv
trainUtterSet = load_pickle(args[1])
trainingExamples = load_pickle(args[2])
print(trainingExamples[0])
correctExamples = load_pickle(args[3])
print(correctExamples[0])
devUtterSet = load_pickle(args[4])
zipAll = [(all, correct[0], utter) for all, correct, utter in zip(trainingExamples, correctExamples, trainUtterSet) if len(correct) > 0]
trainingSet = [(all, correct) for all, correct, utter in zipAll]
trainUtters = [utter for all, correct, utter in zipAll]
devExamples = load_pickle(args[5])
print len(devExamples)
devCorrect = load_pickle(args[6])
zipAll = [(all, correct[0], utter) for all, correct, utter in zip(devExamples, devCorrect, devUtterSet) if len(correct) > 0]
devSet = [(all, correct) for all, correct, utter in zipAll]
devUtters = [utter for all, correct, utter in zipAll]
loadFile = args[7]
#if len(args) > 9:
#  loadFile = args[9]
#  siamese = load
worddict = load_pickle("data/word_to_index.pkl")
inddict = {}
for word in worddict.keys():
  inddict[worddict[word]] = word
  
def parseToWords(parse1):
  words1 = []
  for ind in parse1[0]:
    words1.append(inddict[ind])
  words2 = []
  for ind in parse1[1]:
    words2.append(inddict[ind])
  return words1, words2
   
network = load_pickle(loadFile)
predictions = network.predict(devUtters, devSet)
print("Dev loss: " + str(network.compute_display_loss(devUtters, devSet)))
correct = 0
trainpredictions = network.predict(trainUtters, trainingSet)
for ind, predict in enumerate(trainpredictions): 
    if checkParseEqual(trainingSet[ind][1], trainingSet[ind][0][predict]):
        correct += 1
        print("************* EX " + str(ind))
    else:
        print("------------- EX " + str(ind))
    print("Correct parse (train): ")
    print(parseToWords(trainingSet[ind][1]))
    print("Predicted parse: ")
    print(parseToWords(trainingSet[ind][0][predict]))
    print("Utterance: ")
    print(parseToWords(trainUtters[ind]))
print("Train accuracy " + str(float(correct)/len(trainpredictions)))
correct = 0
for ind, predict in enumerate(predictions): 
    if checkParseEqual(devSet[ind][1], devSet[ind][0][predict]):
        correct += 1
        print("************* EX " + str(ind))
    else: 
        print("------------- EX " + str(ind))
    print("Correct parse (dev): ") 
    print(parseToWords(devSet[ind][1]))
    print("Wrong parse: ")
    print(parseToWords(devSet[ind][0][predict]))
    print("Utterance: ")
    print(parseToWords(devUtters[ind]))
print("Accuracy " + str(float(correct)/len(predictions)))
