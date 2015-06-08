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
#for ind, correct in enumerate(correctExamples):
  #print(correct)
 # if len(correct) < 3:
  #  print ind

zipAll = [(all, correct[0], utter) for all, correct, utter in zip(trainingExamples, correctExamples, trainUtterSet) if len(correct) > 0]
trainingSet = [(all, correct) for all, correct, utter in zipAll]
trainUtters = [utter for all, correct, utter in zipAll]
trainingSet = trainingSet[:-700]
trainUtters = trainUtters[:-700]
devExamples = load_pickle(args[5])
devCorrect = load_pickle(args[6])
zipAll = [(all, correct[0], utter) for all, correct, utter in zip(devExamples, devCorrect, devUtterSet) if len(correct) > 0]
devSet = [(all, correct) for all, correct, utter in zipAll]
devUtters = [utter for all, correct, utter in zipAll]
vectorDim = int(args[7])
middleDim = int(args[8])
saveFile = args[9]
oldRNN = None
if len(args) > 10:
  loadFile = args[10]
  oldRNN = load_pickle(loadFile)

worddict = load_pickle("data/word_to_index.pkl")

def randgen(N, ntrain):
    for i in range(N): 
        k = random.randint(0, ntrain)
	yield k
def alphagen(N, alphastart):
    curralpha = alphastart
    #print(curralpha)
    for i in range(N):
        if i % (N/2) == 0:
            curralpha /= 3 
        yield curralpha
recurrent = None
if oldRNN is None:
  wv = sqrt(0.1)*random.standard_normal((len(worddict.keys()), vectorDim))
  recurrent = RNN(wv, margin=10, middledim = middleDim, backpropwv=True, alpha=0.002, bptt = 10)
else:
  wv = oldRNN.sparams.L
  H = oldRNN.params.H
  recurrent = RNN(wv, margin=10, middledim = middleDim, backpropwv=True, alpha=0.002, bptt = 10)
  recurrent.params.H = identity(H.shape[0])#H
alpha_gen = alphagen(N=300000, alphastart = 0.002)
trainAccs = []
devAccs = []
for i in xrange(20):
  rand_gen = randgen(N=10000, ntrain=len(trainingSet) - 1)
  costs = recurrent.train_sgd(trainingSet, trainUtters, rand_gen, recurrent.annealiter(0.002, 300000), printevery=1000, costevery=10000)
  print(costs)
  write_pickle("models/", recurrent, saveFile)
  predictions = recurrent.predict(devSet, devUtters)
  print("Dev loss: " + str(recurrent.compute_display_loss(devSet, devUtters)))
  correct = 0
  trainpredictions = recurrent.predict(trainingSet, trainUtters)
  for ind, predict in enumerate(trainpredictions): 
    if checkParseEqual(trainingSet[ind][1], trainingSet[ind][0][predict]):
      correct += 1
  print("Train accuracy " + str(float(correct)/len(trainpredictions)))
  trainAccs.append(float(correct)/len(trainpredictions))
  correct = 0
  for ind, predict in enumerate(predictions): 
    if checkParseEqual(devSet[ind][1], devSet[ind][0][predict]):
      correct += 1
  print("Accuracy " + str(float(correct)/len(predictions)))
  devAccs.append(float(correct)/len(predictions)) 
  print("Train array:", trainAccs)
  print("Dev array:", devAccs)
