import collections
import sys
import os
from parseTree import ParseNode
import pickle

args = sys.argv
treePickle = args[1]
flatPickle = args[2]
trainingSet = pickle.load(open(treePickle, "rb"))
flattenedTraining = []

def treeToArrays(node):
  currArray = []
  print(node.parseString)
  if len(node.children) == 0: 
    currArray.append((node.parseString, node.strIndex))
    return currArray
  for childNode in node.children:
    nextArr = treeToArrays(childNode)
    currArray.append(nextArr)
  currArray.append((node.parseString, node.strIndex))
  return currArray

for example in trainingSet:
  exampleArr = []
  for tree in example:
    flat = treeToArrays(tree)
    exampleArr.append(flat)
  flattenedTraining.append(exampleArr)

pickle.dump(flattenedTraining, open(flatPickle, "wb")) 
