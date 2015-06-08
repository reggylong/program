import collections
import sys
import os
from parseTree import ParseNode
import cPickle as pickle

args = sys.argv
treePickle = args[1]
flatPickle = args[2]
trainingSet = pickle.load(open(treePickle, "rb"))
flattenedTraining = []
if len(args) < 4:
  skipInd = 0
else:
  skipInd = int(args[3])

#def treeToArrays(node):
#  currArray = []
#  print(node.parseString)
#  if len(node.children) == 0: 
#    currArray.append((node.parseString, node.strInds))
#    return currArray
#  for childNode in node.children:
#    nextArr = treeToArrays(childNode)
#    currArray.append(nextArr)
#  currArray.append((node.parseString, node.strInds))
#  return currArray

def treeToArrays(node):
  topLevel = []
  secondLevel = []
  topLevel += node.strInds
  for childNode in node.children[1:]:
    topLevel += childNode.strInds
  secondLevel += node.children[0].strInds
  for childNode in node.children[0].children:
    secondLevel += childNode.strInds
  return topLevel, secondLevel

for example in trainingSet:
  exampleArr = []
  for tree in example:
    flat = treeToArrays(tree)
    exampleArr.append(flat)
  flattenedTraining.append(exampleArr)

pickle.dump(flattenedTraining[skipInd:], open(flatPickle, "wb")) 
