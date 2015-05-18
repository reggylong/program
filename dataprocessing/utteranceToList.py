import collections
import sys
import os
from parseTree import ParseNode
import pickle

args = sys.argv
utteranceFile = args[1]
wordDictFile = args[2]
utterancePickle = args[3]
utterSet = open(utteranceFile, "r")
wordDict = pickle.load(open(wordDictFile, "rb"))
utterList = []

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

for line in utterSet:
  if len(line.split()) > 0 and line.split()[0] == "(utterance":
    blockList = []
    line = line.split('"')[1]
    for block in line.split("|||"):
      blockInds = []
      for word in block.split():
        blockInds.append(wordDict[word])
      blockList.append(blockInds)
    utterList.append(blockList)  

pickle.dump(utterList, open(utterancePickle, "wb"))
