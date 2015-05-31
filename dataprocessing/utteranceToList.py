import collections
import sys
import os
from parseTree import ParseNode
import pickle
from data_utils import *

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

def convertExample(exFile, wordDict, verbose=False):
  f = open(exFile, "r")
  firstline = f.readline()
  blockList = []
  for block in firstline.split("|||"):
    blockInds = []
    for word in block.split():
      blockInds.append(wordDict[word])
    blockList.append(blockInds)
  return blockList  

def prepareUtteranceExamples(trainDir, word_to_ind_path, verbose=False, parsePrefix=""):
  trainingSet = []
  wordToInds = load_pickle(word_to_ind_path)
  exampleFiles = [f for f in os.listdir(trainDir) if os.path.isfile(os.path.join(trainDir, f)) and len(f.split(parsePrefix)) > 1]
  for filename in exampleFiles: 
    trainingSet.append(convertExample(os.path.join(trainDir, filename), wordToInds, verbose=verbose))
    if verbose:
      print("Finished example " + filename)
  return trainingSet

if __name__ == "__main__":
  args = sys.argv
  examples = args[1]
  savename = args[2]
  parsePrefix = args[3]
  trainingSet = prepareUtteranceExamples(examples, "../data/word_to_index.pkl", verbose=True, parsePrefix=parsePrefix)
  pickle.dump(trainingSet, open(savename, "wb"))
