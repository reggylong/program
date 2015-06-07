import collections
import sys
import os
import cPickle as pickle
from data_utils import *

class ParseNode: # a node in the tree

  def __init__(self,parseString,strInds, parent=None,children=None):
    self.parseString = parseString
    self.strInds = strInds
    self.parent = parent
    if children is None:
      self.children = []
    else:
      self.children = children

def groupIndsArray(arr, start, end):
  groupedInds = []
  for indSet in arr[start:end]:
    groupedInds += indSet
  return groupedInds

def convertExampleToTree(exFile, wordDict, verbose=False):
  f = open(exFile, "r")
  totalTreeLines = []
  currTreeLines = []
  treeList = []
  for ind, line in enumerate(f):
    if ind == 0: continue
    stripline = line.rstrip()
    if stripline == "lambda": continue	
    if stripline == "call": continue
    if stripline == "g": continue
    if stripline == "var": continue
    if stripline == "": 
      totalTreeLines.append(currTreeLines)
      currTreeLines = []
    else:
      currTreeLines.append(stripline)
  if len(currTreeLines) > 0:
    totalTreeLines.append(currTreeLines)
  for currTree in totalTreeLines:
    currTreeBlock = []
    currFlat = []
    for ind, line in enumerate(currTree):
      if "edu.stanford.nlp.sempre" in line:
        currTreeBlock.append(ind)
    treeToInds = []
    for line in currTree:
      treeToInds.append([wordDict[word] for word in line.split()])
    currFlat.append(groupIndsArray(treeToInds, currTreeBlock[-1], len(treeToInds)))
    if len(currTreeBlock) > 1:
      currFlat.append(groupIndsArray(treeToInds, currTreeBlock[-2], currTreeBlock[-1]))
    if len(currTreeBlock) > 2:
      currFlat.append(groupIndsArray(treeToInds, currTreeBlock[0], currTreeBlock[1]))
    treeList.append(currFlat)
  f.close()
  return treeList

def prepareTrainExamples(trainDir, word_to_ind_path, verbose=False, parsePrefix=""):
  trainingSet = []
  wordToInds = load_pickle(word_to_ind_path)
  exampleFiles = [f for f in os.listdir(trainDir) if os.path.isfile(os.path.join(trainDir, f)) and len(f.split(parsePrefix)) > 1]
  for filename in exampleFiles: 
    trainingSet.append(convertExampleToTree(os.path.join(trainDir, filename), wordToInds, verbose=verbose))
    print(len(trainingSet))
    if verbose:
      print("Finished example " + filename)
  return trainingSet

if __name__ == "__main__":
  args = sys.argv
  examples = args[1]
  savename = args[2]
  parsePrefix = args[3]
  trainingSet = prepareTrainExamples(examples, "../data/word_to_index.pkl", verbose=True, parsePrefix=parsePrefix)
  pickle.dump(trainingSet, open(savename, "wb"))
