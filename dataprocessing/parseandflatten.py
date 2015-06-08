import collections
import sys
import os
import cPickle as pickle
from data_utils import *
import itertools

class ParseNode: # a node in the tree

  def __init__(self,parseString,strInds, parent=None,children=None):
    self.parseString = parseString
    self.strInds = strInds
    self.parent = parent
    if children is None:
      self.children = []
    else:
      self.children = children


def checkListsEqual(list1, list2):
    if len(list1) != len(list2):
        return False
    for l1, l2 in itertools.izip(list1, list2):
        if l1 != l2:
            return False
    return True

def checkParseEqual(parse1, parse2):
    if len(parse1) != len(parse2):
        return False
    for list1, list2 in itertools.izip(parse1, parse2):
        if not checkListsEqual(list1, list2):
            return False
    return True

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
    stripline = line.strip()
    if stripline == "lambda": continue	
    if stripline == "call": continue
    if stripline == "g": continue
    if stripline == "gv": continue
    if stripline == "var": continue
    if stripline == "boolean": continue
    if stripline == "number": continue
    if stripline == "string": continue
    if stripline == "": 
      totalTreeLines.append(currTreeLines)
      currTreeLines = []
    else:
      currTreeLines.append(stripline)
  if len(currTreeLines) > 0:
    totalTreeLines.append(currTreeLines)
  for currTree in totalTreeLines:
    currTreeBlock = []
    currFlat = [[], [], []]
    parseStart = 0
    for ind, line in enumerate(currTree):
      if "edu.stanford.nlp.sempre" in line:
        if "parseStartState" not in line:
          parseStart += 1
	currTreeBlock.append(ind)
    treeToInds = []
    for line in currTree:
      treeToInds.append([wordDict[word] for word in line.split()])
    for i in xrange(parseStart):
      currFlat[i] = groupIndsArray(treeToInds, currTreeBlock[i], currTreeBlock[i + 1])
    currFlat[2] = groupIndsArray(treeToInds, currTreeBlock[-1], len(treeToInds))
    treeList.append(currFlat)
  uniqueSet = []
  for i in xrange(len(treeList)):
    shouldAdd = True
    for j in xrange(i):
      if checkParseEqual(treeList[i], treeList[j]):
	shouldAdd = False
	break
    if shouldAdd:
      uniqueSet.append(treeList[i])
  f.close()
  return uniqueSet

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
