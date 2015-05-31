import collections
import sys
import os
import pickle
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

def convertExampleToTree(exFile, wordDict, verbose=False):
  f = open(exFile, "r")
  totalTreeLines = []
  currTreeLines = []
  treeList = []
  for ind, line in enumerate(f):
    if ind == 0: continue
    if line.rstrip() == "": 
      totalTreeLines.append(currTreeLines)
      currTreeLines = []
    else:
      currTreeLines.append(line)
  if len(currTreeLines) > 0:
    totalTreeLines.append(currTreeLines)
  for tree in totalTreeLines: 
    tabNumsOld = 0
    parseString = tree[0].strip()
    indArr = [wordDict[word] for word in parseString.split()]
    root = ParseNode(parseString, indArr)
    currNode = root
    for line in tree[1:]:
      tabNumsNew = len(line.split("\t")) - 1
      parent = None
      if tabNumsNew == tabNumsOld: 
        parent = currNode.parent
      else: 
        if tabNumsNew > tabNumsOld: 
          parent = currNode
        else: 
          parent = currNode.parent.parent
      parseString = line.strip()
      indArr = [wordDict[word] for word in parseString.split()]
      newNode = ParseNode(parseString, indArr, parent=parent)
      parent.children.append(newNode)
      currNode = newNode
      tabNumsOld = tabNumsNew
    treeList.append(root)
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
