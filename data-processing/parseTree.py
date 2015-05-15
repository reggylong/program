import collections
import sys
import os

class ParseNode: # a node in the tree

  def __init__(self,parseString,parent=None,children=[]):
    self.parseString = parseString
    self.parent = parent
    self.children = children

def convertExampleToTree(exFile, verbose=False):
  f = open(exFile, "r")
  totalTreeLines = []
  currTreeLines = []
  treeList = []
  for line in f:
    if line.rstrip() == "": 
      totalTreeLines.append(currTreeLines)
      currTreeLines = []
    else:
      currTreeLines.append(line)
  for tree in totalTreeLines: 
    tabNumsOld = 0
    root = ParseNode(tree[0].strip)
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
      newNode = ParseNode(line.strip(), parent=parent)
      parent.children.append(newNode)
      currNode = newNode
      tabNumsOld = tabNumsNew
    treeList.append(root)
  return treeList

def prepareTrainExamples(trainDir, verbose=False):
  trainingSet = []
  exampleFiles = [f for f in os.listdir(trainDir) if os.path.isfile(os.path.join(trainDir, f))]
  for filename in exampleFiles: 
    trainingSet.append(convertExampleToTree(os.path.join(trainDir, filename), verbose=verbose))
    if verbose:
      print("Finished example " + filename)
if __name__ == "__main__":
  prepareTrainExamples("../parses/iter.0", verbose=True)
