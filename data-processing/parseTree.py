import collections
import sys
import os

def TreeNode
def convertExampleToTree(exFile):
  f = open(exFile, "r")
  totalTreeLines = []
  currTreeLines = []
  for line in f:
    if line.rstrip() == "": 
      totalTreeLines.append(currTreeLines)
    else:
      currTreeLines.append(line)
  for tree in totalTreeLines: 
    tabNumsOld = 1
    root = TreeNode()
    currNode = root
    root.parseString = tree[0].strip()
    for line in tree[1:]:
      tabNumsNew = len(line.split("\t")) - 1
      if tabNumsNew == tabNumsOld: 
        parent = currNode.parent
        newNode = 
        parent.  
     
    

