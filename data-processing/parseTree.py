import collections
import sys
import os

def convertExampleToTree(exFile):
  f = open(exFile, "r")
  totalTreeLines = []
  currTreeLines = []
  for line in f:
    if line.rstrip() == "": 
      totalTreeLines.append(currTreeLines)
    else:
      currTreeLines.append(line)
