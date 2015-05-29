import collections
import sys
import os
from parseTree import ParseNode
import pickle
import itertools
from data_utils import *

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

args = sys.argv

dataExamples = load_pickle(args[1])
correctExamples = load_pickle(args[2])
zipAll = [(all, correct[0]) for all, correct in zip(dataExamples, correctExamples) if len(correct) > 0]
dataSet = [(all, correct) for all, correct in zipAll]

for ind, (allList, correct) in enumerate(dataSet):
  foundCorrect = False
  for exampleParse in allList:
    if checkParseEqual(exampleParse, correct): 
      foundCorrect = True
      break
  if not foundCorrect: 
    print(ind) 
  
