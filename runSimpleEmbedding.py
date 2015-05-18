import sys
import pickle
import os
from data-processing.data_utils import *
from data-processing.parseTree import ParseNode

args = sys.argv
trainingExamples = load_pickle(args[1])
correctExamples = load_pickle(args[2])
trainingSet = [(all, correct) for all, correct in itertools.zip(trainingExamples, correctExamples) if len(correct) > 0]
utterances = load_pickle(args[3])




