from recurrentnn.rnn import *
from numpy import *
import sys
from dataprocessing.data_utils import *

args = sys.argv

networkFile = args[1]
wvSave = args[2]

network = load_pickle(networkFile)
wv = network.sparams.L
write_pickle("data/", wv, wvSave)
