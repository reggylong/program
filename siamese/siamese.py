import collections
from numpy import *
from nnbase import NNBase
import random
import itertools


class SiameseNetwork(NNBase):

    """
    # utterance
    r1 = sum L[x_i]
    h1 = tanh(W * r1 + b1)

    # logical form
    r2 = sum L[x_j]
    h2 = tanh(W * r2 + b2)

    # Cost Function (Frobenius norm)
    J = 1/2||h1-h2||^2 + reg/2||W||^2
    Arguments:
        L0: initial word vectors
        W: initial weight matrix 
    """
    def __init__(self, L0=None, W=None,
                 reg=0.001, alpha=0.01, rseed=943):

        # call on NNBase constructor
        self.hdim = L0.shape[1] # word vector dimensions
        self.vdim = L0.shape[0] # vocab size
        
        W_shape = (0,0)
        if W is None:
            W_shape = W.shape
        else:
            W_shape = (100, self.hdim)
        assert(W_shape != (0,0))

        param_dims = dict(H = (self.hdim, self.hdim), W = W_shape)
        param_dims_sparse = dict(L = L0.shape)
        NNBase.__init__(self, param_dims, param_dims_sparse)
        
        random.seed(rseed)
        self.sparams.L = L0.copy()
        if W is not None:
            self.params.W = W.copy()
        else:
            self.params.W = random_weight_matrix(W_shape[0], W_shape[1]) 

        self.alpha = alpha

    def random_weight_matrix(m, n):
        e = sqrt(6.0) / sqrt(m + n)
        A0 = random.uniform(-e, e, (m,n))
        return A0

    def _acc_grads(self, xs, ys):
        pass

    def compute_single_loss(self, answers, questions):
        pass

    # Loss over a dataset
    def compute_loss(self, X, Y):
        pass
