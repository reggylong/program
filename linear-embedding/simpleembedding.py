import numpy as np
from nnbase import NNBase

class SimpleLinear(NNBase):
    """
    """
    def __init__(self, W_dim=None, W=None,
                 reg=0.001, alpha=0.01, rseed=10):
        W_shape = W.shape
        if W is None:
            assert(W_dim is not None)
            W_shape = W_dim
        
        # call on NNBase constructor
        param_dims = dict()
        param_dims_sparse = dict(W=W_shape)               
        NNBase.__init__(self, param_dims, param_dims_sparse)
        
        if W is None:
            self.sparams.W = #randomly initialize W
        else: 
            self.sparams.W = W.copy()
        
        # add other stuff as we need it

    def _acc_grads(self, question, answer): 
        # question is a list of indices into W for each word in the question
        # answer is also in index into W (single entity)
        for qindx in question: 
            for aindx in answer:
                self.sgrads.W[aindx] = self.sparams.W[qindx]
    
    def compute_single_q_loss(self, question, answer):
        questionEmbed = np.sum(self.sparams.W[question], axis=1)
        answerEmbed = np.sum(self.sparams.W[answer], axis=1)
        return dot(questionEmbed, answerEmbed)

    def compute_loss(self, X, Y)
        if not isinstance(X[0], ndarray): # single example
            return self.compute_single_q_loss(X, Y)
        else: # multiple examples
            return sum([self.compute_single_q_loss(xs,ys) for xs,ys in itertools.izip(X, Y)])
