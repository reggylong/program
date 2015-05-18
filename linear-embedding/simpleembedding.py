import numpy as np
from nnbase import NNBase

class SimpleLinear(NNBase):
    """
    """
    def __init__(self, W_dim=None, W=None,
                 reg=0.001, alpha=0.01, margin=0.1):
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
        
        self.margin = margin
        self.reg = reg
        self.alpha = alpha
        
    def _acc_grads(self, question, answer, answer_neg): 
        # question is a list of indices into W for each word in the question
        # answer is also in index into W (single entity)
        # right now we assume that every single question is a tuple of two things delimited by "|||"
        leftq, rightq = question
        lefta, righta = answer
        lefta_neg, righta_neg = answer_neg
        if compute_single_q_loss(self, question, answer, answer_neg) <= 0: 
            return
        for qindx in leftq: 
            for aindx in lefta:
                self.sgrads.W[aindx] = -self.sparams.W[qindx]
            for aindx in lefta_neg: 
                self.sgrads.W[aindx] = self.sparams.W[qindx]
        for qindx in rightq:
            for aindx in righta:
                self.sgrads.W[aindx] = -self.sparams.W[qindx]
            for aindx in righta_neg:
                self.sgrads.W[aindx] = self.sparams.W[qindx]
        
    def compute_single_q_loss(self, question, answer, answer_neg):
        # see acc_grads
        leftq, rightq = question
        lefta, righta = answer
        lefta_neg, righta_neg = answer_neg
        margin_sum = 0
        questionEmbed = np.sum(self.sparams.W[leftq], axis=1)
        answerEmbed = np.sum(self.sparams.W[lefta], axis=1)
        answerEmbed_neg = np.sum(self.sparams.W[lefta_neg], axis=1)
        margin_sum -= dot(questionEmbed, answerEmbed) - dot(questionEmbed, answerEmbed_neg)
        questionEmbed = np.sum(self.sparams.W[rightq], axis=1)
        answerEmbed = np.sum(self.sparams.W[righta], axis=1)
        answerEmbed_neg = np.sum(self.sparams.W[righta_neg], axis=1)
        margin_sum -= dot(questionEmbed, answerEmbed) - dot(questionEmbed, answerEmbed_neg)
        margin_sum += self.margin
        return max(margin_sum, 0)

    def compute_loss(self, X, Y, Y_neg)
        if not isinstance(X[0], ndarray): # single example
            return self.compute_single_q_loss(X, Y, Y_neg)
        else: # multiple examples
            return sum([self.compute_single_q_loss(xs,ys,ys_neg) for xs,ys,ys_neg in itertools.izip(X, Y)])
