import collections
from numpy import *
from nnbase import NNBase
import random
import itertools


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
            self.sparams.W = sqrt(0.1)*random.standard_normal(W_shape)#randomly initialize W
        else: 
            self.sparams.W = W.copy()
        
        self.margin = margin
        self.reg = reg
        self.alpha = alpha
           
    def _acc_grads(self, answers, question): 
        # question is a list of indices into W for each word in the question
        # answers is a tuple of (all parses, correct parse) which we sample from
        # right now we assume that every single question is a tuple of two things delimited by "|||"
        answer = answers[1]
        answer_neg = self.sample_neg(answers, question)
        leftq, rightq = question
        lefta, righta = answer
        lefta_neg, righta_neg = answer_neg
        if self.compute_loss_fixed_neg(question, answer, answer_neg) <= 0: 
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
    
    def sample_neg(self, answers, question):
        if len(answers[0]) > 1:
            answer = answers[1]
            neg_answer_ind = random.randint(0, len(answers[0]) - 1)
            while checkParseEqual(answers[0][neg_answer_ind], answer):
                neg_answer_ind = random.randint(0, len(answers[0]) - 1)
            return answers[0][neg_answer_ind]
        else: 
            # if there's only one correct parse create janky neg....
            return [[random.randint(0, self._param_dims_sparse['W'][0] - 1)], [random.randint(0, self._param_dims_sparse['W'][0] - 1)]]

    def compute_single_q_loss(self, answers, question):
        answer = answers[1]
        answer_neg = self.sample_neg(answers, question)
        return self.compute_loss_fixed_neg(question, answer, answer_neg)
    
    def compute_loss_fixed_neg(self, question, answer, answer_neg):
        # see acc_grads
        leftq, rightq = question
        lefta, righta = answer
        lefta_neg, righta_neg = answer_neg
        margin_sum = 0
        questionEmbed = sum(self.sparams.W[leftq], axis=0)
        answerEmbed = sum(self.sparams.W[lefta], axis=0)
        answerEmbed_neg = sum(self.sparams.W[lefta_neg], axis=0)
        margin_sum -= dot(questionEmbed.T, answerEmbed) - dot(questionEmbed.T, answerEmbed_neg)
        questionEmbed = sum(self.sparams.W[rightq], axis=0)
        answerEmbed = sum(self.sparams.W[righta], axis=0)
        answerEmbed_neg = sum(self.sparams.W[righta_neg], axis=0)
        margin_sum -= dot(questionEmbed.T, answerEmbed) - dot(questionEmbed.T, answerEmbed_neg)
        margin_sum += self.margin
        return max(margin_sum, 0)

    def compute_loss(self, X, y):
        if not isinstance(X[0][0], collections.Iterable): # single example
            return self.compute_single_q_loss(X, y)
        else: # multiple examples
            return sum([self.compute_single_q_loss(answers, question) for answers, question in itertools.izip(X, y)])
