import collections
from numpy import *
from nnbase import NNBase
import random
import itertools
import numpy.random as random

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
                 reg=0.001, alpha=0.01, margin=10):
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
        leftq, midq, rightq = question
        lefta, mida, righta = answer
        lefta_neg, mida_neg, righta_neg = answer_neg
        questioncomb = []
        answercomb = []
        answer_negcomb = []
        for q, a, aneg in itertools.izip(question, answer, answer_neg):
            questioncomb += q
            answercomb += a
            answer_negcomb += aneg
        if self.compute_loss_fixed_neg(question, answer, answer_neg) <= 0: 
            return
        #for qindx in leftq: 
        #    for aindx in lefta:
        #        self.sgrads.W[aindx] = -self.sparams.W[qindx]
        #        self.sgrads.W[qindx] = -self.sparams.W[aindx]
        #    for aindx in lefta_neg: 
        #        self.sgrads.W[aindx] = self.sparams.W[qindx]
        #        self.sgrads.W[qindx] = self.sparams.W[aindx]
        #for qindx in rightq:
        #    for aindx in righta:
        #        self.sgrads.W[aindx] = -self.sparams.W[qindx]
        #        self.sgrads.W[qindx] = -self.sparams.W[aindx]
        #    for aindx in righta_neg:
        #        self.sgrads.W[aindx] = self.sparams.W[qindx]
        #        self.sgrads.W[qindx] = self.sparams.W[aindx]
        #for qindx in midq:
        #    for aindx in mida:
        #        self.sgrads.W[aindx] = -self.sparams.W[qindx]
        #        self.sgrads.W[qindx] = -self.sparams.W[aindx]
        #    for aindx in mida_neg:
        #        self.sgrads.W[aindx] = self.sparams.W[qindx] 
        #        self.sgrads.W[qindx] = self.sparams.W[aindx]       
        for qindx in questioncomb:
            for aindx in answercomb:
                 self.sgrads.W[aindx] = self.sparams.W[aindx] - self.sparams.W[qindx]
                 self.sgrads.W[qindx] = self.sparams.W[qindx] - self.sparams.W[aindx]
            for aindx in answer_negcomb:
                 self.sgrads.W[aindx] = -self.sparams.W[aindx]+ self.sparams.W[qindx]
                 self.sgrads.W[qindx] = -self.sparams.W[qindx] + self.sparams.W[aindx]
   
    def sample_neg(self, answers, question):
        if len(answers[0]) > 1:
            answer = answers[1]
            neg_answer_ind = random.randint(0, len(answers[0]) - 1)
            count = 0
            
            while checkParseEqual(answers[0][neg_answer_ind], answer):
                neg_answer_ind = random.randint(0, len(answers[0]) - 1)
                count += 1
                if count > 3:
                    return [[random.randint(0, self._param_dims_sparse['W'][0] - 1)] for q in question] 
            result = list(answers[0][neg_answer_ind])
            while len(result) < len(question):
                result.append([random.randint(0, self._param_dims_sparse['W'][0] - 1)])
            return result
        else: 
            # if there's only one correct parse create janky neg....
            return [[random.randint(0, self._param_dims_sparse['W'][0] - 1)] for q in question] 
    
    def compute_single_q_loss(self, answers, question):
        #print(answers)
        answer = answers[1]
        answer_neg = self.sample_neg(answers, question)
        return self.compute_loss_fixed_neg(question, answer, answer_neg)
    
    def compute_loss_fixed_neg(self, question, answer, answer_neg):
        # see acc_grads
        leftq, midq, rightq = question
        lefta, mida, righta = answer
        lefta_neg, mida_neg, righta_neg = answer_neg
        margin_sum = 0
        #questionEmbed = sum(self.sparams.W[leftq], axis=0)
        #answerEmbed = sum(self.sparams.W[lefta], axis=0)
        #answerEmbed_neg = sum(self.sparams.W[lefta_neg], axis=0)
        #margin_sum -= dot(questionEmbed.T, answerEmbed) - dot(questionEmbed.T, answerEmbed_neg)
        #questionEmbed = sum(self.sparams.W[rightq], axis=0)
        #answerEmbed = sum(self.sparams.W[righta], axis=0)
        #answerEmbed_neg = sum(self.sparams.W[righta_neg], axis=0)
        #margin_sum -= dot(questionEmbed.T, answerEmbed) - dot(questionEmbed.T, answerEmbed_neg)
        #questionEmbed = sum(self.sparams.W[midq], axis=0)
        #answerEmbed = sum(self.sparams.W[mida], axis=0)
        #answerEmbed_neg = sum(self.sparams.W[mida_neg], axis=0)
        questioncomb = []
        answercomb = []
        answer_negcomb = []
        for q, a, aneg in itertools.izip(question, answer, answer_neg):
            questioncomb += q
            answercomb += a
            answer_negcomb += aneg
        questionEmbed = sum(self.sparams.W[questioncomb], axis=0)
        answerEmbed = sum(self.sparams.W[answercomb], axis=0)
        answerEmbed_neg = sum(self.sparams.W[answer_negcomb], axis=0)
        margin_sum -= -sum((questionEmbed - answerEmbed)**2) + sum((questionEmbed - answerEmbed_neg)**2)
        margin_sum += self.margin
        return max(margin_sum, 0)

    def predict_single(self, answers, question):
        leftq, midq, rightq = question
        maxScore = -inf
        argmaxScore = -1
        #lqEmbed = sum(self.sparams.W[leftq], axis=0)
        #mqEmbed = sum(self.sparams.W[midq], axis=0)
        #rqEmbed = sum(self.sparams.W[rightq], axis=0)
        questioncomb = []
        for q in question:
            questioncomb += q
        questionEmbed = sum(self.sparams.W[questioncomb], axis=0)
        for ind, candidate in enumerate(answers[0]):
            lefta, mida, righta = candidate
            answercomb = []
            for a in candidate:
                answercomb += a
            #laEmbed = sum(self.sparams.W[lefta], axis=0)
            #maEmbed = sum(self.sparams.W[mida], axis=0)
            #raEmbed = sum(self.sparams.W[righta], axis=0)
            #score = dot(lqEmbed.T, laEmbed) + dot(rqEmbed.T, raEmbed) + dot(mqEmbed.T, maEmbed)
            answerEmbed = sum(self.sparams.W[answercomb], axis=0)
            score = -sum((questionEmbed.T - answerEmbed)**2)
            if score > maxScore: 
                maxScore = score
                argmaxScore = ind
        return argmaxScore

    def predict(self, parses, utterances):
        outputs = []
        for parseSet, utterance in itertools.izip(parses, utterances):
            outputs.append(self.predict_single(parseSet, utterance))
        return outputs
        
    def compute_loss(self, X, y):
        if not isinstance(y[0][0], collections.Iterable): # single example
            return self.compute_single_q_loss(X, y)
        else: # multiple examples
            return sum([self.compute_single_q_loss(answers, question) for answers, question in itertools.izip(X, y)])

if __name__ == "__main__":
    wv = sqrt(0.1)*random.standard_normal((1000, 10))
    rnn = SimpleLinear(W = wv)
    utterExample = [[411, 339, 46], [341, 591, 83, 355, 175], [2, 3, 4]]
    trainExample = ([[[411, 339, 46], [341, 591, 83, 355, 175], [1, 2, 3]]], [[21, 1], [2, 3, 4], [5, 6, 7]])
    print("Doing grad check")
    rnn.grad_check(trainExample, utterExample)
