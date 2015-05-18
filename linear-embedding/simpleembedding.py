from numpy import *
from nnbase import NNBase
import random

def checkListsEqual(list1, list2):
  if len(list1) != len(list2):
    return False
  for l1, l2 in itertools.zip(list1, list2):
    if l1 != l2:
      return False
  return True

def checkParseEqual(parse1, parse2):
  if len(parse1) != len(parse2):
    return False
  for list1, list2 in itertools.zip(parse1, parse2):
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
            self.sparams.W = #randomly initialize W
        else: 
            self.sparams.W = W.copy()
        
        self.margin = margin
        self.reg = reg
        self.alpha = alpha
    
    def train_point_sgd(self, x, y, alpha):
        """Generic single-point SGD"""
        self._reset_grad_acc()
        self._acc_grads(y[1], y[0], x)
        self._apply_grad_acc(alpha)

    def train_sgd(self, X, y,
                  idxiter=None, alphaiter=None,
                  printevery=10000, costevery=10000,
                  devidx=None):
        # y[0] is correct parse
        # y[1] is original utterance
        if idxiter == None: # default training schedule
            idxiter = xrange(len(y1))
        if alphaiter == None: # default training schedule
            alphaiter = itertools.\repeat(self.alpha)

        costs = []
        counter = 0
        t0 = time.time()

        try:
            print "Begin SGD..."
            for idx, alpha in itertools.izip(idxiter, alphaiter):
                correct_parse = y[idx][0]
                total_samples = X[idx]
                neg_parse_ind = random.randint(0, len(totalSamples) - 1)
                while checkParseEqual(total_samples[neg_parse_ind], correct_parse):
                    neg_parse_ind = random.randint(0, len(total_samples) - 1)
                neg_parse = total_samples[neg_parse_ind]
                if counter % printevery == 0:
                    print "  Seen %d in %.02f s" % (counter, time.time() - t0)
                if counter % costevery == 0:
                    if devidx != None:
                        cost = self.compute_display_loss(X[devidx], y[devidx])
                    else: cost = self.compute_display_loss(X, y)
                    costs.append((counter, cost))
                    print "  [%d]: mean loss %g" % (counter, cost)
                # minibatch not implemented yet
                #if hasattr(idx, "__iter__") and len(idx) > 1: # if iterable
                #    self.train_minibatch_sgd(X[idx], y[idx], alpha)
                #elif hasattr(idx, "__iter__") and len(idx) == 1: # single point
                #    idx = idx[0]
                #    self.train_point_sgd(X[idx], y[idx], alpha)
                #else:
                self.train_point_sgd(neg_parse, y, alpha)
                counter += 1
        except KeyboardInterrupt as ke:
            """
            Allow manual early termination.
            """
            print "SGD Interrupted: saw %d examples in %.02f seconds." % (counter, time.time() - t0)
            return costs

        # Wrap-up
        if devidx != None:
            cost = self.compute_display_loss(X[devidx], y[devidx])
        else: cost = self.compute_display_loss(X, y)
        costs.append((counter, cost))
        print "  [%d]: mean loss %g" % (counter, cost)
        print "SGD complete: %d examples in %.02f seconds." % (counter, time.time() - t0)

        return costs
    
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
        questionEmbed = sum(self.sparams.W[leftq], axis=1)
        answerEmbed = sum(self.sparams.W[lefta], axis=1)
        answerEmbed_neg = sum(self.sparams.W[lefta_neg], axis=1)
        margin_sum -= dot(questionEmbed, answerEmbed) - dot(questionEmbed, answerEmbed_neg)
        questionEmbed = sum(self.sparams.W[rightq], axis=1)
        answerEmbed = sum(self.sparams.W[righta], axis=1)
        answerEmbed_neg = sum(self.sparams.W[righta_neg], axis=1)
        margin_sum -= dot(questionEmbed, answerEmbed) - dot(questionEmbed, answerEmbed_neg)
        margin_sum += self.margin
        return max(margin_sum, 0)

    def compute_loss(self, X, y)
        if not isinstance(X[0], ndarray): # single example
            return self.compute_single_q_loss(y[1], y[0], X)
        else: # multiple examples
            return sum([self.compute_single_q_loss(q, a_right, a_neg) for a_neg, (a_right, q) in itertools.izip(X, Y)])
