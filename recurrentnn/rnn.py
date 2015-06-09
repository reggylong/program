import collections 
from numpy import *
from nnbase import NNBase
import itertools
import random as rand
import sys

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

class RNN(NNBase):
    """
    Implements an RNN language model of the form:
    h(t) = sigmoid(H * h(t-1) + L[x(t)] + b1)
    y(t) = softmax(U * h(t))
    where y(t) predicts the next word in the sequence

    U = |V| * dim(h) as output vectors
    L = |V| * dim(h) as input vectors

    You should initialize each U[i,j] and L[i,j]
    as Gaussian noise with mean 0 and variance 0.1

    Arguments:
        L0 : initial input word vectors
        U0 : initial output word vectors
        alpha : default learning rate
        bptt : number of backprop timesteps
    """

    def random_weight_matrix(self, m, n):
        e = sqrt(6.0) / sqrt(m + n)
        A0 = random.uniform(-e, e, (m,n))
        return A0


    def __init__(self, L0, middledim=30, reg=1e-5,
                 margin=1, backpropwv=False, alpha=0.005, rseed=10, bptt=1):

        self.hdim = L0.shape[1] # word vector dimensions
        self.vdim = L0.shape[0] # vocab size
        param_dims = dict(H = (self.hdim, self.hdim), W = (middledim, self.hdim), b = (middledim))
        # note that only L gets sparse updates
        param_dims_sparse = dict(L = L0.shape)
        NNBase.__init__(self, param_dims, param_dims_sparse)

        #### YOUR CODE HERE ####


        # Initialize word vectors
        self.sparams.L = L0.copy()
        
        self.params.H = self.random_weight_matrix(*self.params.H.shape)
        
        self.params.W = self.random_weight_matrix(*self.params.W.shape)
        self.params.b = zeros(*self.params.b.shape)

        self.reg = reg
        self.bptt = bptt
        self.alpha = alpha
        self.backpropwv = backpropwv
        self.margin = margin
        #### END YOUR CODE ####

    def sigmoid(self, x):
        return 1.0 / (1.0 + exp(-x))

    def sig_grad(self, x):
        return x*(1 - x)

    def tanh(self, x):
        return 2.0 * self.sigmoid(2.0 * x) - 1

    def tanh_grad(self, f):
        return 1.0 - square(f) 

    def calc_hidden_vec(self, xs, hiddenvec, timenum): 
        for i in xrange(timenum):
            hiddenvec[i + 1] = self.sigmoid(dot(self.params.H, hiddenvec[i]) + self.sparams.L[xs[i]])
    
    def calc_backprop(self, xs, hs, delta):
        siggrads = self.sig_grad(hs)
        i = len(xs) - 1
        dh_curr = delta
        for j in xrange(self.bptt):
            if i - j < 0:
                break   
            dsig = siggrads[i - j + 1]*dh_curr
            self.grads.H += outer(dsig, hs[i - j])
            self.sgrads.L[xs[i - j]] = dsig
            dh_curr = dot(self.params.H.T, dsig)

    def sample_neg(self, answers, question):
        if len(answers[0]) > 1:
            answer = answers[1]
            neg_answer_ind = random.randint(0, len(answers[0]) - 1)
            count = 0
            
            while checkParseEqual(answers[0][neg_answer_ind], answer):
                neg_answer_ind = random.randint(0, len(answers[0]) - 1)
                count += 1
                if count > 3:
                    return [[random.randint(0, self._param_dims_sparse['L'][0] - 1)] for q in question] 
            result = list(answers[0][neg_answer_ind])
            while len(result) < len(question):
                result.append([random.randint(0, self._param_dims_sparse['L'][0] - 1)])
            return result
        else: 
            # if there's only one correct parse create janky neg....
            return [[random.randint(0, self._param_dims_sparse['L'][0] - 1)] for q in question]   
    
    def _acc_grads(self, answers, question):
        all_parses, oracle = answers
        question = question[::-1]
        question = [arr[::-1] for arr in question]
        oracle = [arr[::-1] for arr in oracle]
	
        q_lens = [len(arr) for arr in question]
        a_lens = [len(arr) for arr in oracle]
        hs_question = []
        hs_answer = []

        for ind, qpart in enumerate(question):
            hs_question.append(zeros((q_lens[ind] + 1, self.hdim)))
            self.calc_hidden_vec(qpart, hs_question[ind], q_lens[ind])
	
        for ind, apart in enumerate(oracle):
            hs_answer.append(zeros((a_lens[ind] + 1, self.hdim)))
            self.calc_hidden_vec(apart, hs_answer[ind], a_lens[ind])

	    answer_neg = self.sample_neg(answers, question)
	    answer_neg = [arr[::-1] for arr in answer_neg]
	    aneg_lens = [len(arr) for arr in answer_neg]
	    hs_neg = []
	
	    for ind, apart in enumerate(answer_neg):
	        hs_neg.append(zeros((aneg_lens[ind] + 1, self.hdim)))
	        self.calc_hidden_vec(apart, hs_neg[ind], aneg_lens[ind])

        neg_combine = concatenate([seq[aneg_lens[ind]] for ind, seq in enumerate(hs_neg)])
        q_combine = concatenate([seq[q_lens[ind]] for ind, seq in enumerate(hs_question)])
        a_combine = concatenate([seq[a_lens[ind]] for ind, seq in enumerate(hs_answer)])
        W_tile = tile(self.params.W, len(question))
        hvec_neg = self.tanh(dot(W_tile, neg_combine) + self.params.b)
        hvec_q = self.tanh(dot(W_tile, q_combine) + self.params.b)
        hvec_a = self.tanh(dot(W_tile, a_combine) + self.params.b)
        diff = hvec_q - hvec_a
        diffneg = hvec_q - hvec_neg
        margin = max(0, self.margin - sum(diffneg**2) + sum(diff**2))
        if not margin > 0: return
        delta_qneg = -self.tanh_grad(hvec_q)*diffneg
        delta_neg = self.tanh_grad(hvec_neg)*diffneg
        q_sum = sum(reshape(q_combine, (len(question), self.hdim)), axis=0)
        neg_sum = sum(reshape(neg_combine, (len(question), self.hdim)), axis=0)
        a_sum = sum(reshape(a_combine, (len(question), self.hdim)), axis=0)
        self.grads.W += outer(delta_qneg, q_sum) + outer(delta_neg, neg_sum)
        self.grads.b += delta_qneg + delta_neg

        delta_q = self.tanh_grad(hvec_q)*diff
        delta_a = self.tanh_grad(hvec_a)*(-diff)
        self.grads.W += outer(delta_q, q_sum) + outer(delta_a, a_sum)
        self.grads.b += delta_q + delta_a
	
        self.grads.W += self.reg*self.params.W
    
        if not self.backpropwv: return
        d_qcombine = dot(W_tile.T, delta_q + delta_qneg)
        d_acombine = dot(W_tile.T,delta_a)
        d_negcombine = dot(W_tile.T, delta_neg)    
    
        for ind, strList in enumerate(question):
            self.calc_backprop(strList, hs_question[ind], d_qcombine[self.hdim*ind:self.hdim*(ind + 1)])
        for ind, strList in enumerate(oracle):
            self.calc_backprop(strList, hs_answer[ind], d_acombine[self.hdim*ind:self.hdim*(ind + 1)])
        for ind, strList in enumerate(answer_neg):
            self.calc_backprop(strList, hs_neg[ind], d_negcombine[self.hdim*ind:self.hdim*(ind + 1)])
 
    def grad_check(self, x, y, outfd=sys.stderr, **kwargs):
        """
        Wrapper for gradient check on RNNs;
        ensures that backprop-through-time is run to completion,
        computing the full gradient for the loss as summed over
        the input sequence and predictions.

        Do not modify this function!
        """
        bptt_old = self.bptt
        self.bptt = 100
        print >> outfd, "NOTE: temporarily setting self.bptt = len(y) = %d to compute true gradient." % self.bptt
        NNBase.grad_check(self, x, y, outfd=outfd, **kwargs)
        self.bptt = bptt_old
        print >> outfd, "Reset self.bptt = %d" % self.bptt


    def predict_single(self, answers, question):
        all_parses, oracle = answers
        question = question[::-1]
        question = [arr[::-1] for arr in question]

        q_lens = [len(arr) for arr in question]

        hs_question = []

        for ind, qpart in enumerate(question):
            hs_question.append(zeros((q_lens[ind] + 1, self.hdim)))
            self.calc_hidden_vec(qpart, hs_question[ind], q_lens[ind])

	    q_combine = concatenate([seq[q_lens[ind]] for ind, seq in enumerate(hs_question)])
        W_tile = tile(self.params.W, len(question))
        hvec_q = self.tanh(dot(W_tile, q_combine) + self.params.b)

        minCost = inf
        minCostIndex = -1   
        
        for i, candidate in enumerate(all_parses):
            candidate = [arr[::-1] for arr in candidate]
            a_lens = [len(arr) for arr in candidate]
            hs_answer = []   
            for ind, apart in enumerate(candidate):
		        hs_answer.append(zeros((a_lens[ind] + 1, self.hdim)))
		        self.calc_hidden_vec(apart, hs_answer[ind], a_lens[ind])     
            a_combine = concatenate([seq[a_lens[ind]] for ind, seq in enumerate(hs_answer)])
            hvec_a = self.tanh(dot(W_tile, a_combine) + self.params.b)

            cost = sum((hvec_q - hvec_a)**2)
            if cost < minCost:
                minCostIndex = i
                minCost = cost
        return minCostIndex

    def predict(self, parses, utterances):
        outputs = []
        count = 0
        for parseSet, utterance in itertools.izip(parses, utterances):
            outputs.append(self.predict_single(parseSet, utterance))
            count += 1
            print("finished predicting", count)
        return outputs
        
    def compute_single_loss(self, answers, question):
        self.count += 1
	    #print(self.count)
        all_parses, oracle = answers
        question = question[::-1]
        question = [arr[::-1] for arr in question]
        #print("q", question)
        oracle = [arr[::-1] for arr in oracle]
        
        q_lens = [len(arr) for arr in question]
        a_lens = [len(arr) for arr in oracle]
    
        hs_question = []
        hs_answer = []
        
        for ind, qpart in enumerate(question):
            hs_question.append(zeros((q_lens[ind] + 1, self.hdim)))
            self.calc_hidden_vec(qpart, hs_question[ind], q_lens[ind])
        
        for ind, apart in enumerate(oracle):
            hs_answer.append(zeros((a_lens[ind] + 1, self.hdim)))
            self.calc_hidden_vec(apart, hs_answer[ind], a_lens[ind])

        answer_neg = self.sample_neg(answers, question)
        answer_neg = [arr[::-1] for arr in answer_neg]
        aneg_lens = [len(arr) for arr in answer_neg]
        hs_neg = []
        
        for ind, apart in enumerate(answer_neg):
            hs_neg.append(zeros((aneg_lens[ind] + 1, self.hdim)))
            self.calc_hidden_vec(apart, hs_neg[ind], aneg_lens[ind])

        neg_combine = concatenate([seq[aneg_lens[ind]] for ind, seq in enumerate(hs_neg)])
        q_combine = concatenate([seq[q_lens[ind]] for ind, seq in enumerate(hs_question)])
        a_combine = concatenate([seq[a_lens[ind]] for ind, seq in enumerate(hs_answer)])
        W_tile = tile(self.params.W, len(question))
        
        hvec_neg = self.tanh(dot(W_tile, neg_combine) + self.params.b)
        hvec_q = self.tanh(dot(W_tile, q_combine) + self.params.b)
        #print("loss", hvec_q)
	    #print(oracle)
	    #print("q", question)
        hvec_a = self.tanh(dot(W_tile, a_combine) + self.params.b)
 
        diff = hvec_q - hvec_a
        diffneg = hvec_q - hvec_neg
        margin = max(0, self.margin - sum(diffneg**2) + sum(diff**2))
        J = 0.5*margin + 0.5*self.reg*sum(self.params.W**2)
        return J

    def compute_loss(self, X, Y):
        """
        Compute total loss over a dataset.
        (wrapper for compute_seq_loss)

        Do not modify this function!
        """
        self.count = 0
        if not isinstance(Y[0][0], collections.Iterable): # single example
            return self.compute_single_loss(X, Y)
        else: # multiple examples
            return sum([self.compute_single_loss(answers, question)
                       for answers, question in itertools.izip(X, Y)])

    def compute_mean_loss(self, X, Y):
        """
        Normalize loss by total number of points.

        Do not modify this function!
        """
        J = self.compute_loss(X, Y)
        ntot = sum(map(len,Y))
        return J / float(ntot)

if __name__ == "__main__":
    rnn = RNN(sqrt(0.1)*random.standard_normal((1000, 5)), backpropwv = True, margin=20)
    utterExample = [[411, 339, 46], [341, 591, 83, 355, 175], [2, 3, 4]]
    trainExample = ([[[411, 339, 46], [341, 591, 83, 355, 175], [1, 2, 3]]], [[21, 1], [2, 3, 4], [5, 6, 7]])
   
    rnn.grad_check(trainExample, utterExample)
