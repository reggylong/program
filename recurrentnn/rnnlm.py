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

class RNNLM(NNBase):
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

    def __init__(self, L0, U0=None,
                 alpha=0.005, rseed=10, bptt=1):

        self.hdim = L0.shape[1] # word vector dimensions
        self.vdim = L0.shape[0] # vocab size
        param_dims = dict(H = (self.hdim, self.hdim), U = L0.shape)
        # note that only L gets sparse updates
        param_dims_sparse = dict(L = L0.shape)
        NNBase.__init__(self, param_dims, param_dims_sparse)

        #### YOUR CODE HERE ####


        # Initialize word vectors
        self.sparams.L = L0.copy()
        
        if U0 is None:
            self.params.U = sqrt(0.1)*random.standard_normal(L0.shape)
        else:
            self.params.U = U0

        self.params.H = self.random_weight_matrix(*self.params.H.shape)
        
        self.bptt = bptt
        self.alpha = alpha
        #### END YOUR CODE ####

    def sigmoid(self, x):
        return 1.0 / (1.0 + exp(-x))

    def sig_grad(self, x):
        return x*(1 - x)

    def tanh(self, x):
        return 2.0 * self.sigmoid(2.0 * x) - 1

    def tanh_grad(self, f):
        return 1.0 - square(f) 
    
    def softmax(self, x):
        xt = exp(x - max(x))
        return xt / sum(xt)
    
    def calc_hidden_vec(self, xs, hiddenvec, pvec, timenum): 
        J = 0
        for i in xrange(timenum):
            hiddenvec[i + 1] = self.sigmoid(dot(self.params.H, hiddenvec[i]) + self.sparams.L[xs[i]])
            if i < timenum - 1:
                pvec[i] = self.softmax(dot(self.params.U, hiddenvec[i + 1]))   
                J += -log(pvec[i, xs[i + 1]])
        return J
    
    def calc_backprop(self, xs, hs, delta):
        siggrads = self.sig_grad(hs)
        ns = len(xs)
        for i in xrange(ns):
            dh_curr = dot(self.params.U.T, delta[i])
            for j in xrange(self.bptt):
                if i - j < 0:
                    break   
                dsig = siggrads[i - j + 1]*dh_curr
                self.grads.H += outer(dsig, hs[i - j])
                self.sgrads.L[xs[i - j]] = dsig
                dh_curr = dot(self.params.H.T, dsig)

    def _acc_grads(self, answers, question):
        """
        Question is (input, command) where both are lists of indices into the word dict.
        Answers is ([all parses], oracle parse) where both are list of 
        """
        input_q, command_q = question
        all_parses, oracle = answers
        input_a, command_a = oracle

        n_inputq = len(input_q)
        n_commandq = len(command_q)
        n_inputa = len(input_a)
        n_commanda = len(command_a)

        # make matrix here of corresponding h(t)
        # hs[-1] = initial hidden state (zeros)
        # change this computation if we don't want to use the divider
        hs_inputq = zeros((n_inputq + 1, self.hdim))
        hs_commandq = zeros((n_commandq + 1, self.hdim))
        hs_inputa = zeros((n_inputa + 1, self.hdim))
        hs_commanda = zeros((n_commanda + 1, self.hdim))
        ps_inputq = zeros((n_inputq, self.vdim))
        ps_commandq = zeros((n_commandq, self.vdim)) 
        ps_inputa = zeros((n_inputa, self.vdim))
        ps_commanda = zeros((n_commanda, self.vdim))
        
        # Forward propagation
        self.calc_hidden_vec(input_q, hs_inputq, ps_inputq, n_inputq)
        self.calc_hidden_vec(command_q, hs_commandq, ps_commandq, n_commandq)
        self.calc_hidden_vec(input_a, hs_inputa, ps_inputa, n_inputa)
        self.calc_hidden_vec(command_a, hs_commanda, ps_commanda, n_commanda)
        
        # Backward propagation through time
        y_inputq = zeros((n_inputq, self.vdim))
        y_inputq[arange(n_inputq - 1), input_q[1:]] = 1
        y_commandq = zeros((n_commandq, self.vdim))
        y_commandq[arange(n_commandq - 1), command_q[1:]] = 1
        y_inputa = zeros((n_inputa, self.vdim))
        y_inputa[arange(n_inputa - 1), input_a[1:]] = 1
        y_commanda = zeros((n_commanda, self.vdim))
        y_commanda[arange(n_commanda - 1), command_a[1:]] = 1

        delta_inputq = ps_inputq - y_inputq
        delta_commandq = ps_commandq - y_commandq
        delta_inputa = ps_inputa - y_inputa
        delta_commanda = ps_commanda - y_commanda

        self.grads.U += dot(delta_inputq.T, hs_inputq[1:])
        self.grads.U += dot(delta_commandq.T, hs_commandq[1:])
        self.grads.U += dot(delta_inputa.T, hs_inputa[1:])
        self.grads.U += dot(delta_commanda.T, hs_commanda[1:])

        self.calc_backprop(input_q, hs_inputq, delta_inputq)
        self.calc_backprop(input_a, hs_inputa, delta_inputa)
        self.calc_backprop(command_q, hs_commandq, delta_commandq)
        self.calc_backprop(command_a, hs_commanda, delta_commanda)

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

    def compute_single_loss(self, answers, question):
        input_q, command_q = question
        all_parses, oracle = answers
        input_a, command_a = oracle

        n_inputq = len(input_q)
        n_commandq = len(command_q)
        n_inputa = len(input_a)
        n_commanda = len(command_a)

        # make matrix here of corresponding h(t)
        # hs[-1] = initial hidden state (zeros)
        # change this computation if we don't want to use the divider
        hs_inputq = zeros((n_inputq + 1, self.hdim))
        hs_commandq = zeros((n_commandq + 1, self.hdim))
        hs_inputa = zeros((n_inputa + 1, self.hdim))
        hs_commanda = zeros((n_commanda + 1, self.hdim))
        ps_inputq = zeros((n_inputq, self.vdim))
        ps_commandq = zeros((n_commandq, self.vdim)) 
        ps_inputa = zeros((n_inputa, self.vdim))
        ps_commanda = zeros((n_commanda + 1, self.vdim))
        
        # Forward propagation
        J = 0

        J += self.calc_hidden_vec(input_q, hs_inputq, ps_inputq, n_inputq)
        J += self.calc_hidden_vec(command_q, hs_commandq, ps_commandq, n_commandq)
        J += self.calc_hidden_vec(input_a, hs_inputa, ps_inputa, n_inputa)
        J += self.calc_hidden_vec(command_a, hs_commanda, ps_commanda, n_commanda)
    
        return J
    
    def compute_loss(self, X, Y):
        """
        Compute total loss over a dataset.
        (wrapper for compute_seq_loss)

        Do not modify this function!
        """
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
    rnn = RNNLM(sqrt(0.1)*random.standard_normal((1000, 5)))
    utterExample = [[411, 339, 46], [341, 591, 83, 355, 175]]
    trainExample = ([([411, 339, 46], [341, 591, 83, 355, 175])], ([21, 1], [2, 3, 4]))
    rnn.grad_check(trainExample, utterExample)
