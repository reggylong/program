import collections
from numpy import *
from nnbase import NNBase
import random
import itertools


class SiameseNet(NNBase):

    """
    # input + utterance
    r1 = sum L[x_i]
    h1 = tanh(W * r1 + b)

    # logical form (parse)
    r2 = sum L[x_j]
    h2 = tanh(W * r2 + b)

    # Cost Function (Frobenius norm)
    J = 1/2||h1-h2||^2 + reg/2||W||^2
    Arguments:
        L0: initial word vectors
        W: weight matrix  
        W_shape: weight matrix shape (if no W supplied)
    """
    def __init__(self, L0, W=None, W_shape=None,
                 reg=0.001, alpha=0.01, rseed=943):

        self.hdim = L0.shape[1] # word vector dimensions
        self.vdim = L0.shape[0] # vocab size
         
        if W is not None:
            W_shape = W.shape

        param_dims = dict(W = W_shape, b = W_shape[0])
        param_dims_sparse = dict(L = L0.shape)
        NNBase.__init__(self, param_dims, param_dims_sparse)
        
        self.params.W.shape = W_shape
        random.seed(rseed)
        self.sparams.L = L0.copy()
        
        # Randomly initialize W
        if W is None:
            self.params.W = random_weight_matrix(self.params.W.shape[0], self.params.W.shape[1])
        else:
            self.params.W = W.copy()

        self.sparams.b = zeros(W_shape[0]) 
        self.alpha = alpha
        self.reg = reg

    def random_weight_matrix(self, m, n):
        e = sqrt(6.0) / sqrt(m + n)
        A0 = random.uniform(-e, e, (m,n))
        return A0

    def sigmoid(self, x):
        return 1.0 / (1.0 + exp(-x))

    def tanh(self, x):
        return 2.0 * self.sigmoid(2.0 * x) - 1

    def tanh_grad(self, f):
        return 1.0 - square(f) 

    # question = (input, command)
    # answers = ([all parses], oracle parse)
    def _acc_grads(self, question, answers):
        # Forward propagation
        input_q, command_q = question
        all_parses, oracle = answers

        counter_q = collections.Counter()
        counter_a = collections.Counter()

        x1 = zeros(self.hdim) 
        x2 = zeros(self.hdim)
        for idx in itertools.chain(input_q, command_q):
            counter_q[idx] += 1.0
            x1 += self.sparams.L[idx]
        for idx in oracle:
            counter_a[idx] += 1.0
            x2 += self.sparams.L[idx]
        
        h1 = self.tanh(self.params.W.dot(x1) + self.params.b)
        h2 = self.tanh(self.params.W.dot(x2) + self.params.b)
        
        # Backward propagation
        z1 = (h1 - h2) * self.tanh_grad(h1)
        z2 = (h2 - h1) * self.tanh_grad(h2)
        self.grads.b += z1 + z2 
        self.grads.W += outer(z1, x1) + outer(z2, x2) + self.reg * self.params.W
        Lqgrad = self.params.W.T.dot(z1)
        Lagrad = self.params.W.T.dot(z2)
        for k,v in counter_q.iteritems():
            self.sgrads.L[k] = v * Lqgrad
        for k,v in counter_a.iteritems():
            self.sgrads.L[k] = v * Lagrad

    def compute_single_loss(self, question, answers):
        input_q, command_q = question
        all_parses, oracle = answers

        x1 = zeros(self.hdim) 
        x2 = zeros(self.hdim)
        for idx in itertools.chain(input_q, command_q):
            x1 += self.sparams.L[idx]
        for idx in oracle:
            x2 += self.sparams.L[idx]
        
        h1 = self.tanh(self.params.W.dot(x1) + self.params.b)
        h2 = self.tanh(self.params.W.dot(x2) + self.params.b)
        J = 1.0/2.0 * sum((h1 - h2)**2) + self.reg/2.0 * sum(self.params.W ** 2)
        return J

    # Loss over a dataset
    def compute_loss(self, X, Y):
        if not isinstance(X[0][0], collections.Iterable):
            return self.compute_single_loss(X, Y)
        else:
            return sum([self.compute_single_loss(question, answers) \
                for question, answers in itertools.izip(X, Y)])

