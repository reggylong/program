import collections
from numpy import *
from nnbase import NNBase
import itertools
import random as rand

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
            reg=1e-5, alpha=0.01, rseed=943):

        self.hdim = L0.shape[1] # word vector dimensions
        self.vdim = L0.shape[0] # vocab size

        if W is not None:
            W_shape = W.shape
        # next hidden layer
        param_dims = dict(W=W_shape, b=W_shape[0])
        param_dims_sparse = dict(L=L0.shape)
        NNBase.__init__(self, param_dims, param_dims_sparse)

        self.params.W.shape = W_shape
        self.margin = 15
        random.seed(rseed)
        self.sparams.L = L0.copy()

        # Randomly initialize W
        if W is None:
            self.params.W = self.random_weight_matrix(self.params.W.shape[0], self.params.W.shape[1])
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
        all_parses, oracle = answers
        counter_q = collections.Counter()
        counter_a = collections.Counter()

        x1 = zeros(self.hdim) 
        x2 = zeros(self.hdim)
        x1 = zeros(self.hdim) 
        x2 = zeros(self.hdim)
        for lists in question:
            for idx in lists:
                x1 += self.sparams.L[idx]
                counter_q[idx] += 1.0
        for lists in oracle:
            for idx in lists:
                x2 += self.sparams.L[idx]
                counter_a[idx] += 1.0

        h1 = self.tanh(self.params.W.dot(x1) + self.params.b)
        h2 = self.tanh(self.params.W.dot(x2) + self.params.b)
        dist = sum((h1 - h2)**2) 
        contrast = rand.choice(all_parses)
        while checkListsEqual(contrast,oracle):
            contrast = rand.choice(all_parses)
        
        counter_c = collections.Counter()
        x3 = zeros(self.hdim)
        for lists in contrast:
            for idx in lists:
                x3 += self.sparams.L[idx]
                counter_c[idx] += 1.0

        # Contrastive Loss
        h3 = self.tanh(self.params.W.dot(x3) + self.params.b)
        contrast_dist = sum((h1 - h3)**2)

        margin = max(0, self.margin - contrast_dist) 
        # Backward propagation
        z1 = (h1 - h2) * self.tanh_grad(h1)
        z2 = (h2 - h1) * self.tanh_grad(h2)
        z3 = (h3 - h1) * self.tanh_grad(h3)

        self.grads.b += (z1 + z2) 
        self.grads.W +=  (outer(z1, x1) + outer(z2, x2))    
        if margin > 0:
            self.grads.b += -1 * margin * (z1 + z3) 
            self.grads.W += -1 * margin * (outer(z1, x1) + outer(z3,x3))
            Lcgrad = self.params.W.T.dot(z3)
            for k,v in counter_q.iteritems():
                continue
                self.sgrads.L[k] = -1 * margin * v * Lcgrad
        self.grads.W += self.reg * self.params.W

        Lqgrad = self.params.W.T.dot(z1)
        Lagrad = self.params.W.T.dot(z2)
        for k,v in counter_q.iteritems():
            continue
            self.sgrads.L[k] = v * Lqgrad
        for k,v in counter_a.iteritems():
            continue
            self.sgrads.L[k] = v * Lagrad

    def compute_single_loss(self, question, answers):
        all_parses, oracle = answers

        x1 = zeros(self.hdim) 
        x2 = zeros(self.hdim)
        for lists in question:
            for idx in lists:
                x1 += self.sparams.L[idx]
        for lists in oracle:
            for idx in lists:
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

    def predict_single(self, question, answers):
        all_parses, oracle = answers
        x1 = zeros(self.hdim)
        for lists in question:
            for idx in lists:
                x1 += self.sparams.L[idx]
        h1 = self.tanh(self.params.W.dot(x1) + self.params.b)

        minCost = inf
        minCostIndex = -1
        for i, candidate in enumerate(all_parses):
            x2 = zeros(self.hdim)
            for lists in candidate:
                for idx in lists:
                    x2 += self.sparams.L[idx]
            h2 = self.tanh(self.params.W.dot(x2) + self.params.b)
            cost = sum((h1 - h2)**2) 
            if cost < minCost:
                minCostIndex = i
                minCost = cost
        return minCostIndex

    def predict(self, utterances, parses):
        outputs = []
        for utterance, parseSet in itertools.izip(utterances, parses):
            outputs.append(self.predict_single(utterance, parseSet))
        return outputs

