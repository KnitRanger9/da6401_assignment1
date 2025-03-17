from utils import *
from activations import *
from optim import *
from copy import deepcopy

activations = {"sigmoid": Sigmoid(), "relu": ReLU(), "tanh": TanH(), "softmax": Softmax()}

class A_x():
    def __init__(self, X):
        self.X = X
        self.n = X.shape[0]
        self.size=0


class Dense():
    def __init__(self, name, size, activation, last=False):
        self.name = name
        self.size = size
        self.activation = activations[activation]
        self.last = last
        self.W_optimizer = deepcopy(SGD())
        self.b_optimizer = deepcopy(SGD())