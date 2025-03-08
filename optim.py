from utils import *

#iter = update for each iteration
class SGD():
    def __init__(self, lr=0.01):
        self.iter = 0
        self.lr = lr

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])

    def get_iter(self, grad):
        self.iter = self.lr*grad
        return self.iter

class GDMomentum():
    def __init__(self, lr=1e-3, beta=0.9):
        self.iter = 0
        self.lr = lr
        self.beta = beta

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])

    def get_iter(self, grad):
        self.iter = self.lr*grad + self.beta*self.iter
        return self.iter

class GDNesterov():
    def __init__(self, lr=1e-3, beta=0.9):
        self.iter = 0
        self.lr = lr
        self.beta = beta

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])
        
    def get_iter(self, W, grad=None):
        # Have to still work on this
        W_lookahead = W - self.beta*self.iter
        self.iter = self.beta*self.iter + self.lr*gradient(W_lookahead) # Need to call gradient function
        W = W - self.iter
        return W
        

class RMSProp():
    def __init__(self, lr = 1e-3, epsilon = 1e-7, beta=0.9):
        self.v = 0
        self.lr = lr
        self.epsilon = epsilon
        self.beta = beta

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])

    def get_iter(self, grad):
        self.v = self.beta*self.v + (1-self.beta)*(grad**2)
        return (self.lr/(self.v+self.epsilon)**0.5)*grad

class Adam():
    def __init__(self, lr=1e-2, epsilon=1e-8, beta1=0.9, beta2=0.999):
        self.m = 0
        self.v = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.epsilon = epsilon
        self.iter = 1

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])

    def get_iter(self, grad):
        self.m = self.beta1*self.m + (1-self.beta1)*grad
        self.v = self.beta2*self.v + (1-self.beta2)*(grad**2)
        m_cap = self.m/(1-self.beta1**self.iter)
        v_cap = self.v/(1-self.beta2**self.iter)        
        self.iter += 1
        return (self.lr/(v_cap+self.epsilon)**0.5)*m_cap

class Nadam():
    def __init__(self, lr=1e-3, epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.m = 0
        self.v = 0
        self.lr = lr
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 1

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])
    
    def get_iter(self, grad):
        self.m = self.beta1*self.m + (1-self.beta1)*grad
        self.v = self.beta2*self.v + (1-self.beta2)*(grad**2)
        mhat = self.m/(1-self.beta1**self.iter)
        vhat = self.v/(1-self.beta2**self.iter) 
        iter = self.beta1*mhat + ((1-self.beta1)/(1-self.beta1**self.iter))*grad
        self.iter += 1
        return (self.lr/(vhat+self.epsilon)**0.5)*iter