from utils import *

class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x):
        return self.forward(x) * (1 - self.forward(x))
    
class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x):
        return np.where(x > 0, 1, 0)
    
class TanH:
    def __init__(self):
        pass

    def forward(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def backward(self, x):
        return 1 - self.forward(x)**2
    
class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
    
    def backward(self, x):
        return self.forward(x)