from utils import *

class RandomInit():
    def __init__(self, mean = 0, std = 1):
        self.mean = mean
        self.std = std

    def initialize(self, prev, curr):
        weights = np.random.normal(loc = self.mean, scale = self.std, size = (prev, curr))
        bias = np.random.normal(loc = self.mean, scale = self.std, size = (curr,))
        return weights, bias
    
class XavierInit():
    def __init__(self):
        pass

    def initialize(self, prev, curr):
        weights = np.random.normal(loc = 0, scale = np.sqrt(6.0 / (prev + curr)), size = (prev, curr))
        bias = np.zeros((curr,), dtype = np.float64)
        return weights, bias
    