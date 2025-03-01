#Making NN from scratch: Day 1
import numpy as np
import pandas as pd

class Neuron:
  def __init__(self, weight, bias, lr):
    self.w=weight
    self.b=bias
    self.eta=lr

class Layer(Neuron):
  def __init__(self, n, n_out):
    self.n_in = n_in
    self.n_out = n_out
    self.w_matrix = np.zeros(n, n_out)
    self.b_matrix = np.zeros(n, n_out)

  def forward(self, X):
    out = np.dot(w_matrix.T, X)
    return out
