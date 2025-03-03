#Making NN from scratch: Day 1
# !pip install wandb

import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
import wandb
from sklearn.model_selection import train_test_split
from tqdm import tqdm

run = wandb.init(project="fashion_mnist")

#Question1
(X_train, X_test), (y_train, y_test) = fashion_mnist.load_data()
# Normalise
X_train = X_train/255
X_test = X_test/255

class Neuron:
  def __init__(self):
    self.data=0.0

class Layer(Neuron):
  def __init__(self, n):
    self.n_in = n
    self.data = np.zeros(n)
    self.neurons = [Neuron() for _ in range(n)]
    for i in range(n):
      self.data[i] = self.neurons[i].data

class NeuralNetwork:
  def __init__(self, n_inputs, n_hidden, n_outputs):
    self.input_layer = Layer(n_inputs)
    for i in range(n_inputs):
      self.input_layer.data[i] = self.input_layer.neurons[i].data
    self.hidden_layer = []
    for i in range(n_hidden):
      n = int(input(f"Enter number of neurons in hidden layer {i}: "))
      self.hidden_layer.append(Layer(n_hidden))
    self.hidden_layer = Layer(n_hidden)
    self.output_layer = Layer(n_outputs)
    self.weights_ih = np.random.rand(n_hidden, n_inputs)
    self.weights_ho = np.random.rand(n_outputs, n_hidden)
    self.bias_h = np.random.rand(n_hidden)
    self.bias_o = np.random.rand(n_outputs)
    self.lr = 0.1

    def forward(self, inputs):
      self.input_layer.data = inputs
      self.hidden_layer.data = np.dot(self.weights_ih, inputs) + self.bias_h
      self.hidden_layer.data = self.sigmoid(self.hidden_layer.data)
      self.output_layer.data = np.dot(self.weights_ho, self.hidden_layer.data) + self.bias_o
      self.output_layer.data = self.sigmoid(self.output_layer.data)
      return self.output_layer.data
