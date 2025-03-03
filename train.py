#Making NN from scratch: Day 1
# !pip install wandb

import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
import wandb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

run = wandb.init(project="fashion_mnist")

#Question1
(X_train, X_test), (y_train, y_test) = fashion_mnist.load_data()
# Normalise
X_train = X_train/255
X_test = X_test/255

class Neuron:
  def __init__(self):
    self.data=0.0

def visualze_data():
  plt.figure(figsize=(10,10))
  
  for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(f"Image {i+1}:")
    print(X_train[i])
  plt.show()

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
    self.bias = []

    for i in range(n_inputs):
      self.input_layer.data[i] = self.input_layer.neurons[i].data
      # self.weights_ih = np.zeros(n_inputs, dtype=float)

    self.hidden_layer = []
    self.weights_hidden = []

    for i in range(n_hidden):
      n = int(input(f"Enter number of neurons in hidden layer {i}: "))
      self.hidden_layer.append(Layer(n))
      self.weights_hidden.append(np.random.rand(0,high=1,size=(n, n_inputs)))
      n_inputs = n
      if i ==n_hidden - 1:
        self.weights_hidden.append(np.random.rand(0,high=1,size=(n_outputs, n_inputs)))
      self.bias.append(0.0)
    
    self.output_layer = Layer(n_outputs)
    self.weights_hidden.append(np.random.rand(0,high=1,size=(n_outputs, n_inputs)))
    self.lr = 0.1

    def modify(self):
      self.hidden_layer = []
      for i in range(n_hidden):
        n = int(input(f"Enter number of neurons in hidden layer {i}: "))
        self.hidden_layer.append(Layer(n_hidden))

    def forward(self, inputs):
      self.input_layer.data = inputs
      self.hidden_layer.data = np.dot(self.weights_ih, inputs) + self.bias_h
      self.hidden_layer.data = self.sigmoid(self.hidden_layer.data)
      self.output_layer.data = np.dot(self.weights_ho, self.hidden_layer.data) + self.bias_o
      self.output_layer.data = self.sigmoid(self.output_layer.data)
      return self.output_layer.data
