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
  def __init__(self, data =0.0):
    self.data = data

def visualize_data():
  plt.figure(figsize=(10,10))
  
  for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(f"Image {i+1}")
  plt.show()

class Layer(Neuron):
  def __init__(self, n_input, n_output, activation, data = 0.0):
    self.n_input = n_input
    self.n_output = n_output
    self.activation = activation
    self.weights = np.random.rand(n_output, n_input)
    self.bias = np.random.rand(n_output, 1)
    self.neurons = [Neuron() for _ in range(n_output)]
    self.data = data #np.zeros(self.n_input)
    
  def forward(self, inputs):
    self.data = inputs
    for i in range(self.n_output):
      self.neurons[i].data = self.activation(np.dot(self.weights[i], inputs) + self.bias[i])

class NeuralNetwork:
  """
  variables:
    n_layers: Number of layers in the network
    activation: Activation function for hidden layers
    output_activation: Activation function for output layer
    layers: List of layers in the network
    weights: List of weights for each layer
    bias: List of bias for each layer

  methods:
  - forward: Forward pass through the network
  - backward: Backward pass through the network
  """
  def __init__(self, n_layers, activation, output_activation):
    self.n_layers = n_layers
    self.activation = activation
    self.output_activation = output_activation
    self.layers = []
    self.weights = []
    self.bias = []

    n_input = int(input(f"Enter number of neurons in Input layer: "))
    self.layers.append(Layer(n_input, 0, activation))
    for i in range(n_layers-2):
      n_output = int(input(f"Enter number of neurons in layer {i+1}: "))
      self.layers.append(Layer(n_input, n_output, activation))
      self.weights.append(self.layers[i].weights)
      self.bias.append(self.layers[i].bias)
      n_input = n_output
    n_output = int(input(f"Enter number of neurons in Output layer: "))
    n_input = len(self.layers[-1].neurons)
    self.layers.append(Layer(n_input, n_output, output_activation))
    self.weights.append(self.layers[-1].weights)
    self.bias.append(self.layers[-1].bias)


    def modify(self, n_hidden):
      self.layers = []
      for i in range(n_hidden):
        n = int(input(f"Enter number of neurons in hidden layer {i}: "))
        self.layers.append(Layer(n_hidden))

    def forward(self, inputs):
      for i in range(len(self.layers)):
        inputs = self.activation(np.dot(self.weights[i], inputs) + self.bias[i])
        self.layers[i].data = inputs
      return inputs
