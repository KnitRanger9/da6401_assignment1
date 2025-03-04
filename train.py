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

def visualize_data():
  plt.figure(figsize=(10,10))
  
  for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(f"Image {i+1}")
  plt.show()

#Helper function
def sigmoid(x):
  f_x = lambda x: 1/(1+np.exp(-x))
  df_x = lambda x: f_x(x)*(1-f_x(x))
  return f_x, df_x

def relu(x):
  f_x = lambda x: np.maximum(0,x)
  df_x = lambda x: np.where(x>0, 1, 0)
  return f_x, df_x

def tanh(x):
  f_x = lambda x: np.tanh(x)
  df_x = lambda x: 1 - np.tanh(x)**2
  return f_x, df_x

def sgd(weights, bias, lr, grad_w, grad_b):
  weights -= lr*grad_w
  bias -= lr*grad_b
  return weights, bias

def momentum_gd(weights, bias, lr, grad_w, grad_b, beta, v_w, v_b):
  v_w = beta*v_w + lr*grad_w
  v_b = beta*v_b + lr*grad_b
  weights -= v_w
  bias -= v_b
  return weights, bias, v_w, v_b

def nest_acc_gd(weights, bias, lr, grad_w, grad_b, beta, v_w, v_b, gamma, v_w_prev, v_b_prev):
  v_w = beta*v_w + lr*grad_w
  v_b = beta*v_b + lr*grad_b
  weights -= (gamma*v_w_prev + v_w)
  bias -= (gamma*v_b_prev + v_b)
  return weights, bias, v_w, v_b, v_w_prev, v_b_prev

def rms_prop(weights, bias, lr, grad_w, grad_b, beta, v_w, v_b, epsilon):
  v_w = beta*v_w + (1-beta)*(grad_w**2)
  v_b = beta*v_b + (1-beta)*(grad_b**2)
  weights -= lr*grad_w/(np.sqrt(v_w) + epsilon)
  bias -= lr*grad_b/(np.sqrt(v_b) + epsilon)
  return weights, bias, v_w, v_b

def adam(weights, bias, lr, grad_w, grad_b, beta1, beta2, epsilon, m_w, m_b, v_w, v_b, t):
  m_w = beta1*m_w + (1-beta1)*grad_w
  m_b = beta1*m_b + (1-beta1)*grad_b
  v_w = beta2*v_w + (1-beta2)*(grad_w**2)
  v_b = beta2*v_b + (1-beta2)*(grad_b**2)
  m_w_hat = m_w/(1-beta1**t)
  m_b_hat = m_b/(1-beta1**t)
  v_w_hat = v_w/(1-beta2**t)
  v_b_hat = v_b/(1-beta2**t)
  weights -= lr*m_w_hat/(np.sqrt(v_w_hat) + epsilon)
  bias -= lr*m_b_hat/(np.sqrt(v_b_hat) + epsilon)
  return weights, bias, m_w, m_b, v_w, v_b

def nadam(weights, bias, lr, epsilon, beta1, beta2, v_w, v_b, m_w, m_b):
  m_w_prev = beta1*m_w - (1-beta1)*grad_w
  m_b_prev = beta1*m_b - (1-beta1)*grad_b
  v_w_prev = beta2*v_w - (1-beta2)*grad_w**2
  v_b_prev = beta2*v_b - (1-beta2)*grad_b**2
  weights -= lr*(m_w_prev/(np.sqrt(v_w_prev) + epsilon) + beta1*(m_w - m_w_prev)/(np.sqrt(v_w) + epsilon))
  bias -= lr*(m_b_prev/(np.sqrt(v_b_prev) + epsilon) + beta1*(m_b - m_b_prev)/(np.sqrt(v_b) + epsilon))



class Layer:
  """
  meta: A Layer object combining several neurons to form a single layer

  variables:
  - n_input: Number of neurons in the previous layer
  - n_output: Number of neurons in the current layer
  - activation: Activation function for the layer
  - weights: Weights connecting the previous layer to the current layer
  - bias: Bias for the layer

  methods:
  - forward: Forward pass through the layer
  - backward: Backward pass through the layer to update weights and bias
  """
  def __init__(self, n_input, n_output):
    self.n_input = n_input
    self.n_output = n_output
    self.weights = np.random.rand(n_output, n_input+1)
    # self.bias = np.random.rand(n_output, 1)
    self.neurons = np.zeros(self.n_output,1)
    
  def forward(self, inputs):
    inputs = np.concatenate((inputs, np.ones((1,inputs.shape[1]))))
    output = self.activation(np.dot(self.weights, inputs))
    for i in range(self.n_output):
      self.neurons[i] = self.activation(np.dot(self.weights[i], inputs) + self.bias[i])

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
  - construct: Creates a NeuralNetwork
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

  def construct(self):

    n_input = int(input(f"Enter number of neurons in Input layer: "))

    for i in range(self.n_layers-1):
      if i == self.n_layers-1:
        n_output = int(input(f"Enter number of neurons in Output layer: "))
        self.layers.append(Layer(n_input, n_output, self.output_activation))
        self.weights.append(self.layers.weights)
        self.bias.append(self.layers.bias)
      n_output = int(input(f"Enter number of neurons in layer {i+1}: "))
      self.layers.append(Layer(n_input, n_output, self.activation))
      self.weights.append(self.layers[i].weights)
      self.bias.append(self.layers[i].bias)
      n_input = n_output
    # n_output = int(input(f"Enter number of neurons in Output layer: "))
    # n_input = len(self.layers[-1].neurons)
    

  def forward(self, inputs):
    for i in range(len(self.layers)):
      inputs = self.activation(np.dot(self.weights[i], inputs) + self.bias[i])
      self.layers[i].data = inputs
    return inputs

  #Weights and Bias update
  def w_b_update(layer, weights, bias):
    try:


  # def train(self, X, epochs=100):
  #   input=X
  #   for _ in range(epochs):
  #     for i in range(self.n_layers):
  #       w = self.weights[i]
  #       b = self.bias[i]
  #       y = np.dot(w.T, input) + b
      

#1. layers update function for weights and biases
#2. backprop function
#3 Train function (epoch, lr, etc as per optimzer used, along with accuracy params)
#4. Prediction function
#5. Integrate with wandb