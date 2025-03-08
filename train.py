#Making NN from scratch: Day 1
# !pip install 

from utils import *
import visualise
import config
import preprocessing
import losses
import optim
#Helper functions

np.random.seed(1)

wandb.init(config={"batch_size": 32, "l_rate": 0.001, "optimizer": 'nadam', "epochs": 5, "activation": "relu", "initializer": "random", "loss": "squared_error", "n_hlayers": 5, "hlayer_size": 128}, project="Deep-Learning")
myconfig = wandb.config

def visualize_data(X_train):
  plt.figure(figsize=(10,10))
  
  for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(f"Image {i+1}")
  plt.show()


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
    self.neurons = np.zeros(self.n_output,1)

  # forward inside Layer??
    
  # def forward(self, inputs):
  #   inputs = np.concatenate((inputs, np.ones((1,inputs.shape[1]))))
  #   output = self.activation(np.dot(self.weights, inputs))
  #   for i in range(self.n_output):
  #     self.neurons[i] = output[i]
  #   return output

class NeuralNetwork(Layer):
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
  def __init__(self, n_layers, activation, output_activation, optimizer):
    self.n_layers = n_layers
    self.activation = activation
    self.output_activation = output_activation
    self.layers = []
    self.weights = []
    self.bias = []
    self.optimizer = optimizer

  def construct(self):

    n_input = int(input(f"Enter number of neurons in Input layer: "))

    for i in range(self.n_layers-1):
      if i == self.n_layers-1:
        n_output = int(input(f"Enter number of neurons in Output layer: "))
        np.append(self.layers,Layer(n_input, n_output, self.output_activation))
        np.append(self.weights, self.layers[-1].weights)
        np.append(self.bias,self.layers[-1].bias)
      n_output = int(input(f"Enter number of neurons in layer {i+1}: "))
      np.append(self.layers,Layer(n_input, n_output, self.activation))
      np.append(self.weights,self.layers[i].weights)
      np.append(self.bias,self.layers[i].bias)
      n_input = n_output
    

  def forward(self, inputs):
    for i in range(len(self.layers)):
      inputs = self.activation(np.dot(self.weights[i], inputs) + self.bias[i])
      self.layers[i].data = inputs
    return inputs

  def backward(self, target):
    delta_matrix=np.zeros(self.n_layers, dtype=np.float)

if __name__ == '__main__':
  # def __init__(self, n_layers, activation, output_activation):

  run = wandb.init(project="fashion_mnist")

  #Question1
  (X_train, X_test), (y_train, y_test) = fashion_mnist.load_data()
  # Normalise
  X_train = X_train/255
  X_test = X_test/255

  visualize_data(X_train)

  n_layers=int(input("Enter the number of layers in the neural network: "))
  activation=input("Enter the activation function for hidden layers (e.g., sigmoid, ReLU, tanh): ")

  output_activation_function=softmax()
  # Create a Neural Network object with given number of layers, activation function and output activation function
  # nn=NeuralNetwork(n_layers, activation, output_activation)
  nn=NeuralNetwork(n_layers=n_layers, activation=activation_function, output_activation=output_activation_function)
  nn.construct()



#1. layers update function for weights and biases
#2. backprop function
#3 Train function (epoch, lr, etc as per optimzer used, along with accuracy params)
#4. Prediction function
#5. Integrate with wandb