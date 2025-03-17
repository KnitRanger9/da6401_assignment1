#Making NN from scratch: Day 1
# !pip install 

from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist, mnist
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# from utils import *
import config
# from inits import RandomInit, XavierInit
# from preprocessing import MinMax, OneHot
# from layers import A_x,Dense
from new_nn import *
# from activations import Sigmoid, ReLU, TanH, Softmax
from  losses import *
from optim import *
import wandb

np.random.seed(24)

wandb.init(project="Deep-Learning", entity="da24m004-iitmaana")

# Q1: Load the dataset

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

n_class=10

class_dict = {
  0: "T-shirt/top",
  1: "Trouser",
  2: "Pullover",
  3: "Dress",
  4: "Coat",
  5: "Sandal",
  6: "Shirt",
  7: "Sneaker",
  8: "Bag",
  9: "Ankle boot"
  }

def visualize_data(X_train):
  plt.figure(figsize=(10,10))
  
  for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(f"Image {i+1}")
  plt.show()

plt.figure(figsize=[10, 5])
images = []
classes = []

for i in range(n_class):
    position = next((x for x in range(len(y_train)) if y_train[x] == i), None)
    image = x_train[position,:,:]
    plt.subplot(2, 5, i+1)
    plt.imshow(image)
    plt.title(class_dict[i])
    images.append(image)
    classes.append(class_dict[i])
    
wandb.log({"Question 1": [wandb.Image(img, caption=caption) for img, caption in zip(images, classes)]})

#Q2

import warnings
warnings.filterwarnings("ignore")

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

def parse_args():
    parser = argparse.ArgumentParser(description='Neural Network Training with Various Optimizers')
    parser.add_argument('--tune', action='store_true', 
                   help='Enable hyperparameter tuning mode')
    parser.add_argument('-wp', '--wandb_project', type=str, default="neural-network-sweep",
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', type=str, default=None,
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
    parser.add_argument('-d', '--dataset', type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"],
                        help='Dataset to use for training and testing')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs to train neural network.')
    parser.add_argument('-b', '--batch_size', type=int, default=4,
                        help='Batch size used to train neural network.')
    parser.add_argument('-l', '--loss', type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"],
                        help='Loss function to use for training')
    parser.add_argument('-o', '--optimizer', type=str, default="sgd", 
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                        help='Optimizer to use for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1,
                        help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum', type=float, default=0.5,
                        help='Momentum used by momentum and nag optimizers.')
    parser.add_argument('-beta', '--beta', type=float, default=0.5,
                        help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5,
                        help='Beta1 used by adam and nadam optimizers.')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5,
                        help='Beta2 used by adam and nadam optimizers.')
    parser.add_argument('-eps', '--epsilon', type=float, default=0.000001,
                        help='Epsilon used by optimizers.')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0,
                        help='Weight decay used by optimizers.')
    parser.add_argument('-w_i', '--weight_init', type=str, default="random", choices=["random", "xavier"],
                        help='Weight initialization method')
    parser.add_argument('-nhl', '--num_layers', type=int, default=1,
                        help='Number of hidden layers used in feedforward neural network.')
    parser.add_argument('-sz', '--hidden_size', type=int, default=4,
                        help='Number of hidden neurons in a feedforward layer.')
    parser.add_argument('-a', '--activation', type=str, default="sigmoid", 
                        choices=["identity", "sigmoid", "tanh", "relu"],
                        help='Activation function to use')
    
    return parser.parse_args()

def log_examples():
    wandb.init(project=WANDB_PROJECT, entity=wandb.api.api.default_entity)
    # log one image of each class
    (x_train, y_train), _ = fashion_mnist.load_data()
    for i in range(10):
        wandb.log({"Examples": [wandb.Image(x_train[y_train == i][0], caption=CLASS_NAMES[i])]})
    # wandb.finish()

def initialize_weights(self):
    self.weights.append(np.random.randn(self.input_size, self.neurons))
    for _ in range(self.hidden_layers - 1):
        self.weights.append(np.random.randn(self.neurons, self.neurons))
    self.weights.append(np.random.randn(self.neurons, self.output_size))

    if self.weight_init == "xavier":
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] * np.sqrt(1 / self.weights[i].shape[0])

def initiate_biases(self):
    for _ in range(self.hidden_layers):
        self.biases.append(np.zeros(self.neurons))
    self.biases.append(np.zeros(self.output_size))


        
