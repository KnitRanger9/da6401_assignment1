import numpy as np
from new_nn import *

#iter = update for each iteration

class Optimizer:
    def __init__(self,
                 nn: FFNeuralNetwork,
                 bp: Backpropagation,
                 lr=0.001,
                 optimizer="sgd",
                 momentum=0.9,
                 epsilon=1e-8,
                 beta=0.9,
                 beta1=0.9,
                 beta2=0.999,
                 t=0,
                 decay=0):

        self.nn, self.bp, self.lr, self.optimizer = nn, bp, lr, optimizer
        self.momentum, self.epsilon, self.beta1, self.beta2, self.beta = momentum, epsilon, beta1, beta2, beta
        self.h_weights = [np.zeros_like(w) for w in self.nn.weights]
        self.h_biases = [np.zeros_like(b) for b in self.nn.biases]
        self.hm_weights = [np.zeros_like(w) for w in self.nn.weights]
        self.hm_biases = [np.zeros_like(b) for b in self.nn.biases]
        self.t = t
        self.decay = decay

    def run(self, d_weights, d_biases, y_batch=None, x_batch=None):
        if(self.optimizer == "sgd"):
            self.SGD(d_weights, d_biases)
        elif(self.optimizer == "momentum"):
            self.MomentumGD(d_weights, d_biases)
        elif(self.optimizer == "nag"):
            self.NAG(d_weights, d_biases)
        elif(self.optimizer == "rmsprop"):
            self.RMSProp(d_weights, d_biases)
        elif(self.optimizer == "adam"):
            self.Adam(d_weights, d_biases)
        elif (self.optimizer == "nadam"):
            self.NAdam(d_weights, d_biases)
        else:
            raise Exception("Invalid optimizer")
        
    def SGD(self, d_weights, d_biases):
        for i in range(self.nn.hidden_layers + 1):
            self.nn.weights[i] -= self.lr * (d_weights[i] + self.decay * self.nn.weights[i])
            self.nn.biases[i] -= self.lr * (d_biases[i] + self.decay * self.nn.biases[i])

    def MomentumGD(self, d_weights, d_biases):
        for i in range(self.nn.hidden_layers + 1):
            self.h_weights[i] = self.momentum * self.h_weights[i] + d_weights[i]
            self.h_biases[i] = self.momentum * self.h_biases[i] + d_biases[i]

            self.nn.weights[i] -= self.lr * (self.h_weights[i] + self.decay * self.nn.weights[i])
            self.nn.biases[i] -= self.lr * (self.h_biases[i] + self.decay * self.nn.biases[i])

    def NAG(self, d_weights, d_biases):
        for i in range(self.nn.hidden_layers + 1):
            self.h_weights[i] = self.momentum * self.h_weights[i] + d_weights[i]
            self.h_biases[i] = self.momentum * self.h_biases[i] + d_biases[i]

            self.nn.weights[i] -= self.lr * (self.momentum * self.h_weights[i] + d_weights[i] + self.decay * self.nn.weights[i])
            self.nn.biases[i] -= self.lr * (self.momentum * self.h_biases[i] + d_biases[i] + self.decay * self.nn.biases[i])

    def RMSProp(self, d_weights, d_biases):
        for i in range(self.nn.hidden_layers + 1):
            self.h_weights[i] = self.momentum * self.h_weights[i] + (1 - self.momentum) * d_weights[i]**2
            self.h_biases[i] = self.momentum * self.h_biases[i] + (1 - self.momentum) * d_biases[i]**2

            self.nn.weights[i] -= (self.lr / (np.sqrt(self.h_weights[i]) + self.epsilon)) * d_weights[i] + self.decay * self.nn.weights[i] * self.lr
            self.nn.biases[i] -= (self.lr / (np.sqrt(self.h_biases[i]) + self.epsilon)) * d_biases[i] + self.decay * self.nn.biases[i] * self.lr

    def Adam(self, d_weights, d_biases):
        for i in range(self.nn.hidden_layers + 1):
            self.hm_weights[i] = self.beta1 * self.hm_weights[i] + (1 - self.beta1) * d_weights[i]
            self.hm_biases[i] = self.beta1 * self.hm_biases[i] + (1 - self.beta1) * d_biases[i]

            self.h_weights[i] = self.beta2 * self.h_weights[i] + (1 - self.beta2) * d_weights[i]**2
            self.h_biases[i] = self.beta2 * self.h_biases[i] + (1 - self.beta2) * d_biases[i]**2

            self.hm_weights_hat = self.hm_weights[i] / (1 - self.beta1**(self.t + 1))
            self.hm_biases_hat = self.hm_biases[i] / (1 - self.beta1**(self.t + 1))

            self.h_weights_hat = self.h_weights[i] / (1 - self.beta2**(self.t + 1))
            self.h_biases_hat = self.h_biases[i] / (1 - self.beta2**(self.t + 1))

            self.nn.weights[i] -= self.lr * (self.hm_weights_hat / ((np.sqrt(self.h_weights_hat)) + self.epsilon)) + self.decay * self.nn.weights[i] * self.lr
            self.nn.biases[i] -= self.lr * (self.hm_biases_hat / ((np.sqrt(self.h_biases_hat)) + self.epsilon)) + self.decay * self.nn.biases[i] * self.lr

    def NAdam(self, d_weights, d_biases):
        for i in range(self.nn.hidden_layers + 1):
            self.hm_weights[i] = self.beta1 * self.hm_weights[i] + (1 - self.beta1) * d_weights[i]
            self.hm_biases[i] = self.beta1 * self.hm_biases[i] + (1 - self.beta1) * d_biases[i]

            self.h_weights[i] = self.beta2 * self.h_weights[i] + (1 - self.beta2) * d_weights[i]**2
            self.h_biases[i] = self.beta2 * self.h_biases[i] + (1 - self.beta2) * d_biases[i]**2

            self.hm_weights_hat = self.hm_weights[i] / (1 - self.beta1 ** (self.t + 1))
            self.hm_biases_hat = self.hm_biases[i] / (1 - self.beta1 ** (self.t + 1))

            self.h_weights_hat = self.h_weights[i] / (1 - self.beta2 ** (self.t + 1))
            self.h_biases_hat = self.h_biases[i] / (1 - self.beta2 ** (self.t + 1))

            temp_update_w = self.beta1 * self.hm_weights_hat + ((1 - self.beta1) / (1 - self.beta1 ** (self.t + 1))) * d_weights[i]
            temp_update_b = self.beta1 * self.hm_biases_hat + ((1 - self.beta1) / (1 - self.beta1 ** (self.t + 1))) * d_biases[i]

            self.nn.weights[i] -= self.lr * (temp_update_w / ((np.sqrt(self.h_weights_hat)) + self.epsilon)) + self.decay * self.nn.weights[i] * self.lr
            self.nn.biases[i] -= self.lr * (temp_update_b / ((np.sqrt(self.h_biases_hat)) + self.epsilon)) + self.decay * self.nn.biases[i] * self.lr