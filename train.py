import os
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist, mnist
import wandb

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def parse_args():
    parser = argparse.ArgumentParser(description='Neural Network Training with Various Optimizers')
    parser.add_argument('-s', '--sweep', action='store_true',
                       help='Run hyperparameter sweep with wandb')
    parser.add_argument('-sc', '--sweep_count', type=int, default=100,
                        help='Number of runs to execute during the sweep')
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

def log_examples(wandb_project, wandb_entity):
    wandb.init(project=wandb_project, entity=wandb_entity)
    # log one image of each class
    (x_train, y_train), _ = fashion_mnist.load_data()
    for i in range(10):
        wandb.log({"Examples": [wandb.Image(x_train[y_train == i][0], caption=CLASS_NAMES[i])]})
    wandb.finish()

class FFNeuralNetwork():
    def __init__(self,
                neurons=128,
                hid_layers=4,
                input_size=784,
                output_size=10,
                act_func="sigmoid",
                weight_init="random",
                out_act_func="softmax",
                init_toggle=True):

        self.neurons, self.hidden_layers = neurons, hid_layers
        self.weights, self.biases = [], []
        self.input_size, self.output_size = input_size, output_size
        self.activation_function, self.weight_init = act_func, weight_init
        self.output_activation_function = out_act_func

        if init_toggle:
            self.initialize_weights()
            self.initiate_biases()

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

    def activation(self, x):
        if self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == "tanh":
            return np.tanh(x)
        elif self.activation_function == "relu":
            return np.maximum(0, x)
        elif self.activation_function == "identity":
            return x
        else:
            raise Exception("Invalid activation function")

    def output_activation(self, x):
        if self.output_activation_function == "softmax":
            max_x = np.max(x, axis=1)
            max_x = max_x.reshape(max_x.shape[0], 1)
            exp_x = np.exp(x - max_x)
            softmax_mat = exp_x / np.sum(exp_x, axis=1).reshape(exp_x.shape[0], 1)
            return softmax_mat
        else:
            raise Exception("Invalid output activation function")

    def forward(self, x):
        self.pre_activation, self.post_activation = [x], [x]

        for i in range(self.hidden_layers):
            self.pre_activation.append(np.matmul(self.post_activation[-1], self.weights[i]) + self.biases[i])
            self.post_activation.append(self.activation(self.pre_activation[-1]))

        self.pre_activation.append(np.matmul(self.post_activation[-1], self.weights[-1]) + self.biases[-1])
        self.post_activation.append(self.output_activation(self.pre_activation[-1]))

        return self.post_activation[-1]

def loss(loss_type, y, y_pred):
    if loss_type == "cross_entropy": # Cross Entropy
        return -np.sum(y * np.log(y_pred))
    elif loss_type == "mean_squared_error": # Mean Squared Error
        return np.sum((y - y_pred) ** 2) / 2
    else:
        raise Exception("Invalid loss function")

class Backpropagation():
    def __init__(self,
                 nn: FFNeuralNetwork,
                 loss="cross_entropy",
                 act_func="sigmoid"):

        self.nn, self.loss, self.activation_function = nn, loss, act_func

    def loss_derivative(self, y, y_pred):
        if self.loss == "cross_entropy":
            return -y / y_pred
        elif self.loss == "mean_squared_error":
            return (y_pred - y)
        else:
            raise Exception("Invalid loss function")

    def activation_derivative(self, x):
        # x is the post-activation value
        if self.activation_function == "sigmoid":
            return x * (1 - x)
        elif self.activation_function == "tanh":
            return 1 - x ** 2
        elif self.activation_function == "relu":
            return (x > 0).astype(int)
        elif self.activation_function == "identity":
            return np.ones(x.shape)
        else:
            raise Exception("Invalid activation function")

    def output_activation_derivative(self, y, y_pred):
        if self.nn.output_activation_function == "softmax":
            # derivative of softmax is a matrix
            return np.diag(y_pred) - np.outer(y_pred, y_pred)
        else:
            raise Exception("Invalid output activation function")

    def backward(self, y, y_pred):
        self.d_weights, self.d_biases = [], []
        self.d_h, self.d_a = [], []

        self.d_h.append(self.loss_derivative(y, y_pred))
        output_derivative_matrix = []
        for i in range(y_pred.shape[0]):
            output_derivative_matrix.append(np.matmul(self.loss_derivative(y[i], y_pred[i]), self.output_activation_derivative(y[i], y_pred[i])))
        self.d_a.append(np.array(output_derivative_matrix))

        for i in range(self.nn.hidden_layers, 0, -1):
            self.d_weights.append(np.matmul(self.nn.post_activation[i].T, self.d_a[-1]))
            self.d_biases.append(np.sum(self.d_a[-1], axis=0))
            self.d_h.append(np.matmul(self.d_a[-1], self.nn.weights[i].T))
            self.d_a.append(self.d_h[-1] * self.activation_derivative(self.nn.post_activation[i]))

        self.d_weights.append(np.matmul(self.nn.post_activation[0].T, self.d_a[-1]))
        self.d_biases.append(np.sum(self.d_a[-1], axis=0))

        self.d_weights.reverse()
        self.d_biases.reverse()
        for i in range(len(self.d_weights)):
            self.d_weights[i] = self.d_weights[i] / y.shape[0]
            self.d_biases[i] = self.d_biases[i] / y.shape[0]

        return self.d_weights, self.d_biases

class Optimizer():
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

def load_data(type, dataset='fashion_mnist'):
    x, y, x_test, y_test = None, None, None, None

    if dataset == 'mnist':
        (x, y), (x_test, y_test) = mnist.load_data()
    elif dataset == 'fashion_mnist':
        (x, y), (x_test, y_test) = fashion_mnist.load_data()

    if type == 'train':
        x = x.reshape(x.shape[0], 784) / 255
        y = np.eye(10)[y]
        return x, y
    elif type == 'test':
        x_test = x_test.reshape(x_test.shape[0], 784) / 255
        y_test = np.eye(10)[y_test]
        return x_test, y_test

def train(args):
    x_train, y_train = load_data('train', dataset=args.dataset)
    best_accuracy = 0
    best_epoch = 0
    best_model_weights = None
    best_model_biases = None

    nn = FFNeuralNetwork(input_size=784,
                         hid_layers=args.num_layers,
                         neurons=args.hidden_size,
                         output_size=10,
                         act_func=args.activation,
                         out_act_func="softmax",
                         weight_init=args.weight_init)
    bp = Backpropagation(nn=nn,
                         loss=args.loss,
                         act_func=args.activation)
    opt = Optimizer(nn=nn,
                    bp=bp,
                    lr=args.learning_rate,
                    optimizer=args.optimizer,
                    momentum=args.momentum,
                    epsilon=args.epsilon,
                    beta=args.beta,
                    beta1=args.beta1,
                    beta2=args.beta2,
                    decay=args.weight_decay)

    batch_size = args.batch_size
    x_train_act, x_val, y_train_act, y_val = train_test_split(x_train, y_train, test_size=0.1)

    print("Initial Accuracy: {}".format(np.sum(np.argmax(nn.forward(x_train), axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]))

    for epoch in range(args.epochs):
        for i in range(0, x_train_act.shape[0], batch_size):
            x_batch = x_train_act[i:i+batch_size]
            y_batch = y_train_act[i:i+batch_size]

            y_pred = nn.forward(x_batch)
            d_weights, d_biases = bp.backward(y_batch, y_pred)
            opt.run(d_weights, d_biases)

        opt.t += 1

        y_pred = nn.forward(x_train_act)
        train_loss = loss(args.loss, y_train_act, y_pred)
        train_accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_train_act, axis=1)) / y_train_act.shape[0]
        val_loss = loss(args.loss, y_val, nn.forward(x_val))
        val_accuracy = np.sum(np.argmax(nn.forward(x_val), axis=1) == np.argmax(y_val, axis=1)) / y_val.shape[0]
        
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy
            best_epoch = epoch + 1
            # Save weights and biases
            best_model_weights = [w.copy() for w in nn.weights]
            best_model_biases = [b.copy() for b in nn.biases]

    x_test, y_test = load_data('test', dataset=args.dataset)
    y_pred_test = nn.forward(x_test)
    test_accuracy = np.sum(np.argmax(y_pred_test, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
    
    # Compute and print confusion matrix
    conf_matrix = compute_confusion_matrix(y_test, y_pred_test)
    print(f"Confusion Matrix (Best Model, Training Accuracy: {best_accuracy}):")
    print(conf_matrix)
    
    print(f"Test Accuracy: {test_accuracy}")

    return nn

sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize',
        'name': 'val_accuracy'
    },
    'parameters': {
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'learning_rate': {
            'values': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
        },
        'neurons': {
            'values': [16, 32, 64, 128]
        },
        'hidden_layers': {
            'values': [1, 2, 3, 4]
        },
        'activation': {
            'values': ['relu', 'tanh', 'sigmoid', 'identity']
        },
        'weight_init': {
            'values': ['xavier', 'random']
        },
        'optimizer': {
            'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']
        },
        'momentum': {
            'values': [0.7, 0.8, 0.9]
        },
        'input_size': {
            'value': 784
        },
        'output_size': {
            'value': 10
        },
        'loss': {
            'value': 'cross_entropy'
        },
        'epochs': {
            'value': 10
        },
        'beta': {
            'value': 0.9
        },
        'beta1': {
            'value': 0.9
        },
        'beta2': {
            'value': 0.999
        },
        'output_activation': {
            'value': 'softmax'
        },
        'epsilon': {
            'value': 1e-8
        },
        'decay': {
            'values': [0, 0.0005, 0.005]
        },
        'dataset': {
            'value': 'fashion_mnist'
        }
    }
}

def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    """
    Compute confusion matrix from true labels and predicted labels.
    
    Parameters:
    y_true (numpy.ndarray): One-hot encoded true labels or class indices
    y_pred (numpy.ndarray): Predicted probabilities or class indices
    num_classes (int): Number of classes
    
    Returns:
    numpy.ndarray: Confusion matrix of shape (num_classes, num_classes)
    """
    # Convert one-hot encoded labels to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Initialize confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    # Fill the confusion matrix
    for i in range(len(y_true)):
        conf_matrix[y_true[i], y_pred[i]] += 1
    
    return conf_matrix

def train_sweep():
    run = wandb.init()
    parameters = wandb.config
    run.name = f"{parameters['activation']}_neurons={parameters['neurons']}_layers={parameters['hidden_layers']}_lr={parameters['learning_rate']}_batch={parameters['batch_size']}_opt={parameters['optimizer']}_mom={parameters['momentum']}_init={parameters['weight_init']}"
    
    x_train, y_train = load_data('train', dataset=parameters['dataset'])
    best_accuracy = 0
    best_params = None
    best_nn = None

    nn = FFNeuralNetwork(input_size=parameters['input_size'],
                         hid_layers=parameters['hidden_layers'],
                         neurons=parameters['neurons'],
                         output_size=parameters['output_size'],
                         act_func=parameters['activation'],
                         out_act_func=parameters['output_activation'],
                         weight_init=parameters['weight_init'])
    bp = Backpropagation(nn=nn,
                         loss=parameters['loss'],
                         act_func=parameters['activation'])
    opt = Optimizer(nn=nn,
                    bp=bp,
                    lr=parameters['learning_rate'],
                    optimizer=parameters['optimizer'],
                    momentum=parameters['momentum'],
                    epsilon=parameters['epsilon'],
                    beta1=parameters['beta1'],
                    beta2=parameters['beta2'],
                    decay=parameters['decay'])

    batch_size = parameters['batch_size']
    x_train_act, x_val, y_train_act, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    print("Initial Accuracy: {}".format(np.sum(np.argmax(nn.forward(x_train), axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]))
    

    for epoch in range(parameters['epochs']):
        for i in range(0, x_train_act.shape[0], batch_size):
            x_batch = x_train_act[i:i+batch_size]
            y_batch = y_train_act[i:i+batch_size]

            y_pred = nn.forward(x_batch)
            d_weights, d_biases = bp.backward(y_batch, y_pred)
            opt.run(d_weights, d_biases, y_batch, x_batch)

        opt.t += 1

        y_pred = nn.forward(x_train_act)
        train_loss = loss(parameters['loss'], y_train_act, y_pred)
        train_accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_train_act, axis=1)) / y_train_act.shape[0]
        val_pred = nn.forward(x_val)
        val_loss = loss(parameters['loss'], y_val, val_pred)
        val_accuracy = np.sum(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1)) / y_val.shape[0]

        print(f"Epoch: {epoch + 1}, Loss: {train_loss}")
        print(f"Train Accuracy: {train_accuracy}")
        print(f"Validation Accuracy: {val_accuracy}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

    if train_accuracy > best_accuracy:
        best_accuracy = train_accuracy
        best_params = parameters.copy()
        # Create a deep copy of the neural network
        best_nn = FFNeuralNetwork(
            neurons=nn.neurons,
            hid_layers=nn.hidden_layers,
            input_size=nn.input_size,
            output_size=nn.output_size,
            act_func=nn.activation_function,
            weight_init=nn.weight_init,
            out_act_func=nn.output_activation_function,
            init_toggle=False
        )
        best_nn.weights = [w.copy() for w in nn.weights]
        best_nn.biases = [b.copy() for b in nn.biases]

    # After training is complete, log confusion matrix
    x_test, y_test = load_data('test', dataset=parameters['dataset'])
    y_pred_test = nn.forward(x_test)
    test_accuracy = np.sum(np.argmax(y_pred_test, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
    
    print(f"Test Accuracy: {test_accuracy}")
    wandb.log({"test_accuracy": test_accuracy})

    if best_nn is not None:
        best_y_pred = best_nn.forward(x_test)
        best_test_accuracy = np.sum(np.argmax(best_y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        conf_matrix = compute_confusion_matrix(y_test, best_y_pred)
        
        print("Best Parameters (Based on Training Accuracy):")
        for key, value in best_params.items():
            if key in ['input_size', 'output_size', 'loss', 'epochs', 'output_activation', 'dataset']:
                continue
            print(f"  {key}: {value}")
        print(f"Best Training Accuracy: {best_accuracy}")
        print(f"Test Accuracy with Best Parameters: {best_test_accuracy}")
        print("Confusion Matrix (Best Parameters):")
        print(conf_matrix)
        wandb.log({'conf_mat': wandb.plot.confusion_matrix(
            probs=None, 
            y_true=np.argmax(y_test, axis=1), 
            preds=np.argmax(y_pred_test, axis=1), 
            class_names=CLASS_NAMES)
        })

    return nn

def run_single_sweep(args):
    print("Logging into W&B...")
    wandb.login()
    
    print("Starting sweep...")
    # Update sweep config with command line args
    update_sweep_config(sweep_configuration, args)
    
    sweep_id = wandb.sweep(sweep_configuration, project=args.wandb_project, entity=args.wandb_entity)
    
    print(f"Sweep ID: {sweep_id}")
    print("Running sweep agent...")
    wandb.agent(sweep_id, function=train_sweep, count=100)
    
    print("Sweep completed!")

def update_sweep_config(config, args):
    """Update sweep config with command line args"""
    config['parameters']['dataset']['value'] = args.dataset
    config['parameters']['epochs']['value'] = args.epochs
    config['parameters']['batch_size']['values'] = [args.batch_size]
    config['parameters']['loss']['value'] = args.loss
    config['parameters']['optimizer']['values'] = [args.optimizer]
    config['parameters']['learning_rate']['values'] = [args.learning_rate]
    config['parameters']['momentum']['values'] = [args.momentum]
    config['parameters']['beta']['value'] = args.beta
    config['parameters']['beta1']['value'] = args.beta1
    config['parameters']['beta2']['value'] = args.beta2
    config['parameters']['epsilon']['value'] = args.epsilon
    config['parameters']['decay']['values'] = [args.weight_decay]
    config['parameters']['weight_init']['values'] = [args.weight_init.lower()]
    config['parameters']['hidden_layers']['values'] = [args.num_layers]
    config['parameters']['neurons']['values'] = [args.hidden_size]
    
    return config

# Adding the main execution block
if __name__ == "__main__":
    args = parse_args()
    
    if args.wandb_project:
        # If running with wandb integration
        if args.sweep:
            try:
                # Try to log example images once
                log_examples(args.wandb_project, args.wandb_entity)
                # Run hyperparameter sweep
                run_single_sweep(args)
            except Exception as e:
                print(f"Error running wandb sweep: {e}")
                print("Falling back to regular training...")
                train(args)
        else:
            # Regular training with wandb
            wandb.init(project=args.wandb_project, entity=args.wandb_entity)
            nn = train(args)
            wandb.finish()

    else:
        # Regular training without wandb
        nn = train(args)
        
        # Evaluate on test set
        x_test, y_test = load_data('test', dataset=args.dataset)
        y_pred = nn.forward(x_test)
        test_accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        print(f"Test Accuracy: {test_accuracy}")