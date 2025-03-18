import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist, mnist
import matplotlib.pyplot as plt
import wandb

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

WANDB_PROJECT = "neural-network-sweep_mse"
DATASET = "fashion_mnist"
EPOCHS = 10



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
                'value': 'mean_squared_error' #'cross_entropy'
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

def log_examples():
    wandb.init(project=WANDB_PROJECT, entity=wandb.api.default_entity)  # Fixed entity attribute
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
        # Add small epsilon to prevent log(0)
        y_pred = np.clip(y_pred, 1e-10, 1.0)
        return -np.sum(y * np.log(y_pred)) / y.shape[0]
    elif loss_type == "mean_squared_error": # Mean Squared Error
        return np.sum((y - y_pred) ** 2) / (2 * y.shape[0])
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
            # Add small epsilon to prevent division by zero
            y_pred = np.clip(y_pred, 1e-10, 1.0)
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

    def output_activation_derivative(self, y_true, y_pred):
        # For softmax + cross-entropy, the derivative simplifies
        if self.nn.output_activation_function == "softmax" and self.loss == "cross_entropy":
            return y_pred - y_true
        # For other combinations, we need the full Jacobian
        elif self.nn.output_activation_function == "softmax" and self.loss == "mean_squared_error":
            return (y_pred - y_true)**2
        elif self.nn.output_activation_function == "softmax":
            m = y_pred.shape[0]
            jacobians = np.zeros((m, y_pred.shape[1], y_pred.shape[1]))
            for i in range(m):
                for j in range(y_pred.shape[1]):
                    for k in range(y_pred.shape[1]):
                        if j == k:
                            jacobians[i, j, k] = y_pred[i, j] * (1 - y_pred[i, j])
                        else:
                            jacobians[i, j, k] = -y_pred[i, j] * y_pred[i, k]
            return jacobians
        else:
            raise Exception("Invalid output activation function")

    def backward(self, y, y_pred):
        self.d_weights, self.d_biases = [], []
        m = y.shape[0]  # batch size

        # For softmax + cross-entropy, the gradient simplifies
        if self.nn.output_activation_function == "softmax" and self.loss == "cross_entropy":
            delta = (y_pred - y) / m
        elif self.nn.output_activation_function == "softmax" and self.loss == "mean_squared_error":
            delta = (y_pred - y) / m
        else:
            # General case (not efficient for large networks)
            delta = np.zeros_like(y_pred)
            for i in range(m):
                loss_grad = self.loss_derivative(y[i], y_pred[i])
                act_jacobian = self.output_activation_derivative(y[i], y_pred[i])
                if len(act_jacobian.shape) == 2:  # It's a Jacobian matrix
                    delta[i] = np.dot(loss_grad, act_jacobian)
                else:  # It's already the combined gradient
                    delta[i] = loss_grad * act_jacobian
            delta /= m

        # Output layer gradients
        self.d_weights.insert(0, np.dot(self.nn.post_activation[-2].T, delta))
        self.d_biases.insert(0, np.sum(delta, axis=0))

        # Hidden layers
        for l in range(self.nn.hidden_layers - 1, -1, -1):
            delta = np.dot(delta, self.nn.weights[l+1].T) * self.activation_derivative(self.nn.post_activation[l+1])
            self.d_weights.insert(0, np.dot(self.nn.post_activation[l].T, delta))
            self.d_biases.insert(0, np.sum(delta, axis=0))

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

            hm_weights_hat = self.hm_weights[i] / (1 - self.beta1**(self.t + 1))
            hm_biases_hat = self.hm_biases[i] / (1 - self.beta1**(self.t + 1))

            h_weights_hat = self.h_weights[i] / (1 - self.beta2**(self.t + 1))
            h_biases_hat = self.h_biases[i] / (1 - self.beta2**(self.t + 1))

            self.nn.weights[i] -= self.lr * (hm_weights_hat / ((np.sqrt(h_weights_hat)) + self.epsilon)) + self.decay * self.nn.weights[i] * self.lr
            self.nn.biases[i] -= self.lr * (hm_biases_hat / ((np.sqrt(h_biases_hat)) + self.epsilon)) + self.decay * self.nn.biases[i] * self.lr

    def NAdam(self, d_weights, d_biases):
        for i in range(self.nn.hidden_layers + 1):
            self.hm_weights[i] = self.beta1 * self.hm_weights[i] + (1 - self.beta1) * d_weights[i]
            self.hm_biases[i] = self.beta1 * self.hm_biases[i] + (1 - self.beta1) * d_biases[i]

            self.h_weights[i] = self.beta2 * self.h_weights[i] + (1 - self.beta2) * d_weights[i]**2
            self.h_biases[i] = self.beta2 * self.h_biases[i] + (1 - self.beta2) * d_biases[i]**2

            hm_weights_hat = self.hm_weights[i] / (1 - self.beta1 ** (self.t + 1))
            hm_biases_hat = self.hm_biases[i] / (1 - self.beta1 ** (self.t + 1))

            h_weights_hat = self.h_weights[i] / (1 - self.beta2 ** (self.t + 1))
            h_biases_hat = self.h_biases[i] / (1 - self.beta2 ** (self.t + 1))

            temp_update_w = self.beta1 * hm_weights_hat + ((1 - self.beta1) / (1 - self.beta1 ** (self.t + 1))) * d_weights[i]
            temp_update_b = self.beta1 * hm_biases_hat + ((1 - self.beta1) / (1 - self.beta1 ** (self.t + 1))) * d_biases[i]

            self.nn.weights[i] -= self.lr * (temp_update_w / ((np.sqrt(h_weights_hat)) + self.epsilon)) + self.decay * self.nn.weights[i] * self.lr
            self.nn.biases[i] -= self.lr * (temp_update_b / ((np.sqrt(h_biases_hat)) + self.epsilon)) + self.decay * self.nn.biases[i] * self.lr

def load_data(type, dataset=DATASET):
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

def train(parameters):
    x_train, y_train = load_data('train', dataset=parameters['dataset'])
    best_accuracy = 0
    best_epoch = 0
    best_model_weights = None
    best_model_biases = None

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
                    beta=parameters['beta'],
                    beta1=parameters['beta1'],
                    beta2=parameters['beta2'],
                    decay=parameters['decay'])

    batch_size = parameters['batch_size']
    x_train_act, x_val, y_train_act, y_val = train_test_split(x_train, y_train, test_size=0.1)

    print("Initial Accuracy: {}".format(np.sum(np.argmax(nn.forward(x_train), axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]))

    for epoch in range(parameters['epochs']):
        for i in range(0, x_train_act.shape[0], batch_size):
            x_batch = x_train_act[i:i+batch_size]
            y_batch = y_train_act[i:i+batch_size]

            y_pred = nn.forward(x_batch)
            d_weights, d_biases = bp.backward(y_batch, y_pred)
            opt.run(d_weights, d_biases)

        opt.t += 1

        y_pred = nn.forward(x_train_act)
        train_loss = loss(parameters['loss'], y_train_act, y_pred)
        train_accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_train_act, axis=1)) / y_train_act.shape[0]
        val_pred = nn.forward(x_val)
        val_loss = loss(parameters['loss'], y_val, val_pred)
        val_accuracy = np.sum(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1)) / y_val.shape[0]
        
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy
            best_epoch = epoch + 1
            # Save weights and biases
            best_model_weights = [w.copy() for w in nn.weights]
            best_model_biases = [b.copy() for b in nn.biases]

    if best_model_weights is not None and best_model_biases is not None:
        print(f"Restoring best model from epoch {best_epoch} with training accuracy {best_accuracy}")
        nn.weights = best_model_weights
        nn.biases = best_model_biases

    x_test, y_test = load_data('test', dataset=parameters['dataset'])
    y_pred_test = nn.forward(x_test)
    test_accuracy = np.sum(np.argmax(y_pred_test, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]

    # conf_matrix = compute_confusion_matrix(y_test, y_pred_test)
    # print("Confusion Matrix:")
    # print(conf_matrix)
    print(f"Test Accuracy: {test_accuracy}")

    return nn

def train_sweep():
    run = wandb.init()
    parameters = wandb.config
    run.name = f"{parameters['activation']}_neurons={parameters['neurons']}_layers={parameters['hidden_layers']}_lr={parameters['learning_rate']}_batch={parameters['batch_size']}_opt={parameters['optimizer']}_mom={parameters['momentum']}_init={parameters['weight_init']}"
    
    x_train, y_train = load_data('train', dataset=parameters['dataset'])
    

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
        # Shuffle training data for each epoch
        indices = np.random.permutation(len(x_train_act))
        x_train_act = x_train_act[indices]
        y_train_act = y_train_act[indices]
        
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

        

    # After training is complete, log confusion matrix
    x_test, y_test = load_data('test', dataset=parameters['dataset'])
    y_pred_test = nn.forward(x_test)
    test_accuracy = np.sum(np.argmax(y_pred_test, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
    
    print(f"Test Accuracy: {test_accuracy}")
    wandb.log({"test_accuracy": test_accuracy})
    

    return nn

def run_sweep(sweep_conf=sweep_configuration):
    print("Logging into W&B...")
    wandb.login()
    
    print("Starting sweep...")
    sweep_id = wandb.sweep(sweep_conf, project=WANDB_PROJECT)
    
    print(f"Sweep ID: {sweep_id}")
    print("Running sweep agent...")
    wandb.agent(sweep_id, function=train_sweep, count=100)
    # api = wandb.Api()
    # sweep = api.sweep(f"your-username/your-project-name/{sweep_id}")
    # best_run = sweep.best_run()
    
    print("Sweep completed!")

if __name__ == "__main__":
    log_examples()
    
    # run_sweep(sweep_configuration)
    # sweep_configuration['parameters']['loss']['value'] = 'mean_squared_error'
    run_sweep(sweep_configuration)
    # sweep_configuration['parameters']['dataset']['value'] = 'mnist'
    # run_sweep(sweep_configuration)