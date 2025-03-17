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