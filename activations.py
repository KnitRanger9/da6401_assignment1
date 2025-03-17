import numpy as np

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