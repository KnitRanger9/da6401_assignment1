from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Neural Network Training Arguments")
        
    # Weights & Biases arguments
    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname")
    parser.add_argument("-we", "--wandb_entity", type=str, default="myname")
    
    # Dataset and training parameters
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist")
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    
    # Loss function
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy")
    
    # Optimizer parameters
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1)
    parser.add_argument("-m", "--momentum", type=float, default=0.5)
    parser.add_argument("-beta", "--beta", type=float, default=0.5)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5)
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0)
    
    # Neural Network architecture parameters
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="random")
    parser.add_argument("-nhl", "--num_layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", type=int, default=4)
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid")

    parser = argparse.ArgumentParser(description='Train a neural network on Fashion MNIST')
    parser.add_argument('--wandb_entity', type=str, default='', help='WandB entity name')
    parser.add_argument('--wandb_project', type=str, default='fashion-mnist-nn', help='WandB project name')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--hidden_layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('--hidden_size', type=int, default=64, help='Size of hidden layers')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use: sgd, momentum, nesterov, rmsprop, adam, nadam')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')
    parser.add_argument('--weight_init', type=str, default='XavierInit', help='Weight initialization: RandomInit, XavierInit')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function: sigmoid, tanh, relu')
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()

