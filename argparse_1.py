import argparse
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Neural Network Training Arguments")
        
    # Weights & Biases arguments
    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", type=str, default="myname", help="Wandb Entity used to track experiments in the Weights & Biases dashboard")
    
    # Dataset and training parameters
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset to use")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs to train neural network")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size used to train neural network")
    
    # Loss function
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", help="Loss function")
    
    # Optimizer parameters
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd", help="Optimizer type")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum for momentum and nag optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta for rmsprop optimizer")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5, help="Beta1 for adam and nadam optimizers")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5, help="Beta2 for adam and nadam optimizers")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001, help="Epsilon for optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay for optimizers")
    
    # Neural Network architecture parameters
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="random", help="Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=4, help="Number of neurons in each hidden layer")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid", help="Activation function")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize Weights & Biases
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    
    # Print arguments (for debugging)
    print("Training Configuration:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    
    # Here, you can add your model training logic

if __name__ == "__main__":
    main()
