# DA6400-Assignments: Assignment 1

## Authors: Aritra Maji (DA24M004)

1. There are two files train_wandb.py and train.py. train_wandb.py is to perform sweep using wandb, and train.py is to test the model for specific hyperparameters. 

2. For creating a model, the syntax is given below. It is almost the same as in tensorflow/keras.

    ```python
    # define the model architecture in the list passed to NeuralNet() class
    nn = FFNeuralNetwork(input_size=784,
                         hid_layers=4,
                         neurons=32,
                         output_size=10,
                         act_func='relu',
                         out_act_func='softmax',
                         weight_init='xavier')
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
    ```

3. To compile the model so that it knows what loss and optimizer it's working with, use the Optimizer class

    ```python
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
    ```

4. To train the model, you have to pass the training data in batched format

    ```python
    model.forward(X_batch)
    ```



6. For running a wandb sweep, I have made the run_sweep() function where you have to pass a dictionary containing all the sweep parameters

    ```python

    train_sweep()
    ```

## Available options for the neural network

1. Loss functions

    ```python
    CrossEntropyLoss()
    MSELoss()

    ```

2. Activation functions

    ```python
    sigmoid
    tanh
    relu
    identity

    ```

3. Optimizers 

    ```python
    SGD()
    MomentumGD()
    NAG()
    RMSProp()
    Adam()
    NAdam()

    ```
