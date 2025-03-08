from utils import *
from activations import *
from losses import *
from optim import *
from layers import *
from preprocessing import *

optimizers = {"sgd": SGD(), "gdm": GDMomentum(), "gdn": GDNesterov(), "rmsprop": RMSProp(), "adam": Adam(), "nadam": Nadam()}
losses = {"mse": MSE_loss(), "cross_entropy": Cross_entropy_loss()}

class NeuralNetwork():
    def __init__(self, layers, batch, num_epochs, optimizer, w_init, targets, loss_fn, validate=False, validation_inputs = None, val_target = None, optim_params = None, wandb_=False):
        self.layers = layers
        self.batch = batch
        self.num_epochs = num_epochs
        self.optimizer = optimizers[optimizer]
        self.w_init = w_init
        self.targets = targets
        self.loss_fn = losses[loss_fn]
        self.wandb_ = wandb_

        if validate is not None:
            self.X_val = validation_inputs
            self.layers[0].v_inputs = validation_inputs
            self.val_target = val_target

        self.init_params(optimizer,optim_params)

    def init_params(self, optimizer, optim_params): 
        for layer in self.layers:
            if hasattr(layer, 'initialise'):
                layer.initialise(self.w_init, optimizer, optim_params)