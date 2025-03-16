#Making NN from scratch: Day 1
# !pip install 

from utils import *
import config
from inits import RandomInit, XavierInit
from preprocessing import MinMax, OneHot
from layers import A_x,Dense
from nn import NeuralNetwork
from activations import Sigmoid, ReLU, TanH, Softmax
from  losses import MSE_loss, Cross_entropy_loss
from optim import SGD, GDMomentum, GDNesterov, RMSProp, Adam, Nadam

np.random.seed(24)

wandb.init(project="Deep-Learning")

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

scaler = MinMax()

X_train = scaler.normalise(x_train)
X_val = scaler.normalise(x_val)
X_test = scaler.normalise(x_test)

onehot = OneHot()

y_train_encoded = onehot.transform(y_train)
y_val_encoded = onehot.transform(y_val)
y_test_encoded = onehot.transform(y_test)

y_train_encoded = np.array([img.flatten() for img in y_train_encoded]).T
y_val_encoded = np.array([img.flatten() for img in y_val_encoded]).T
y_test_encoded = np.array([img.flatten() for img in y_test_encoded]).T

layers = [A_x(X_train),
          Dense("Dense1", 64, "relu"),
          Dense("Dense2", 32, "relu"),
          Dense("Dense3", 16, "relu"),
          Dense("Dense4", 10, "softmax", last=True)]

# def __init__(self, layers, batch, num_epochs, optimizer, w_init, targets, loss_fn, validate=False, validation_inputs = None, val_target = None, optim_params = None, wandb_=False):
model = NeuralNetwork(layers,
                      batch=64,
                      num_epochs=5,
                      optimizer="RMSProp",
                      w_init="XavierInit",
                      targets=y_train_encoded,
                      loss_fn="Cross_entropy_loss",
                      validate=True,
                      validation_inputs=X_val,
                      val_target=y_val_encoded,
                      optim_params={"lr":0.001})

