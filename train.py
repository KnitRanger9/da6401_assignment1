#Making NN from scratch: Day 1
# !pip install 

from utils import *
import config
from inits import RandomInit, XavierInit
from preprocessing import MinMax, OneHot
from layers import A_x,Dense
from new_nn import NeuralNetwork
from activations import Sigmoid, ReLU, TanH, Softmax
from  losses import MSE_loss, Cross_entropy_loss
from optim import SGD, GDMomentum, GDNesterov, RMSProp, Adam, Nadam

np.random.seed(24)

wandb.init(project="Deep-Learning", entity="da24m004-iitmaana")

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

x_train = x_train.reshape(x_train.shape[0], -1).T
x_val = x_val.reshape(x_val.shape[0], -1).T
x_test = x_test.reshape(x_test.shape[0], -1).T

scaler = MinMax()
X_train = scaler.normalise(x_train)
X_val = scaler.normalise(x_val)
X_test = scaler.normalise(x_test)

onehot = OneHot()
y_train_encoded = onehot.fit_transform(y_train)
y_val_encoded = onehot.fit_transform(y_val)
y_test_encoded = onehot.fit_transform(y_test)

print(f"Train data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Test data shape: {X_test.shape}")

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


model.train(x_train, y_train_encoded, x_val, y_val_encoded)
val_accuracy = model.calculate_accuracy(x_val, y_val_encoded)
train_accuracy = model.calculate_accuracy(x_train, y_train_encoded)

wandb.log({
    "train_accuracy": train_accuracy,
    "val_accuracy": val_accuracy,
    "final_train_loss": model.history["train_loss"][-1],
    "final_val_loss": model.history["val_loss"][-1]
})

sweep_config = {
        'method': 'bayes',  # Use Bayesian optimization for efficient searching
        'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
        'parameters': {
            'epochs': {'values': [5, 10]},
            'hidden_layers': {'values': [3, 4, 5]},
            'hidden_size': {'values': [32, 64, 128]},
            'weight_decay': {'values': [0, 0.0005, 0.5]},
            'lr': {'values': [1e-3, 1e-4]},
            'optimizer': {'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']},
            'batch_size': {'values': [16, 32, 64]},
            'weight_init': {'values': ['random', 'xavier']},
            'activation': {'values': ['sigmoid', 'tanh', 'relu']}
        }
    }
sweep_id = wandb.sweep(sweep_config, project="Deep-Learning", entity="da24m004-iitmaana")

wandb.agent(sweep_id, model, count = 50)

#Evaluate best model:
api = wandb.Api()
sweep = api.sweep(f"your-username/fashion-mnist-nn/{sweep_id}")
best_run = sweep.best_run()
best_config = best_run.config

print(f"Best configuration found: {best_config}")

# Train model with best hyperparameters
hidden_layers = [best_config["hidden_size"]] * best_config["hidden_layers"]

best_model = NeuralNetwork(
    input_dim=784,
    hidden_layers=hidden_layers,
    activation=best_config["activation"],
    weight_init=best_config["weight_init"],
    optimizer=best_config["optimizer"],
    learning_rate=best_config["lr"],
    weight_decay=best_config["weight_decay"],
    batch_size=best_config["batch_size"],
    num_epochs=best_config["epochs"]
)

best_model.train(x_train, y_train_encoded, x_val, y_val_encoded)

# Evaluate on test set
test_accuracy = best_model.calculate_accuracy(x_test, y_test_encoded)
print(f"Test accuracy with best model: {test_accuracy:.4f}")

# Generate predictions
y_pred = best_model.predict(x_test)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[class_dict[i] for i in range(10)],
            yticklabels=[class_dict[i] for i in range(10)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix - Test Accuracy: {test_accuracy:.4f}')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
wandb.log({"Confusion Matrix": wandb.Image('confusion_matrix.png')})

# Display some sample predictions
plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    sample_idx = np.random.randint(0, x_test.shape[0])
    img = x_test[sample_idx]
    plt.imshow(img, cmap='gray')
    
    pred_class = class_dict[y_pred[sample_idx]]
    true_class = class_dict[y_test[sample_idx]]
    
    if y_pred[sample_idx] == y_test[sample_idx]:
        color = 'green'
    else:
        color = 'red'
        
    plt.title(f"Pred: {pred_class}\nTrue: {true_class}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.savefig('sample_predictions.png')
wandb.log({"Sample Predictions": wandb.Image('sample_predictions.png')})
