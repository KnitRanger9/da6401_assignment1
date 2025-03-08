from utils import *

def visualize_data(X_train):
  plt.figure(figsize=(10,10))
  
  for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(f"Image {i+1}")
  plt.show()