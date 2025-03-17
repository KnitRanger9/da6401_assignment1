import numpy as np

class MSE_loss:
  def __init__(self):
    pass
  def loss(self, x, y):
    return (x-y)**2
  def differential(self, x, y):
    return 2*(x-y)
  
class Cross_entropy_loss:
  def __init__(self):
    pass
  def loss(self, x, y):
    return -np.sum(y*np.log(x))
  def differential(self, x, y):
    return -y/x
