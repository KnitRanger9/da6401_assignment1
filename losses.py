from utils import *

class Sigmoid_loss:
    def __init__(self):
      pass
    def 
def sigmoid(x):
  f_x = lambda x: 1/(1+np.exp(-x))
  df_x = lambda x: f_x(x)*(1-f_x(x))
  return f_x(x), df_x(x)

def relu(x):
  f_x = lambda x: np.maximum(0,x)
  df_x = lambda x: np.where(x>0, 1, 0)
  return f_x(x), df_x(x)

def tanh(x):
  f_x = lambda x: np.tanh(x)
  df_x = lambda x: 1 - np.tanh(x)**2
  return f_x(x), df_x(x)

def softmax(x):
  ans=np.array([])
  for i in range(len(x)):
    np.append(ans, np.exp(x[i])/np.sum(np.exp(x)))
  return ans