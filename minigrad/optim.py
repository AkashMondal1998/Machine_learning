import numpy as np


class Optimizer:
  def __init__(self, params):
    self.params = params

  def zero_grad(self):
    for param in self.params:
      if param is not None:
        param.grad = None


class SGD(Optimizer):
  def __init__(self, parameters, lr=1e-3):
    super().__init__(parameters)
    self.lr = lr

  def step(self):
    for param in self.params:
      param.data -= self.lr * param.grad


# Todo
class Adam(Optimizer):
  def __init__(self, parameters, b1=0.9, b2=0.999, e=1e-8, lr=1e-3):
    super().__init__(parameters)
    self.beta1 = b1
    self.beta2 = b2
    self.ep = e
    self.m = np.array(0, dtype=np.float32)
    self.v = np.array(0, dtype=np.float32)
    self.t = 0

  def step(self):
    for param in self._params:
      pass
