from .tensor import Function
import numpy as np


class Add(Function):
  def forward(self, x, y):
    return x + y

  def backward(self, grad_out):
    return grad_out, grad_out


class Mul(Function):
  def forward(self, x, y):
    self.x = x
    self.y = y
    return x * y

  def backward(self, grad_out):
    return self.y * grad_out, self.x * grad_out


class Sum(Function):
  def forward(self, x, axis):
    self.x = x
    return x.sum()

  def backward(self, grad_out):
    return grad_out * np.ones_like(self.x)


class Neg(Function):
  def forward(self, x):
    return -x

  def backward(self, grad_out):
    return -grad_out


class Reshape(Function):
  def forward(self, x, shape):
    self.shape = x.shape
    return x.reshape(shape)

  def backward(self, grad_out):
    return grad_out.reshape(self.shape)


class Log(Function):
  def forward(self, x):
    self.x = x
    return np.log(x)

  def backward(self, grad_out):
    return (1 / self.x) * grad_out


class Dot(Function):
  def forward(self, x, y):
    self.x = x
    self.y = y
    return x.dot(y)

  def backward(self, grad_out):
    x_grad = grad_out.dot(self.y.T)
    y_grad = self.x.T.dot(grad_out)
    return x_grad, y_grad


class Mean(Function):
  def forward(self, x):
    self.x = x
    return np.mean(x)

  def backward(self, grad_out):
    return (1 / self.x.size) * grad_out


class Relu(Function):
  def forward(self, x):
    self.x = x
    return np.maximum(x, 0)

  def backward(self, grad_out):
    return np.where(self.x > 0, 1, 0) * grad_out


class Sigmoid(Function):
  def forward(self, x):
    ret = 1 / (1 + np.exp(-x))
    self.ret = ret
    return ret

  def backward(self, grad_out):
    return self.ret * (1 - self.ret) * grad_out


class LogSoftMax(Function):
  def forward(self, x):
    def logsumexp(x):
      c = x.max(axis=1, keepdims=True)
      return c + np.log(np.exp(x - c).sum(axis=1, keepdims=True))

    ret = x - logsumexp(x)
    self.ret = ret
    return ret

  def backward(self, grad_out):
    return grad_out - np.exp(self.ret) * grad_out.sum(axis=1, keepdims=True)
