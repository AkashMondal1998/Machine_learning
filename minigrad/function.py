from .tensor import Function
from .helpers import _expand,_reduce
import numpy as np


class Add(Function):
  def forward(self, x, y):
    self.x_shape = x.shape
    self.y_shape = y.shape
    return x + y

  def backward(self, grad_out):
    return _reduce(grad_out,self.x_shape), _reduce(grad_out,self.y_shape)

class Sub(Function):
  def forward(self,x,y):
    self.x_shape = x.shape
    self.y_shape = y.shape
    return x - y
  
  def backward(self,grad_out): 
    return _reduce(grad_out,self.x_shape), _reduce(-grad_out,self.y_shape)

class Mul(Function):
  def forward(self, x, y):
    self.x = x
    self.y = y
    return x * y

  def backward(self, grad_out): 
    return _reduce(self.y * grad_out,self.x.shape), _reduce(self.x * grad_out,self.y.shape)
  

class Div(Function):
  def forward(self, x, y):
    self.x = x
    self.y = y
    return x / y

  def backward(self, grad_out):
    grad_x = self.y**-1
    grad_y = self.x * -self.y**-2
    return _reduce(grad_x * grad_out,self.x.shape), _reduce(grad_y * grad_out,self.y.shape)

class Pow(Function):
  def forward(self,x,y):
    self.x = x
    self.y = y
    return x ** y
  
  def backward(self,grad_out):
    return _reduce(self.y * (self.x**(self.y - 1)) * grad_out,self.x.shape), _reduce((self.x**self.y)* np.log(self.x) * grad_out,self.y.shape)

class Dot(Function):
  def forward(self, x, y):
    self.x = x
    self.y = y
    return x.dot(y)

  def backward(self, grad_out):
    x_grad = np.dot(grad_out,self.y.T)
    y_grad = np.dot(self.x.T,grad_out)
    return x_grad, y_grad


class Maximum(Function):
  def forward(self,x,y):
    self.x = x
    self.y = y
    return np.maximum(x,y)
  
  def backward(self,grad_out):
    grad_x = np.where(self.x > self.y,1,np.where(self.x < self.y,0,0.5))
    grad_y = np.where(self.y > self.x,1,np.where(self.y < self.x,0,0.5))
    return _reduce(grad_x * grad_out,self.x.shape) ,_reduce(grad_y * grad_out,self.y.shape)


class Sum(Function):
  def forward(self, x, axis,keepdims):
    self.axis = axis
    self.x = x
    self.shape = x.shape
    self.keepdims = keepdims
    return x.sum(axis=axis,keepdims=keepdims)

  def backward(self, grad_out):
    if self.axis: return _expand(grad_out,self.shape,self.axis,self.keepdims)
    else: return np.ones_like(self.x) * grad_out


class Neg(Function):
  def forward(self, x): 
    return -x

  def backward(self, grad_out): 
    return -grad_out


class Exp(Function):
  def forward(self,x):
    self.x = x
    return np.exp(x)
  
  def backward(self,grad_out):
    return np.exp(self.x) * grad_out


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


class Relu(Function):
  def forward(self, x):
    self.x = x
    return np.maximum(x, 0)

  def backward(self, grad_out):
    return np.where(self.x > 0, 1, 0) * grad_out


class Sigmoid(Function):
  def forward(self, x):
    self.ret = np.piecewise(x,[x > 0],[lambda i: 1 / (1 + np.exp(-i)), lambda i: np.exp(i) / (1 + np.exp(i))],)
    return self.ret

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
  