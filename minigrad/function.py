from .tensor import Function
from .helpers import expand,reduce
import numpy as np


class Add(Function):
  def forward(self, x, y):
    self.x_shape = x.shape
    self.y_shape = y.shape
    return x + y

  def backward(self, grad_out):
    return reduce(grad_out,self.x_shape), reduce(grad_out,self.y_shape)

class Sub(Function):
  def forward(self,x,y):
    self.x_shape = x.shape
    self.y_shape = y.shape
    return x - y
  
  def backward(self,grad_out): 
    return reduce(grad_out,self.x_shape), reduce(-grad_out,self.y_shape)

class Mul(Function):
  def forward(self, x, y):
    self.x ,self.y = x, y
    return x * y

  def backward(self, grad_out): 
    return reduce(self.y * grad_out,self.x.shape), reduce(self.x * grad_out,self.y.shape)
  

class Div(Function):
  def forward(self, x, y):
    self.x, self.y = x, y
    return x / y

  def backward(self, grad_out):
    grad_x = self.y**-1
    grad_y = self.x * -self.y**-2
    return reduce(grad_x * grad_out,self.x.shape), reduce(grad_y * grad_out,self.y.shape)

class Pow(Function):
  def forward(self,x,y):
    self.x, self.y = x, y
    return x ** y
  
  def backward(self,grad_out):
    return reduce(self.y * (self.x**(self.y - 1)) * grad_out,self.x.shape), \
           reduce((self.x**self.y)* np.log(self.x) * grad_out,self.y.shape)

class Dot(Function):
  def forward(self, x, y):
    self.x, self.y = x, y
    return x.dot(y)

  def backward(self, grad_out):
    x_grad = np.dot(grad_out,self.y.T)
    y_grad = np.dot(self.x.T,grad_out)
    return x_grad, y_grad


class Maximum(Function):
  def forward(self,x,y):
    self.x, self.y = x, y
    return np.maximum(x,y)
  
  def backward(self,grad_out):
    grad_x = np.where(self.x > self.y,1,np.where(self.x < self.y,0,0.5))
    grad_y = np.where(self.y > self.x,1,np.where(self.y < self.x,0,0.5))
    return reduce(grad_x * grad_out,self.x.shape) ,reduce(grad_y * grad_out,self.y.shape)


class Max(Function):
  def forward(self,x,axis,keepdims):
    self.axis = axis
    self.x = x
    self.ret = np.max(x,axis=axis,keepdims=keepdims)
    return self.ret

  def backward(self,grad_out):
    max_1s = (self.x == expand(self.ret,self.axis,self.x.shape)).astype(np.float32)
    num_1s = max_1s.sum(axis=self.axis,keepdims=True)
    return (max_1s / num_1s) * expand(grad_out,self.axis,self.x.shape)
    

class Sum(Function):
  def forward(self, x, axis,keepdims):
    self.axis = axis
    self.x = x
    return x.sum(axis=axis,keepdims=keepdims)

  def backward(self, grad_out):
    ret = expand(grad_out,self.axis,self.x.shape) if self.axis else np.ones_like(self.x) * grad_out
    return ret


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


class Slice(Function):
  def forward(self,x,indices):
    self.x = x
    self.indices = indices
    return x[indices]
  
  def backward(self,grad_out):
    grad_x = np.zeros_like(self.x)
    grad_x[self.indices] = 1.0
    return grad_x * expand(grad_out,1,self.x.shape)
