import numpy as np
from functools import partialmethod


class Tensor:
  def __init__(self, data):
    self.data = np.array(data)
    self.grad = None

  def __repr__(self):
    return f"Tensor({self.data})"

  def backward(self):
    if not hasattr(self, "_ctx"):
      return
    self.grad = np.ones_like(self.data)
    ctx = self._ctx
    grads = ctx.backward(ctx, self.grad)
    for tensor, grad in zip(ctx.saved_tensors, grads):
      tensor.grad = grad
      tensor.backward()


class Function:
  def __init__(self):
    self.saved_tensors = []

  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)

  # for supporting unary ops have to do something
  def apply(self, arg, tensor):
    if type(self) == Tensor:
      op = arg
      x = [self, tensor]
    else:
      op = self
      x = [arg, tensor]
    ctx = op()
    out = Tensor(ctx.forward(ctx, x))
    out._ctx = ctx
    return out


def register(name, fnx):
  setattr(Tensor, name, partialmethod(fnx.apply, fnx))


class Add(Function):
  @staticmethod
  def forward(ctx, inputs):
    x, y = inputs
    ctx.save_for_backward(x, y)
    return x.data + y.data

  @staticmethod
  def backward(ctx, grad_out):
    return grad_out, grad_out


register("add", Add)


class Mul(Function):
  @staticmethod
  def forward(ctx, inputs):
    x, y = inputs
    ctx.save_for_backward(x, y)
    return x.data * y.data

  @staticmethod
  def backward(ctx, grad_out):
    x, y = ctx.saved_tensors
    return y.data * grad_out, x.data * grad_out


register("mul", Mul)


class Sub(Function):
  @staticmethod
  def forward(ctx, inputs):
    x, y = inputs
    ctx.save_for_backward(x, y)
    return x.data - y.data

  @staticmethod
  def backward(ctx, grad_out):
    return grad_out, -grad_out


register("sub", Sub)
