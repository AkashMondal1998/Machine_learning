import numpy as np
from functools import partialmethod


class Tensor:
  def __init__(self, data):
    self.data = np.array(data)
    self.grad = None

  def __repr__(self):
    return f"Tensor({self.data})"

  def __add__(self, x):
    return self.add(x)

  def __neg__(self):
    return self.neg()

  def __sub__(self, x):
    return self.sub(x)

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

  def apply(self, arg, *tensor):
    if type(self) == Tensor:
      op = arg
      x = [self] + list(tensor)
    else:
      op = self
      x = [arg] + list(tensor)
    ctx = op()
    ret = Tensor(ctx.forward(ctx, x))
    ret._ctx = ctx
    return ret


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


class Neg(Function):
  @staticmethod
  def forward(ctx, input):
    (x,) = input
    ctx.save_for_backward(x)
    return -x.data

  @staticmethod
  def backward(ctx, grad_out):
    return -1 * grad_out


register("neg", Neg)
