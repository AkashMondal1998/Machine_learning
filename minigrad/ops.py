from .tensor import Function, register
import numpy as np


class Add(Function):
  @staticmethod
  def forward(ctx, x, y):
    return x + y

  @staticmethod
  def backward(ctx, grad_out):
    return grad_out, grad_out


register("add", Add)


class Mul(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x * y

  @staticmethod
  def backward(ctx, grad_out):
    x, y = ctx.saved_tensors
    return y * grad_out, x * grad_out


register("mul", Mul)


class Neg(Function):
  @staticmethod
  def forward(ctx, x):
    return -x

  @staticmethod
  def backward(ctx, grad_out):
    return -grad_out


register("neg", Neg)


class Sum(Function):
  @staticmethod
  def forward(ctx, x):
    return x.sum()

  @staticmethod
  def backward(ctx, grad_out):
    return grad_out


register("sum", Sum)


class Log(Function):
  @staticmethod
  def forward(ctx, x):
    ctx.save_for_backward(x)
    return np.log(x)

  @staticmethod
  def backward(ctx, grad_out):
    (x,) = ctx.saved_tensors
    return (1 / x) * grad_out


register("log", Log)
