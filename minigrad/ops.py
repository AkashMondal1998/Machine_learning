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


class Dot(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x.dot(y)

  @staticmethod
  def backward(ctx, grad_out):
    x, y = ctx.saved_tensors
    x_grad = grad_out.dot(y.T)
    y_grad = x.T.dot(grad_out)
    return x_grad, y_grad


register("dot", Dot)
register("matmul", Dot)


class Square(Function):
  @staticmethod
  def forward(ctx, x):
    ctx.save_for_backward(x)
    return np.square(x)

  @staticmethod
  def backward(ctx, grad_out):
    (x,) = ctx.saved_tensors
    return 2 * x * grad_out


register("square", Square)


class Mean(Function):
  @staticmethod
  def forward(ctx, x):
    ctx.save_for_backward(x)
    return np.mean(x)

  @staticmethod
  def backward(ctx, grad_out):
    (x,) = ctx.saved_tensors
    return (1 / x.size) * grad_out


register("mean", Mean)


class Relu(Function):
  @staticmethod
  def forward(ctx, x):
    ctx.save_for_backward(x)
    return np.maximum(x, 0)

  @staticmethod
  def backward(ctx, grad_out):
    (x,) = ctx.saved_tensors
    return np.where(x > 0, x, 0)


register("relu", Relu)


class Sigmoid(Function):
  @staticmethod
  def forward(ctx, x):
    ret = 1 / (1 + np.exp(-x))
    ctx.save_for_backward(ret)
    return ret

  @staticmethod
  def backward(ctx, grad_out):
    (ret,) = ctx.saved_tensors
    return ret * (1 - ret) * grad_out


register("sigmoid", Sigmoid)


class LogSoftMax(Function):
  @staticmethod
  def forward(ctx, x):
    def logsumexp(x):
      c = x.max(axis=1)
      return c + np.log(np.exp(x - c.reshape(-1, 1)).sum(axis=1, keepdims=True))

    ret = x - logsumexp(x)
    ctx.save_for_backward(ret)
    return ret

  @staticmethod
  def backward(ctx, grad_out):
    (ret,) = ctx.saved_tensors
    return grad_out - np.exp(ret) * grad_out.sum(axis=1, keepdims=True)


register("logsoftmax", LogSoftMax)
