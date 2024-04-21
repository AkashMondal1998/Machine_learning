import numpy as np
from functools import partialmethod


class Function:
  def __init__(self, *tensors):
    self.parents = tensors
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
    ctx = op(*x)
    ret = Tensor(ctx.forward(ctx, *[t.data for t in x]))
    ret._ctx = ctx
    return ret


class Tensor:
  def __init__(self, data):
    self.data = np.array(data, dtype=np.float32)
    self.grad = None
    self._ctx = None

  @property
  def shape(self):
    return self.data.shape

  @property
  def dtype(self):
    return self.data.dtype

  def __repr__(self):
    return f"Tensor({self.data})"

  def __add__(self, x):
    return self.add(x)

  def __mul__(self, x):
    return self.mul(x)

  def __neg__(self):
    return self.neg()

  def __sub__(self, x):
    return self + (-x)

  def mean(self):
    div = Tensor(np.array([1 / self.data.size], dtype=self.data.dtype))
    print(div)
    return self.sum().mul(div)

  def backward(self, auto_fill=True):
    if not self._ctx:
      return

    if not self.grad and auto_fill:
      assert self.data.size == 1
      self.grad = np.ones_like(self.data)

    grads = self._ctx.backward(self._ctx, self.grad)
    if len(self._ctx.parents) == 1:
      grads = [grads]
    for t, g in zip(self._ctx.parents, grads):
      t.grad = g
      t.backward(False)


def register(name, fnx):
  setattr(Tensor, name, partialmethod(fnx.apply, fnx))


# register the tensors ops
import minigrad.ops  # noqa: F401
