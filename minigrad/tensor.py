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

  @property
  def ndim(self):
    return self.data.ndim

  def __repr__(self):
    return f"Tensor({self.data})"

  def __getitem__(self, key):
    return Tensor(self.data[key])

  def __add__(self, x):
    return self.add(x)

  def __mul__(self, x):
    return self.mul(x)

  def __neg__(self):
    return self.neg()

  def __sub__(self, x):
    return self + (-x)

  def __matmul__(self, x):
    return self.dot(x)

  def mean_squared_error(self, x):
    return (self - x).square().mean()

  @classmethod
  def zeros(cls, *shape):
    return cls(np.zeros(shape, dtype=np.float32))

  @classmethod
  def eye(cls, n, m=None):
    return cls(np.eye(n, m, dtype=np.float32))

  @classmethod
  def xavier_uniform(cls, in_features, out_features):
    range = np.sqrt(6 / (in_features + out_features))
    return cls(np.random.uniform(-range, +range, (in_features, out_features)))

  @classmethod
  def xavier_normal(cls, in_features, out_features):
    scale = np.sqrt(2 / (in_features + out_features))
    return cls(np.random.normal(0.0, scale, (in_features, out_features)))

  @classmethod
  def from_numpy(cls, array):
    return cls(array.astype(float))

  def backward(self, auto_fill=True):
    if self._ctx is None:
      return

    if self.grad is None and auto_fill:
      assert self.data.size == 1
      self.grad = np.ones_like(self.data)

    grads = self._ctx.backward(self._ctx, self.grad)
    if len(self._ctx.parents) == 1:
      grads = [grads]
    for t, g in zip(self._ctx.parents, grads):
      t.grad = g if t.grad is None else (t.grad + g)
      t.backward(False)


def register(name, fnx):
  setattr(Tensor, name, partialmethod(fnx.apply, fnx))


# register the tensors ops
import minigrad.ops  # noqa: F401
