import numpy as np
import math


class Function:
  def __init__(self, *tensors):
    self.requires_grad = any(t.requires_grad for t in tensors)
    if self.requires_grad: self.parents = tensors

  def forward(self, *args, **kwargs): raise NotImplementedError()
  def backward(self, *args, **kwargs): raise NotImplementedError()

  @classmethod
  def apply(op, *x, **kwargs):
    ctx = op(*x)
    ret = Tensor(ctx.forward(*[t.data for t in x], **kwargs), requires_grad=ctx.requires_grad)
    if ctx.requires_grad: ret._ctx = ctx
    return ret


import minigrad.function as F  # noqa: F402


class Tensor:
  __slots__ = "data", "grad", "_ctx", "requires_grad"

  def __init__(self, data, dtype=None, requires_grad=False):
    self.data = np.array(data, dtype=dtype) if not isinstance(data, np.ndarray) else data
    self.grad = None
    self._ctx = None
    self.requires_grad = requires_grad

  @property
  def shape(self): return self.data.shape

  @property
  def dtype(self): return self.data.dtype

  @property
  def ndim(self): return self.data.ndim

  @property
  def size(self): return self.data.size

  def __repr__(self): return f"Tensor({self.data},requires_grad={self.requires_grad})"

  def __len__(self): return len(self.data)

  def __hash__(self): return id(self)

  def add(self, x, reverse=False): return F.Add.apply(*self._const(x, reverse))

  def sub(self, x, reverse=False): return F.Sub.apply(*self._const(x, reverse))

  def div(self, x, reverse=False): return F.Div.apply(*self._const(x, reverse))

  def mul(self, x, reverse=False): return F.Mul.apply(*self._const(x, reverse))

  def pow(self, x, reverse=False): return F.Pow.apply(*self._const(x, reverse))

  def maximum(self, x, reverse=False): return F.Maximum.apply(*self._const(x, reverse))

  def reshape(self, shape=None): return F.Reshape.apply(self, shape=shape)

  def sum(self, axis=None, keepdims=False): return F.Sum.apply(self, axis=axis, keepdims=keepdims)

  def max(self, axis=None, keepdims=False): return F.Max.apply(self, axis=axis, keepdims=keepdims)

  def argmax(self, axis=None, keepdims=False): return Tensor(np.argmax(self.data, axis=axis, keepdims=keepdims))

  def mean(self, axis=None, keempdims=False):
    if axis == 1: return self.sum(axis=axis, keepdims=keempdims).div(self.shape[1])
    elif axis == 0: return self.sum(axis=axis, keepdims=keempdims).div(self.shape[0])
    else: return self.sum().div(self.size)

  def neg(self): return F.Neg.apply(self)

  def log(self): return F.Log.apply(self)

  def exp(self): return F.Exp.apply(self)

  def flatten(self): return F.Flatten.apply(self)

  def sqrt(self): return self.pow(0.5)

  def dot(self, x):
    if self.ndim == x.ndim == 1: assert self.shape[0] == x.shape[0], f"1D tensors are not aligned {self.shape[0]} (dim0) != {x.shape[0]} (dim0)"
    else: assert self.shape[1] == x.shape[0], f"2D tensors are not aligned {self.shape[1]} (dim1) != {x.shape[0]} (dim0)"
    return F.Dot.apply(self, x)

  def matmul(self, x):
    assert self.ndim == x.ndim == 2, f"2D tensors are expected"
    return self.dot(x)

  def relu(self): return F.Relu.apply(self)

  def sigmoid(self): return F.Sigmoid.apply(self)

  def _softmax(self, axis):
    m = self - self.max(axis=axis, keepdims=True)
    e = m.exp()
    e_sum = e.sum(axis=axis, keepdims=True)
    return m, e, e_sum

  def softmax(self, axis=-1):
    _, e, e_sum = self._softmax(axis=axis)
    return e.div(e_sum)

  def log_softmax(self, axis=-1):
    m, _, e_sum = self._softmax(axis=axis)
    return m - e_sum.log()

  def square(self): return self * self

  def abs(self): return self.relu() + self.neg().relu()

  def mean_squared_error(self, x): return (self.sub(x)).square().mean()

  def binary_crossentropy(self, x):
    return ((-x * self.log()) - ((1 - x) * (1 - self).log())).mean()

  def binary_crossentropy_withlogits(self, x):
    return (self.maximum(0) - (self * x) + (1 + self.abs().neg().exp()).log()).mean()

  def sparse_categorical_crossentropy(self, y):
    # self is the raw logits
    self = self.log_softmax()
    loss = self[Tensor.arange(len(y)), y].neg()
    return loss.mean()

  def item(self):
    assert self.shape == tuple(), "Only scalar tensors are allowed"
    return self.data.item()

  def _eq(x, y):
    return x.data == y.data

  def eq(self, x):
    t = Tensor._eq(*self._const(x, reverse=False))
    return Tensor(t)

  @classmethod
  def zeros(cls, *shape, requires_grad=False): return cls(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)

  @classmethod
  def arange(cls, start, stop=0, step=1, dtype=None, requires_grad=False):
    if stop == 0: start, stop = stop, start
    return cls(np.arange(start, stop, step, dtype=dtype), requires_grad=requires_grad)

  @classmethod
  def eye(cls, n, m=None, dtype=None, requires_grad=False): return cls(np.eye(n, m, dtype=dtype), requires_grad=requires_grad)

  @classmethod
  def randint(cls, low, high=None, size=None, dtype=int, requires_grad=False):
    return cls(np.random.randint(low, high, size, dtype), requires_grad=requires_grad)

  @classmethod
  def xavier_uniform(cls, in_features, out_features, requires_grad=False):
    range = math.sqrt(6 / (in_features + out_features))
    return cls(np.random.uniform(-range, +range, (in_features, out_features)), requires_grad=requires_grad)

  @classmethod
  def xavier_normal(cls, in_features, out_features, requires_grad=False):
    scale = math.sqrt(2 / (in_features + out_features))
    return cls(np.random.normal(0.0, scale, (in_features, out_features)), requires_grad=requires_grad)

  def build_topo(self):
    def _build_topo(t, visited):
      if t not in visited:
        visited.add(t)
        if t._ctx:
          for p in t._ctx.parents: yield from _build_topo(p, visited)
          yield t
    return list(_build_topo(self, set()))

  def backward(self):
    assert self.data.shape == tuple(), "Backward can only be called on the scalar tensors"
    self.grad = np.array(1.0)

    for t0 in reversed(self.build_topo()):
      grads = t0._ctx.backward(t0.grad)
      grads = [grads] if len(t0._ctx.parents) == 1 else grads

      for t, g in zip(t0._ctx.parents, grads):
        if t.requires_grad:
          assert t.shape == g.shape, f"{t.shape} != {g.shape}"
          t.grad = g if t.grad is None else (t.grad + g)
    return self

  def _const(self, val, reverse):
    val = Tensor(np.full_like(self.data, val), requires_grad=False) if isinstance(val, (int, float)) else val
    if reverse: return val, self
    else: return self, val

  def __getitem__(self, indices):
    if isinstance(indices, tuple) and isinstance(indices[0], Tensor):
      indices = tuple(t.data for t in indices)
    if isinstance(indices, Tensor): indices = indices.data
    return F.Slice.apply(self, indices=indices)

  def __neg__(self): return self.neg()
  def __add__(self, x): return self.add(x)
  def __sub__(self, x): return self.sub(x)
  def __mul__(self, x): return self.mul(x)
  def __pow__(self, x): return self.pow(x)
  def __eq__(self, x): return self.eq(x)
  def __matmul__(self, x): return self.dot(x)
  def __truediv__(self, x): return self.div(x)

  def __radd__(self, x): return self.add(x, True)
  def __rsub__(self, x): return self.sub(x, True)
  def __rmul__(self, x): return self.mul(x, True)
  def __rpow__(self, x): return self.pow(x, True)
  def __rtruediv__(self, x): return self.div(x, True)
