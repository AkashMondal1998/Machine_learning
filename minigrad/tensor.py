import numpy as np


# A context for the saved Tensors
class Context:
  def __init__(self):
    self.saved_tensors = []

  def saved_for_backward(self, *tensors):
    self.saved_tensors.extend(tensors)


class Tensor:
  def __init__(self, data, _childern=(), requires_grad=False) -> None:
    self._data = np.array(data, dtype=np.float32)
    self._prev = set(_childern)
    self._backward = lambda: None
    self.grad = np.zeros_like(self._data, dtype=np.float32)
    self.shape = self._data.shape
    self.ndim = self._data.ndim
    self._requires_grad = requires_grad
    self.dtype = self._data.dtype

  def __repr__(self) -> str:
    return f"Tensor({self._data},requires_grad={self._requires_grad})"

  def __len__(self):
    return len(self._data)

  def __getitem__(self, item):
    return self._data[item]

  def __pow__(self, exponent):
    out = Tensor(self._data**exponent, (self,))

    def _backward():
      self.grad += exponent * (self._data ** (exponent - 1)) * out.grad

    out._backward = _backward
    return out

  def __add__(self, other):
    if isinstance(other, (int, float)):
      out = Tensor(self._data + other, (self,))
    else:
      out = Tensor(self._data + other._data, (self, other))

    def _backward():
      if isinstance(other, (int, float)):
        self.grad += out.grad.flatten()[: self.grad.size].reshape(self.grad.shape)
      else:
        self.grad += (out.grad.size / self.grad.size) * out.grad.flatten()[: self.grad.size].reshape(self.grad.shape)
        other.grad += (out.grad.size / other.grad.size) * out.grad.flatten()[: other.grad.size].reshape(other.grad.shape)

    out._backward = _backward
    return out

  def __radd__(self, other):
    return self + other

  def __sub__(self, other):
    if isinstance(other, (int, float)):
      out = Tensor(self._data - other, (self,))
    else:
      out = Tensor(self._data - other._data, (self, other))

    def _backward():
      if isinstance(other, (int, float)):
        self.grad += out.grad.flatten()[: self.grad.size].reshape(self.grad.shape)
      else:
        self.grad += (out.grad.size / self.grad.size) * out.grad.flatten()[: self.grad.size].reshape(self.grad.shape)
        other.grad -= (out.grad.size / other.grad.size) * out.grad.flatten()[: other.grad.size].reshape(other.grad.shape)

    out._backward = _backward
    return out

  def __rsub__(self, other):
    return self - other

  def __mul__(self, other):
    assert self.shape == other.shape
    out = Tensor(self._data * other._data, (self, other))

    def _backward():
      self.grad += other._data * out.grad
      out.grad += self._data * out.grad

    out._backward = _backward
    return out

  def __matmul__(self, other):
    assert self._data.shape[1] == other._data.shape[0]
    out = Tensor(self._data @ other._data, (self, other))

    def _backward():
      self.grad += out.grad @ other._data.T
      other.grad += self._data.T @ out.grad

    out._backward = _backward
    return out

  def __neg__(self):
    out = Tensor(-self._data, (self,))

    def _backward():
      self.grad += -out.grad

    out._backward = _backward
    return out

  def exp(self):
    out = Tensor(np.exp(self._data), (self,))

    def _backward():
      self.grad += out._data * out.grad

    out._backward = _backward
    return out

  def log(self):
    out = Tensor(np.log(self._data), (self,))

    def _backward():
      self.grad += (1 / self._data) * out.grad

    out._backward = _backward
    return out

  def square(self):
    out = Tensor(self._data**2, (self,))

    def _backward():
      self.grad += 2.0 * self._data * out.grad

    out._backward = _backward
    return out

  def sum(self):
    out = Tensor(self._data.sum(), (self,))

    def _backward():
      self.grad += out.grad

    out._backward = _backward
    return out

  def mean(self):
    out = Tensor(self._data.mean(), (self,))

    def _backward():
      self.grad += (1.0 / self._data.size) * out.grad

    out._backward = _backward
    return out

  def sigmoid(self):
    def _sigmoid(x):
      return 1 / (1 + np.exp(-x))

    out = Tensor(_sigmoid(self._data), (self,))

    def _backward():
      self.grad += _sigmoid(self._data) * (1 - _sigmoid(self._data)) * out.grad

    out._backward = _backward
    return out

  def maxmimum(self, value):
    out = Tensor(np.maximum(self._data, value), (self,))

    def _backward():
      self.grad += np.where(self._data > value, out.grad, 0)

    out._backward = _backward
    return out

  def neg(self):
    return self.__neg__()

  def relu(self):
    out = Tensor(np.maximum(0, self._data), (self,))

    def _backward():
      self.grad += np.where(self._data > 0, out.grad, 0)

    out._backward = _backward
    return out

  def abs(self):
    out = Tensor(np.abs(self._data), (self,))

    def _backward():
      self.grad += np.where(self._data < 0, -1, np.where(self._data > 0, 1, 0)) * out.grad

    out._backward = _backward
    return out

  def dot(self, other):
    assert isinstance(other, Tensor) and other._data.ndim == 1 and self._data.ndim == 1
    out = Tensor(np.dot(self._data, other._data), (self, other))

    def _backward():
      self.grad += out.grad * other._data
      other.grad += out.grad * self._data

    out._backward = _backward
    return out

  def numpy(self):
    return self._data

  def backward(self):
    topo = []
    visited = set()
    assert self._data.ndim == 0, "Can call backward only on scalar Tensors"

    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)

    build_topo(self)
    self.grad = np.array(1, dtype=np.float32)
    for node in reversed(topo):
      node._backward()

  @classmethod
  def zeros(cls, *shape, dtype=np.float32):
    out = np.zeros(shape, dtype=dtype)
    return cls(out)

  @classmethod
  def zeros_like(cls, a, dtype=None):
    out = np.zeros_like(a._data, dtype)
    return cls(out)

  @classmethod
  def random_uniform(cls, low: float = 0.0, high: float = 1.0, size=None):
    out = np.random.uniform(low, high, size)
    return cls(out)

  @classmethod
  def rand(cls, d1, d2):
    out = np.random.rand(d1, d2)
    return cls(out)

  @classmethod
  def random_sample(cls, *shape):
    out = np.random.random_sample(shape)
    return cls(out)

  @classmethod
  def normal(cls, loc=0.0, scale=1.0, size=None):
    out = np.random.normal(loc, scale, size)
    return cls(out)
