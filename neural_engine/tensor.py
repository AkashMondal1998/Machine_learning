import numpy as np
from numpy.typing import ArrayLike


class Tensor:
    def __init__(self, data: ArrayLike, dtype=np.float32, _children=()):
        if isinstance(data, list):
            data = np.array(data, dtype=dtype)
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype)
        self._data = data
        self._prev = set(_children)
        self.dtype = self._data.dtype
        self.shape = self._data.shape
        self.ndim = self._data.ndim
        self._backward = lambda: None
        self.grad = np.zeros_like(self._data)

    def __repr__(self):
        return f"Tensor({self._data},grad={self.grad})"

    def __iter__(self):
        return self._data

    def __index__(self):
        return self._data.__index__()

    def __len__(self):
        return len(self._data)

    def __add__(self, other):
        other = Tensor(other) if not isinstance(other, Tensor) else other
        out = self._data + other._data
        out = Tensor(out, out.dtype, (self, other))

        def _backward():
            self.grad += np.eye(*self.shape) * out.grad
            other.grad += np.eye(*other.shape) * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = Tensor(other) if not isinstance(other, Tensor) else other
        out = self._data - other._data
        return Tensor(out, out.dtype, (self, other))

    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):
        other = Tensor(other) if not isinstance(other, Tensor) else other
        out = self._data * other._data
        return Tensor(out, out.dtype, (self, other))

    def __rmul__(self, other):
        assert isinstance(other, Tensor)
        return self * other

    def __neg__(self):
        out = -1 * self._data
        return Tensor(out, out.data, (self,))

    def __truediv__(self, other):
        other = Tensor(other) if not isinstance(other, Tensor) else other
        out = self._data / other._data
        return Tensor(out, out.dtype, (self, other))

    def dot(self, other):
        assert isinstance(other, Tensor)
        if self.ndim > 1 and other.ndim > 1:
            out = np.dot(self._data, other._data)
            out = Tensor(out, out.dtype, (self, other))

            def _backward():
                self.grad += other._data.T * out.grad
                other.grad += self._data.T * out.grad

            out._backward = _backward

            return out

        return np.dot(self._data, other._data)

    def matmul(self, other):
        return self.dot(other)

    def log(self):
        out = np.log(self._data)
        return Tensor(out, out.dtype, (self,))

    def relu(self):
        out = np.maximum(self._data, 0)
        return Tensor(out, out.dtype, (self,))

    def max(self):
        out = np.max(self._data)
        return Tensor(out, out.dtype, (self,))

    def sigmoid(self):
        max_x = np.max(self._data)
        out = np.exp(self._data - max_x) / (1 + np.exp(self._data - max_x))
        return Tensor(out, out.dtype, (self,))

    def numpy(self):
        return self._data

    def exp(self):
        out = np.exp(self._data)
        return Tensor(out, out.dtype, (self,))

    def mean(self):
        out = self._data.mean()
        return Tensor(out, out.dtype, (self,))

    def sum(self, axis=None, keepdims=False):
        out = np.sum(self._data, axis=axis, keepdims=keepdims)
        return Tensor(out, out.dtype, (self,))

    def flatten(self):
        out = self._data.flatten()
        return Tensor(out, out.dtype, (self,))

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        assert self._prev != set(), "cannot call backward on a leaf node"
        build_topo(self)
        self.grad = np.array(1.0)
        for node in reversed(topo):
            node._backward()
