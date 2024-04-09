import numpy as np
from numpy.typing import ArrayLike


class Tensor:
    def __init__(self, data: ArrayLike, dtype=np.float32, _children=()):
        self._data = data if isinstance(data, np.ndarray) else np.array(data, dtype=dtype)
        self._prev = set(_children)
        self.dtype = self._data.dtype
        self.shape = self._data.shape
        self.ndim = self._data.ndim

    def __repr__(self):
        return f"Tensor({self._data}, dtype={self.dtype})"

    def __getitem__(self, item):
        out = Tensor(self._data[item]) if isinstance(self._data[item], np.ndarray) else self._data[item]
        return out

    def __iter__(self):
        return self._data

    def __add__(self, other):
        assert isinstance(other, Tensor)
        out = self._data + other._data
        return Tensor(out, out.dtype, (self, other))

    def __mul__(self, other):
        assert isinstance(other, Tensor)
        out = self._data * other._data
        return Tensor(out, out.dtype, (self, other))

    def dot(self, other):
        assert isinstance(other, Tensor)
        return np.dot(self._data, other._data)

    def matmul(self, other):
        out = np.matmul(self._data, other._data)
        return Tensor(out, out.dtype, (self, other))

    def log(self):
        out = np.log(self._data)
        return Tensor(out, out.dtype, (self,))

    def relu(self):
        out = self._data * (self._data > 0)
        return Tensor(out, out.dtype, (self,))

    def max(self):
        out = np.max(self._data)
        return Tensor(out, out.dtype)

    def sigmoid(self):
        max_x = np.max(self._data)
        out = np.exp(self._data - max_x) / (1 + np.exp(self._data - max_x))
        return Tensor(out, out.dtype, (self,))
