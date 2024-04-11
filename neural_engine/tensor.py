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

    def __repr__(self):
        return f"Tensor({self._data}, dtype={self.dtype})"

    """def __getitem__(self, item):
        return self._data.__getitem__(item)"""

    def __iter__(self):
        return self._data

    def __index__(self):
        return self._data.__index__()

    def __len__(self):
        return len(self._data)

    def __add__(self, other):
        other = Tensor(other) if not isinstance(other, Tensor) else other
        out = self._data + other._data
        return Tensor(out, out.dtype, (self, other))

    def __sub__(self, other):
        other = Tensor(other) if not isinstance(other, Tensor) else other
        out = self._data - other._data
        return Tensor(out, out.dtype, (self, other))

    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):
        assert isinstance(other, Tensor)
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
        assert isinstance(other, Tensor)
        out = self._data / other._data
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
