import numpy as np


class Tensor:
    def __init__(self, data: np.ndarray, _childern=()) -> None:
        self._data = data
        self._prev = set(_childern)
        self._backward = lambda: None
        self.grad = np.zeros_like(self._data, dtype=np.float32)
        self.shape = data.shape
        self.ndim = data.ndim

    def __repr__(self) -> str:
        return f"Tensor({self._data},grad={self.grad})"

    def __getitem__(self, item):
        return self._data[item]

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

    def __pow__(self, exponent):
        out = Tensor(self._data**exponent, (self,))

        def _backward():
            self.grad += exponent * (self._data ** (exponent - 1)) * out.grad

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
            self.grad[:] += 1.0 * out.grad

        out._backward = _backward
        return out

    def __add__(self, other):
        out = Tensor(self._data + other._data, (self, other))

        def _backward():
            self.grad = self.grad + out.grad
            other.grad = other.grad + out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        out = Tensor(self._data - other._data, (self, other))

        def _backward():
            self.grad = self.grad + out.grad
            other.grad = other.grad - out.grad

        out._backward = _backward
        return out

    def mean(self):
        out = Tensor(self._data.mean(), (self,))

        def _backward():
            self.grad[:] += (1.0 / self._data.size) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        def compute_sigmoid(x):
            return 1 / (1 + np.exp(-x))

        out = Tensor(compute_sigmoid(self._data), (self,))

        def _backward():
            self.grad += (
                compute_sigmoid(self._data)
                * (1 - compute_sigmoid(self._data))
                * out.grad
            )

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self._data), (self,))

        def _backward():
            self.grad += np.where(self._data > 0, out.grad, 0)

        out._backward = _backward
        return out

    def dot(self, other):
        assert (
            isinstance(other, Tensor) and other._data.ndim == 1 and self._data.ndim == 1
        )
        out = Tensor(np.dot(self._data, other._data), (self, other))

        def _backward():
            self.grad += out.grad * other._data
            other.grad += out.grad * self._data

        out._backward = _backward
        return out

    def __matmul__(self, other):
        assert self._data.shape[1] == other._data.shape[0]
        out = Tensor(self._data @ other._data, (self, other))

        def _backward():
            self.grad = out.grad @ other._data.T
            other.grad = self._data.T @ out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        assert self._data.ndim == 0

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

    def numpy(self):
        return self._data
