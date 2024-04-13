import numpy as np

np.set_printoptions(4)


class Tensor:
    def __init__(self, data: np.ndarray, _childern=()) -> None:
        self._data = data
        self._prev = set(_childern)
        self._backward = lambda: None
        self.grad = np.zeros_like(self._data, dtype=np.float32)

    def __repr__(self) -> str:
        return f"Tensor({self._data},grad={self.grad})"

    def __getitem__(self, item):
        return self._data[item]

    def sum(self):
        out = Tensor(self._data.sum(), (self,))

        def _backward():
            self.grad[:] = 1.0 * out.grad

        out._backward = _backward
        return out

    def mean(self):
        out = Tensor(self._data.mean(), (self,))

        def _backward():
            self.grad[:] = (1.0 / self._data.size) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self._data), (self,))

        def _backward():
            # if for a self.data item it is greater than zero then
            # just copy the out.grad value to self.grad or if the self.data item is 0 or less than
            # set the self.grad to 0
            self.grad = np.where(self._data > 0, out.grad, 0)

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
