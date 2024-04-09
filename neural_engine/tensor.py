import numpy as np
from numpy.typing import NDArray


# wraps a numpy array
class Tensor:
    def __init__(self, data: NDArray, _children=()):
        assert isinstance(data, (np.ndarray))
        self.data = data
        self._prev = set(_children)
        self.dtype = data.dtype
        self.shape = data.shape

    def __repr__(self):
        return f"Tensor({self.data})"

    def __getitem__(self, item):
        out = Tensor(self.data[item]) if isinstance(self.data[item], np.ndarray) else self.data[item]
        return out

    def __iter__(self):
        return self.data
