from typing import Iterable


# Tensor
# Not sure if it is correct
class Tensor:
    def __init__(self, data: Iterable):
        self._data = tuple(data) if isinstance(data, Iterable) else (data,)
        self.shape = (len(self._data),) if isinstance(data, Iterable) else (1,)
        self._prev = set()

    def __repr__(self) -> str:
        rep = f"Tensor({self._data[0]})" if self.shape[0] == 1 else f"Tensor{self._data}"
        return rep

    def __iter__(self) -> Iterable:
        return iter(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def __add__(self, other):
        assert isinstance(other, Tensor)
        assert self.shape[0] == other.shape[0]
        tensor = Tensor((x + y for x, y in zip(self._data, other._data)))
        setattr(tensor, "_prev", set((self, other)))
        return tensor

    def __sub__(self, other):
        assert isinstance(other, Tensor)
        assert self.shape[0] == other.shape[0]
        tensor = Tensor((x - y for x, y in zip(self._data, other._data)))
        setattr(tensor, "_prev", set((self, other)))
        return tensor

    def __mul__(self, other):
        assert isinstance(other, Tensor)
        assert self.shape[0] == other.shape[0]
        tensor = Tensor((x * y for x, y in zip(self._data, other._data)))
        setattr(tensor, "_prev", set((self, other)))
        return tensor

    def dot(self, other):
        assert isinstance(other, Tensor)
        assert self.shape[0] == other.shape[0]
        tensor = Tensor(sum(x * y for x, y in zip(self._data, other._data)))
        setattr(tensor, "_prev", set((self, other)))
        return tensor

    def value(self):
        assert self.shape[0] == 1, "Can only done with a Tensor of shape (1,)"
        return self._data[0]
