from __future__ import annotations
from typing import TYPE_CHECKING, Generator
from .linear import Linear

if TYPE_CHECKING:
  from minigrad import Tensor


# Base class that should be extended
class Base:
  def __repr__(self):
    modules = (f"({module_name}): {module}" for module_name, module in vars(self).items())
    return self.__class__.__name__ + "(\n    " + "\n    ".join(modules) + "\n)"

  def __call__(self, x: Tensor):
    return self.forward(x)

  def parameters(self) -> Generator[Tensor, None, None]:
    for module in vars(self).values():
      if isinstance(module, Linear):
        yield module._w
        if module._b is not None:
          yield module._b