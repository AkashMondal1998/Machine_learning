from __future__ import annotations
from itertools import pairwise
from typing import TYPE_CHECKING, Generator
import numpy as np
from .linear import Linear

if TYPE_CHECKING:
  from minigrad import Tensor


# Base class that should be extended
class Base:
  def __repr__(self):
    modules = (f"({module_name}): {module}" for module_name, module in vars(self).items())
    return self.__class__.__name__ + "(\n    " + "\n    ".join(modules) + "\n)"

  def __call__(self, x: "Tensor"):
    return self.forward(x)

  def parameters(self) -> Generator["Tensor", None, None]:
    for layer in vars(self).values():
      if isinstance(layer, Linear):
        yield layer._w
        if layer._b is not None:
          yield layer._b

  # do not know if this new implementation works or not for all scenarios
  def zero_grad(self):
    for w, b in pairwise(self.parameters()):
      w.grad = np.zeros_like(w.grad)
      if b is not None:
        b.grad = np.zeros_like(b.grad)
