from itertools import pairwise
from typing import Generator

import numpy as np

from neural_engine import Tensor


# Base class for adding functionalities
class Base:
  def __repr__(self):
    layers = (f"({layer_name}): {layer}" for layer_name, layer in vars(self).items())
    return self.__class__.__name__ + "(\n    " + "\n    ".join(layers) + "\n)"

  def __call__(self, x: Tensor):
    return self.forward(x)

  def parameters(self) -> Generator[Tensor, None, None]:
    for layer in vars(self).values():
      if isinstance(layer, Layer):
        yield layer._w
        if layer._b is not None:
          yield layer._b

  # do not know if this new implementation works or not for all scenarios
  def zero_grad(self):
    """for layer in vars(self).values():
    if isinstance(layer, Layer):
        layer._w.grad = np.zeros_like(layer._w.grad)
        layer._b.grad = np.zeros_like(layer._b.grad)"""

    for w, b in pairwise(self.parameters()):
      w.grad = np.zeros_like(w.grad)
      b.grad = np.zeros_like(b.grad)


# Layer for linear transformation
class Layer:
  def __init__(self, in_features: int, no_of_neurons: int, bias=True) -> None:
    self._w = Tensor.normal(0.0, np.sqrt(2 / (in_features + no_of_neurons)), (in_features, no_of_neurons))
    self._b = Tensor.zeros(1, no_of_neurons) if bias else None

  def __repr__(self):
    return f"Layer(in_features = {self._w.shape[0]}, out_features = {(self._w.shape[1])}, requires_bias={True if self._b is not None else False})"

  @property
  def weight(self):
    return self._w

  @property
  def bias(self):
    return self._b

  def __call__(self, x: Tensor):
    if x.ndim != 2:
      raise ValueError("Input must be a 2D matrix")
    if self._b is None:
      return x @ self._w
    else:
      return x @ self._w + self._b
