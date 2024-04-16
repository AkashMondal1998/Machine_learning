import numpy as np

from minigrad import Tensor


# linear transformation
class Linear:
  def __init__(self, in_features: int, out_features: int, bias=True) -> None:
    self._w = Tensor.normal(0.0, np.sqrt(2 / (in_features + out_features)), (in_features, out_features))
    self._b = Tensor.zeros(1, out_features) if bias else None

  def __repr__(self):
    return f"Linear(in_features = {self._w.shape[0]}, out_features = {(self._w.shape[1])}, requires_bias={True if self._b is not None else False})"

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
