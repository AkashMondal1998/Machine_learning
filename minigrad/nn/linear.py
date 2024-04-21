import numpy as np

from minigrad import Tensor


# linear transformation
class Linear:
  def __init__(self, in_features: int, out_features: int) -> None:
    self._w = Tensor.normal(0.0, np.sqrt(2 / (in_features + out_features)), (in_features, out_features))

  def __repr__(self):
    return f"Linear(in_features = {self._w.shape[0]}, out_features = {(self._w.shape[1])})"

  @property
  def weight(self):
    return self._w

  def __call__(self, x: Tensor):
    assert x.ndim == 2
    return x.dot(self._w)
