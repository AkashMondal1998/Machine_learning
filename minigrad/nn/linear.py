from minigrad import Tensor


# fully connected layer
class Linear:
  def __init__(self, in_features: int, out_features: int) -> None:
    self._w = Tensor.xavier_uniform(in_features, out_features)

  def __repr__(self):
    return f"Linear(in_features = {self._w.shape[0]}, out_features = {(self._w.shape[1])})"

  @property
  def weight(self):
    return self._w

  @weight.setter
  def weight(self, weight):
    self._w = weight

  def __call__(self, x: Tensor):
    assert x.ndim == 2, "Tensor has to be 2D"
    return x.dot(self._w)
