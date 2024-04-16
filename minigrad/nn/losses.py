from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from minigrad import Tensor


# Mean Squared error loss function
class MSELoss:
  def __call__(self, x: Tensor, y: Tensor):
    if x.shape != y.shape:
      raise ValueError("mat1 and mat2 be must have same shape")
    return (x - y).square().mean()


# Binary CrossEntropy Loss
class BCELoss:
  def __call__(self, x: Tensor, y: Tensor):
    if x.shape != y.shape:
      raise ValueError("mat1 and mat2 be must have same shape")
    return (-y * x.log() - (1 - y) * (1 - x).log()).mean()


"""# CrossEntropyLoss
class CrossEntropyLoss:
    @staticmethod
    def _softmax(x: Tensor):
        exp_sum = x.exp().sum(axis=1, keepdims=True)
        return x.exp() / exp_sum

    def __call__(self, x: Tensor, y: Tensor):
        x = self._softmax(x)
        loss = -x[np.arange(len(y)), y].log()
        return loss.mean()"""
