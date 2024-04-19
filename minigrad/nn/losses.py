from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from minigrad import Tensor


class MSELoss:
  """Mean Squared error loss function"""

  def __call__(self, x: Tensor, y: Tensor):
    if x.shape != y.shape:
      raise ValueError("mat1 and mat2 be must have same shape")
    return (x - y).square().mean()


class BCEWithLogitsLoss:
  """Binary CrossEntropy Loss with logits"""

  def __call__(self, x: Tensor, y: Tensor):
    if x.shape != y.shape:
      raise ValueError("mat1 and mat2 be must have same shape")
    return (x.maxmimum(0) - (x * y) + (1 + x.abs().neg().exp()).log()).mean()


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
