from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from .tensor import Tensor


def _reduction(x: Tensor, y: Tensor):
  red_axis = None
  for axis, (sizeA, sizeB) in enumerate(zip(x.shape, y.shape)):
    if sizeA != sizeB:
      red_axis = axis
  return red_axis
