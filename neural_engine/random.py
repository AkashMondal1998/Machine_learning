from .tensor import Tensor
import numpy as np


def rand(d1, d2):
    out = np.random.rand(d1, d2)
    return Tensor(out, out.dtype)
