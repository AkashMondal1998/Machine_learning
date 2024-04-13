from neural_engine import Tensor
import numpy as np


def rand(d1, d2):
    out = np.random.rand(d1, d2)
    return Tensor(out, out.dtype)


def random_sample(shape: tuple):
    out = np.random.random_sample(shape)
    return Tensor(out)
