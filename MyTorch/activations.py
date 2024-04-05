import numpy as np


# ReLU activation
class ReLU:
    def __call__(self, x):
        return np.where(x > 0, x, 0)


# sigmoid activation
class Sigmoid:
    def __call__(self, x):
        max_x = np.max(x)
        return np.exp(x - max_x) / (1 + np.exp(x - max_x))
