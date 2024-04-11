import numpy as np
from neural_engine import Tensor


# Mean Squared error loss function
class MSELoss:
    def __call__(self, x, y):
        if x.shape != y.shape:
            raise ValueError("mat1 and mat2 be must have same shape")
        return np.mean(np.square(x - y))


# Binary CrossEntropy Loss
class BCELoss:
    def __call__(self, x: Tensor, y: Tensor):
        if x.shape != y.shape:
            raise ValueError("mat1 and mat2 be must have same shape")
        return (-y * x.log()) + (1 - y) * (1 - x).log()


# CrossEntropyLoss
class CrossEntropyLoss:
    @staticmethod
    def _softmax(x: Tensor):
        exp_sum = x.exp().sum(axis=1, keepdims=True)
        return x.exp() / exp_sum

    def __call__(self, x: Tensor, y: Tensor):
        x = self._softmax(x)
        loss = -x[np.arange(len(y)), y].log()
        return loss.mean()
