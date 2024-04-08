import numpy as np


# Mean Squared error loss function
class MSELoss:
    def __call__(self, x, y):
        if x.shape != y.shape:
            raise ValueError("mat1 and mat2 be must have same shape")
        return np.mean(np.square(x - y))


# Binary CrossEntropy Loss
# handle numerical instabality
class BCELoss:
    def __call__(self, x, y):
        if x.shape != y.shape:
            raise ValueError("mat1 and mat2 be must have same shape")
        return np.mean(-y * np.log(x) + (1 - y) * np.log(1 - x))


# CrossEntropyLoss
class CrossEntropyLoss:
    @staticmethod
    def _softmax(x):
        """
        x is an m x n matrix containing the raw logits from the neural network
        Each row is the raw logits from a particular traning sample
        so we have to apply softmax activation to each row in this m x n matrix
        """
        max = np.max(x, axis=1, keepdims=True)
        x_exp = np.exp(x - max)
        x_exp_sum = np.sum(x_exp, axis=1, keepdims=True)
        return x_exp / x_exp_sum

    def __call__(self, x, y):
        """
        Here we expect the raw logits from the neural network
        Then apply the softmax activation on it
        Now we have the probabilities of the models
        """
        y = y.flatten()
        x = self._softmax(x)  # apply the softmax activation to the model output
        prob = x[np.arange(len(y)), y]
        return -np.log(prob + 1e-9)
