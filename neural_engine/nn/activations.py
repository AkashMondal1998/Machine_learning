import numpy as np

# ReLU Activation
class ReLU:
    def __call__(self, x):
        return np.where(x > 0, x, 0)

    def __repr__(self):
        return f"ReLU()"


# Sigmoid Activation
class Sigmoid:
    def __call__(self, x):
        max_x = np.max(x)
        return np.exp(x - max_x) / (1 + np.exp(x - max_x))

    def __repr__(self):
        return f"Sigmoid()"