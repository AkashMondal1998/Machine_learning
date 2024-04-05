import numpy as np


# Base class
class Base:
    def __call__(self, x):
        return self.forward(x)


# Neuron
class Neuron:
    def __init__(self, in_features):
        self._w = np.random.rand(in_features, 1)
        self._b = np.random.rand(1, 1)

    def __repr__(self):
        return f"Neuron(w={self._w.shape},b={self._b.shape})"

    def __call__(self, x):
        return x @ self._w + self._b


# Layer
class Layer:
    def __init__(self, in_features, no_of_neurons):
        self._in = in_features
        self._neurons = [Neuron(in_features) for _ in range(no_of_neurons)]

    def __call__(self, x):
        acts = []
        for neuron in self._neurons:
            acts.append(neuron(x))
        return np.concatenate(acts, axis=1)


# Sigmoid activation function
class Sigmoid:
    def __call__(self, x):
        max_x = np.max(x)
        return np.exp(x - max_x) / (1 + np.exp(x - max_x))
