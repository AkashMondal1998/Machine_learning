import numpy as np


# Base class for adding functionalities
class Base:
    def __repr__(self):
        layers = (f"({layer_name}): {layer}" for layer_name, layer in vars(self).items())
        return self.__class__.__name__ + "(\n    " + "\n    ".join(layers) + "\n)"

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        for layer in vars(self).values():
            if isinstance(layer, Layer):
                for neuron in layer._neurons:
                    yield neuron._w
                    yield neuron._b


# A Neuron
class Neuron:
    def __init__(self, in_features, requires_bias):
        self._w = np.random.rand(in_features, 1)
        if requires_bias:
            self._b = np.random.rand(1, 1)

    def __repr__(self):
        if self._b:
            return f"Neuron(w={self._w.shape},b={self._b.shape})"
        return f"Neuron(w={self._w.shape}"

    def __call__(self, x):
        if not hasattr(self, "_b"):
            return x @ self._w
        return x @ self._w + self._b


# A Layer of Neurons
class Layer:
    def __init__(self, in_features, no_of_neurons, requires_bias=True):
        self._in = in_features
        self._requires_bias = requires_bias
        self._neurons = tuple(Neuron(in_features, requires_bias) for _ in range(no_of_neurons))

    def __repr__(self):
        return f"Layer(in_features = {self._in}, out_features = {len(self._neurons)}, requires_bias={self._requires_bias})"

    @property
    def weight(self):
        return np.concatenate(tuple(neuron._w for neuron in self._neurons), axis=1)

    @property
    def bias(self):
        if self._requires_bias:
            return np.concatenate(tuple(neuron._b for neuron in self._neurons), axis=1)

    def __call__(self, x):
        if x.ndim != 2:
            raise ValueError("Input must be a 2D matrix")
        return np.concatenate(tuple(neuron(x) for neuron in self._neurons), axis=1)
