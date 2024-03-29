import numpy as np
from tqdm import tqdm


class LinearRegression():
    def __init__(self, t_in, t_out, w, b):
        self.t_in = t_in
        self.t_out = t_out
        self.w = w
        self.b = b

    def __repr__(self):
        return f"LinearRegression(w={self.w},b={self.b})"

    def _cost(self):
        cost = 0.0
        for i in range(self.t_in.shape[0]):
            cost += ((np.dot(self.t_in[i], self.w) +
                     self.b) - self.t_out[i]) ** 2
        return cost / (2 * self.t_in.shape[0])

    def _grad(self):
        dw, db = 0.0, 0.0
        for i in range(self.t_in.shape[0]):
            dw += ((np.dot(self.t_in[i], self.w) + self.b) - self.t_out[i]) * self.t_in[
                i
            ]
            db += (np.dot(self.t_in[i], self.w) + self.b) - self.t_out[i]
        return dw / self.t_in.shape[0], db / self.t_in.shape[0]

    def train(self, epochs=100, l_rate=0.01):
        for i in tqdm(range(epochs), delay=0.1, desc="Traning Progress"):
            dw, db = self._grad()
            self.w -= l_rate * dw
            self.b -= l_rate * db
