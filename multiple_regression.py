import numpy as np
from linear_regression import LinearRegression


class MultipleRegression(LinearRegression):
    def __repr__(self):
        return f"MultipleRegression(w={self.w},b={self.b})"

    def _grad(self):
        dw, db = np.zeros((self.t_in.shape[1],)), 0.0
        for i in range(self.t_in.shape[0]):
            for j in range(self.t_in.shape[1]):
                dw[j] += (
                    (np.dot(self.t_in[i], self.w) + self.b) - self.t_out[i]
                ) * self.t_in[i, j]
            db += (np.dot(self.t_in[i], self.w) + self.b) - self.t_out[i]
        return dw / self.t_in.shape[0], db / self.t_in.shape[0]
