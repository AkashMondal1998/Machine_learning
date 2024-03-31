import torch
from tqdm import tqdm


class MultipleLinearRegression:
    def __init__(self, t_in, t_out):
        self._in = t_in
        self._out = t_out
        self.w = torch.zeros(t_in.shape[1], requires_grad=True)
        self.b = torch.tensor(0.0, requires_grad=True)

    def __repr__(self) -> str:
        return f"MultipleLinearRegression(weights={self.w},bias={self.b})"

    @property
    def _loss(self):
        loss = 0.0
        for i in range(self._in.shape[0]):
            loss += torch.square(
                (torch.dot(self._in[i], self.w) + self.b) - self._out[i]
            )
        return loss / (2 * self._in.shape[0])

    def train(self, epochs, l_rate):
        for _ in tqdm(range(epochs), delay=0.25, total=epochs, desc="Training Process"):
            self._loss.backward()
            with torch.no_grad():
                self.w -= l_rate * self.w.grad
                self.b -= l_rate * self.b.grad
                self.w.grad.zero_()
                self.b.grad.zero_()

    def predict(self, x, y):
        for i in range(x.shape[0]):
            pred = torch.dot(x[i], self.w) + self.b
            print(
                f"Predicted {pred:0.2f}    Expected {y[i]}  Diff {torch.abs(pred - y[i])}"
            )

    @staticmethod
    def normalize(x):
        std = torch.min(x, dim=0).values
        mean = torch.mean(x, dim=0)
        x_norm = (x - mean) / std
        return x_norm, mean, std
