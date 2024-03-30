import torch
from tqdm import tqdm

class Regression:
    def __init__(self,t_in,t_out):
        self.t_in = t_in
        self.t_out = t_out
        self.w = torch.zeros(t_in.shape[1],requires_grad=True)
        self.b = torch.tensor(0.0,requires_grad=True)

    @property
    def _loss(self):
        loss = 0.0
        for i in range(self.t_in.shape[0]):
            loss += torch.square((torch.dot(self.t_in[i],self.w) + self.b) - self.t_out[i])
        return (loss)/(2*self.t_in.shape[0])


    def train(self,epochs,l_rate):
        for i in tqdm(range(epochs),delay=0.25,total=epochs,desc="Training Process"):
            if i % 100 == 0: print(self._loss)
            self._loss.backward()
            with torch.no_grad():
                self.w -= l_rate * self.w.grad
                self.b -= l_rate * self.b.grad
                self.w.grad.zero_()
                self.b.grad.zero_()

    def predict(self,test_data):
        preds = torch.zeros(test_data.shape[0])
        for i in range(test_data.shape[0]):
            preds[i] = torch.dot(test_data[i],self.w) + self.b
        return preds