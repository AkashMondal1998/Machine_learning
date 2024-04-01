import torch.nn as nn
import torch
from torch.optim import Adam

class LinearRegression(nn.Module):
    def __init__(self,in_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features,1)

    def forward(self,x):
        a1 = self.layer1(x)
        return a1


def train(epochs,x,y,model):
    optimizer = Adam(model.parameters())
    l_loss = 0
    criterion = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x),y.reshape(-1,1))
        l_loss = loss
        loss.backward()
        optimizer.step()
    return l_loss


def predict(x,w,b):
    return torch.matmul(x,w.reshape(-1,1)) + b