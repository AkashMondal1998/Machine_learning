import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

class LinearRegression(nn.Module):
    def __init__(self,in_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features,1)

    def forward(self,x):
        a1 = self.layer1(x)
        return a1


def train(epochs,x,y,model):
    optimizer = Adam(model.parameters())
    criterion = nn.MSELoss()
    p_bar = tqdm(total=epochs, desc="Training", unit="epoch")
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x),y.reshape(-1,1))
        loss.backward()
        optimizer.step()
        p_bar.set_postfix({"Loss": loss.item()})
        p_bar.update(1)
    p_bar.close()
