from minigrad import Tensor
import minigrad.nn as nn
from minigrad.nn import DataSet,Adam
import numpy as np
from tqdm import trange
import math

class Net(nn.Base):
    def __init__(self):
        self.l1 = nn.Linear(784,128)
        self.l2 = nn.Linear(128,10)
        
    def forward(self,x):
        x = self.l1(x).relu()
        x = self.l2(x)
        return x


if __name__ == "__main__":
    dataset = DataSet("fashion_mnist")
    X_train,Y_train,X_test,Y_test = dataset.get_dataset()
    Y_one_hot = np.eye(np.max(Y_train) + 1)[Y_train]
    x_test = Tensor(X_test)
    model = Net()
    optimizer = Adam(model.parameters())
    BS = 256
    num_iters = math.ceil(X_train.shape[0] / BS)

    for i in range(20):
        for _ in (t:=trange(num_iters)):
            samp = np.random.randint(0,X_train.shape[0],size=(BS))
            x = Tensor(X_train[samp])
            y = Tensor(Y_one_hot[samp])
            optimizer.zero_grad()
            loss = model(x).sparse_categorical_crossentropy(y)
            loss.backward()
            optimizer.step()
            accuracy = (np.argmax(model(x).log_softmax().data,axis=1) == Y_train[samp]).mean()
            t.set_description(f"epoch-->{i+1}  loss={loss.data:0.2f} accuracy={accuracy:0.2f}")
    
    print(f"accuracy = {(np.argmax(model(x_test).log_softmax().data,axis=1) == Y_test).mean()*100}")