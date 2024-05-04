from minigrad import Tensor
import minigrad.nn as nn
from minigrad.nn import DataSet,Adam
import numpy as np
from tqdm import trange
import math

class Net(nn.Base):
    def __init__(self):
        self.l1 = nn.Linear(784,200)
        self.l2 = nn.Linear(200,10)
        
    def forward(self,x):
        x = self.l1(x).relu()
        x = self.l2(x)
        return x


if __name__ == "__main__":
    dataset = DataSet("mnist")
    X_train,Y_train,X_test,Y_test = dataset.get_dataset()
    x_train,y_train,x_test = Tensor(X_train),Tensor(Y_train), Tensor(X_test)
    model = Net()
    optimizer = Adam(model.parameters())
    BS = 256
    num_iters = math.ceil(X_train.shape[0] / BS)

    for i in range(10):
        for _ in (t:=trange(num_iters)):
            samp = Tensor.randint(x_train.shape[0],size=(BS))
            optimizer.zero_grad()
            loss = model(x_train[samp]).sparse_categorical_crossentropy(y_train[samp])
            loss.backward()
            optimizer.step()
            accuracy = (np.argmax(model(x_train[samp]).log_softmax().data,axis=1) == y_train[samp].data).mean()
            t.set_description(f"epoch-->{i+1}  loss={loss.data:0.2f} accuracy={accuracy:0.2f}")
    
    print(f"accuracy on test set = {(np.argmax(model(x_test).log_softmax().data,axis=1) == Y_test).mean()*100:0.2f}")