from minigrad import Tensor
import minigrad.nn as nn
import numpy as np
from tqdm import trange
from minigrad.nn import get_mnist,Adam,SGD,RMSProp
import unittest


class Net(nn.Base):
    def __init__(self):
        self.l1 = nn.Linear(784,128)
        self.l2 = nn.Linear(128,10)
        
    def forward(self,x):
        x = self.l1(x).relu()
        x = self.l2(x)
        return x

def train(model,optimizer,epochs,BS,X_train,Y_one_hot,Y_train):
    num_iters = X_train.shape[0] // BS
    for i in range(epochs):
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


def evaluate(model,x_test,Y_test):
    return (np.argmax(model(x_test).log_softmax().data,axis=1) == Y_test).mean()

class TestMnist(unittest.TestCase):
    def setUp(self):
        self.X_train,self.Y_train,self.X_test,self.Y_test = get_mnist()
        self.Y_one_hot = np.eye(np.max(self.Y_train) + 1)[self.Y_train]
        self.model = Net()
        self.x_test = Tensor(self.X_test)

    def test_adam(self):
        optimizer = Adam(self.model.parameters())
        train(self.model,optimizer,5,64,self.X_train,self.Y_one_hot,self.Y_train)
        accuracy = evaluate(self.model,self.x_test,self.Y_test)
        self.assertGreaterEqual(accuracy,0.95,f"Model accuracy {accuracy}")

    def test_sgd(self):
        optimizer = SGD(self.model.parameters(),momentum=0.9)
        train(self.model,optimizer,5,64,self.X_train,self.Y_one_hot,self.Y_train)
        accuracy = evaluate(self.model,self.x_test,self.Y_test)
        self.assertGreaterEqual(accuracy,0.95,f"Model accuracy {accuracy}")
    
    def test_rms(self):
        optimizer = RMSProp(self.model.parameters())
        train(self.model,optimizer,5,64,self.X_train,self.Y_one_hot,self.Y_train)
        accuracy = evaluate(self.model,self.x_test,self.Y_test)
        self.assertGreaterEqual(accuracy,0.95,f"Model accuracy {accuracy}")

if __name__ == "__main__":
    unittest.main()