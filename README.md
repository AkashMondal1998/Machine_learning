# Minigrad
### Inspired by [Tinygrad](https://github.com/tinygrad/tinygrad) and [Micrograd](https://github.com/karpathy/micrograd)


### Mnist classification using MiniGrad
``` python
from minigrad import Tensor
import minigrad.nn as nn
from minigrad.nn import get_mnist,Adam
import numpy as np
from tqdm import trange


# Neural network 
class Net(nn.Base):
    def __init__(self):
        self.l1 = nn.Linear(784,128)
        self.l2 = nn.Linear(128,10)
        
    def forward(self,x):
        x = self.l1(x).relu()
        x = self.l2(x)
        return x

#load the mnsit dataset
X_train,Y_train,X_test,Y_test = get_mnist()

# sparse_categorical_crossentropy expects one hot encoded matrix
Y_one_hot = np.eye(np.max(Y_train) + 1)[Y_train]

# traning the model
epochs = 5
optimizer = Adam(model.parameters())
BS = 64
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


# evaluate the model
x_test = Tensor(self.X_test)
(np.argmax(model(x_test).log_softmax().data,axis=1) == Y_test).mean()
```