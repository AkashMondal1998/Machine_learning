# Minigrad



### Done
- Tensor (wrapper for numpy arrays)
- Dense Layer
- Forward pass
- Backward pass
- Sigmoid Activation 
- ReLU Activation 
- MSE loss
- BCE loss
- Optimizer(SGD)
- Auto udpate of parameters using optimizers



### Defining and training of a neural network using MiniGrad
``` python
from minigrad import Tensor
import minigrad.nn as nn

# Neural network using Layer
class Net(nn.Base):
    def __init__(self, in_features):
        self.l1 = nn.Linear(in_features, 3)
        self.l2 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.l1(x).relu()
        x = self.l2(x)
        return x


# Sample training data
x_train = Tensor.random_uniform(size=(10,3))
y_train = Tensor.random_unifrom(size=(10,1))

# Instantiate the network    
model = Net(x_train.shape[1])

#Forward pass using the model
model(x_train)

# loss function
loss_fn = nn.MSELoss()

# Traning the model
lr = 1e-3
epochs=100
for _ in range(epochs):
    model.zero_grad() # zero the gradients
    loss = loss_fn(model(x_train),y_train)
    loss.backward() 

    # upate the weights and bias
    model.l1.weight.numpy() -= lr * model.l1.grad
    model.l1.bias.numpy() -= lr * model.l1.grad
    model.l2.weight.numpy() -= lr * model.l2.grad
    model.l2.bias.numpy() -= lr * model.l2.grad
```