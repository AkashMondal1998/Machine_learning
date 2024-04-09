## Jupyter Notebooks for machine learning algorithms
### Models Done:
- Handwritten 1 or 0 prediction using MNIST DATASET
- Handwirtten digits prediction using MNIST DATASET (Tensorflow and Pytorch)
- Coffee Roasting Model (Tensorflow and Pytorch)

### Build neural network using numpy only
### Todo
- Back propagation
- Custom type for wrapping the numpy arrays

### Done
- Layer
- Forward pass
- Sigmoid Activation function
- ReLU Activation function
- MSE loss function
- BCE loss function



### Forward pass and loss calucation using neural_engine
``` python
from neural_engine import Tensor,random
import neural_engine.nn as nn

# Neural network using Layer
class Net(nn.Base):
    def __init__(self, in_features):
        self.l1 = nn.Layer(in_features, 3)
        self.l2 = nn.Layer(3, 1)

    def forward(self, x):
        x = (self.l1(x)).relu()
        x = (self.l2(x)).sigmoid()
        return x


# Create a Tensor
x_train = random.rand(10,3)

# Instantiate the network    
model = Net(x_train.shape[1])

#Forward pass using the model
model(x_train)
```