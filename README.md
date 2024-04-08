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
import neural_engine.nn as nn

# Neural network using Layer
class Net(nn.Base):
    def __init__(self, in_features):
        self.l1 = nn.Layer(in_features, 3)
        self.relu = nn.ReLU()
        self.l2 = nn.Layer(3, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.sig(self.l2(x))
        return x

# Instantiate the network    
model = Net(x_train.shape[1])

# Loss function
loss_fn  = nn.BCELoss()
loss = loss_fn(model(x_train),y_train)
```