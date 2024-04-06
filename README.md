## Jupyter Notebooks for machine learning algorithms
### Models Done:
- Handwritten 1 or 0 prediction using MNIST DATASET
- Handwirtten digits prediction using MNIST DATASET (Tensorflow and Pytorch)
- Coffee Roasting Model (Tensorflow and Pytorch)


### Extra:
- Calculate gradient using auto back propagation


### Build neural network using numpy only
### Todo
- Back propagation
- Custom datatype for wrapping the 

### Done
- Neuron 
- Layer
- Forward pass
- Sigmoid Activation function
- ReLU Activation function
- MSE loss function
- BCE loss function



### Forward pass and loss calucation using neural_engine
``` python
from neural_engine import nn

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